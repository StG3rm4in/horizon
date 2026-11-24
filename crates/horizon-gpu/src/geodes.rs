//! Batched classification of the cells of admitted vanilla geodes.
use crate::cuda::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use crate::{Context, context, load_module};
use horizon_core::noise::NoiseStack;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub(crate) const SOURCE: &str = concat!(include_str!("base.cu"), "\n", include_str!("geodes.cu"));

pub const GEODE_SIDE: usize = 33;
const CELLS: usize = GEODE_SIDE * GEODE_SIDE * GEODE_SIDE;
const MAX_BATCH: usize = 8;
const COALESCE: Duration = Duration::from_micros(50);
const POINTS: usize = 12;
const MAX_POINTS: usize = 20;
const CRACKS: usize = POINTS + MAX_POINTS * 4;
const NOISE: usize = CRACKS + 3 * 4;
const LAYER: usize = 5 + 32;
const STRIDE: usize = NOISE + 2 * LAYER;

pub const GEODE_EMPTY: u8 = 0;
pub const GEODE_FILL: u8 = 1;
pub const GEODE_CRACK: u8 = 2;
pub const GEODE_INNER: u8 = 3;
pub const GEODE_MIDDLE: u8 = 4;
pub const GEODE_OUTER: u8 = 5;

static BROKEN: AtomicBool = AtomicBool::new(false);

pub struct GeodeGeometry<'a> {
    pub origin: [i32; 3],
    pub offsets: [i32; 2],
    pub points: &'a [([i32; 3], i32)],
    pub crack_points: &'a [[i32; 3]],
    pub crack_offset: i32,
    pub multiplier: f64,
    /// Filling, inner, middle, outer, crack, in CPU evaluation order.
    pub thresholds: [f64; 5],
}

fn encode(shape: &GeodeGeometry<'_>, noise: &NoiseStack) -> Option<[i64; STRIDE]> {
    let within_bounds = |p: &[i32; 3]| p.iter().all(|v| (-30_001_000..=30_001_000).contains(v));
    if shape.offsets != [-16, 16]
        || !(1..=MAX_POINTS).contains(&shape.points.len())
        || !matches!(shape.crack_points.len(), 0 | 3)
        || !within_bounds(&shape.origin)
        || shape.points.iter().any(|(p, _)| !within_bounds(p))
        || shape.crack_points.iter().any(|p| !within_bounds(p))
        || !shape.multiplier.is_finite()
        || shape.thresholds.iter().any(|t| !t.is_finite())
        || noise.layers().len() != 2
    {
        return None;
    }

    let mut words = [0; STRIDE];
    words[..3].copy_from_slice(&shape.origin.map(i64::from));
    words[3] = shape.points.len() as i64;
    words[4] = shape.crack_points.len() as i64;
    words[5] = i64::from(shape.crack_offset);
    words[6] = shape.multiplier.to_bits() as i64;
    words[7..POINTS].copy_from_slice(&shape.thresholds.map(|v| v.to_bits() as i64));

    for (row, (point, offset)) in words[POINTS..CRACKS]
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(shape.points)
    {
        row[..3].copy_from_slice(&point.map(i64::from));
        row[3] = i64::from(*offset);
    }
    for (row, point) in words[CRACKS..NOISE]
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(shape.crack_points)
    {
        row[..3].copy_from_slice(&point.map(i64::from));
    }
    for (row, layer) in words[NOISE..]
        .as_chunks_mut::<LAYER>()
        .0
        .iter_mut()
        .zip(noise.layers())
    {
        let table = layer.noise.as_perlin()?.table();
        // The kernel does not wrap: up to 0.5, |coordinate × frequency| stays below 2^24 everywhere in
        // the world (30,000,000 blocks plus a geode's reach), where `PerlinNoise::wrap` is the identity.
        if !(0.0..=0.5).contains(&layer.frequency)
            || !layer.amplitude.is_finite()
            || table.offsets().iter().any(|v| !(0.0..=256.0).contains(v))
        {
            return None;
        }
        row[0] = layer.frequency.to_bits() as i64;
        row[1] = i64::from(layer.amplitude.to_bits() as i32);
        row[2..5].copy_from_slice(&table.offsets().map(|v| v.to_bits() as i64));
        for (word, bytes) in row[5..]
            .iter_mut()
            .zip(table.permutations().as_chunks::<8>().0)
        {
            *word = i64::from_le_bytes(*bytes);
        }
    }
    Some(words)
}

pub struct GeodeClasses {
    batch: Arc<Vec<u8>>,
    index: usize,
}
impl GeodeClasses {
    pub fn cells(&self) -> &[u8] {
        &self.batch[self.index * CELLS..(self.index + 1) * CELLS]
    }
}

struct Request {
    words: [i64; STRIDE],
    reply: mpsc::Sender<GeodeClasses>,
}

fn collect_batch<T>(inbox: &mpsc::Receiver<T>, batch: &mut Vec<T>, first: T, max: usize) {
    batch.push(first);
    let deadline = Instant::now() + COALESCE;
    while batch.len() < max {
        // Drain arrivals already queued even when the coalescing window has elapsed.
        match inbox.try_recv() {
            Ok(job) => batch.push(job),
            Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match inbox.recv_timeout(remaining) {
                    Ok(job) => batch.push(job),
                    Err(_) => break,
                }
            }
        }
    }
}

#[derive(Default)]
struct Stats {
    batches: AtomicU64,
    geodes: AtomicU64,
    singletons: AtomicU64,
    admitted: AtomicU64,
    fallbacks: AtomicU64,
    failures: AtomicU64,
    checked: AtomicU64,
    mismatches: AtomicU64,
    request_ns: AtomicU64,
    launch_ns: AtomicU64,
}

struct Worker {
    stream: Arc<CudaStream>,
    function: CudaFunction,
    input: CudaSlice<i64>,
    output: CudaSlice<u8>,
}

impl Worker {
    fn classify(&mut self, image: &[i64], count: usize) -> Result<Vec<u8>, String> {
        let mut cells = vec![0; count * CELLS];
        let stream = &self.stream;
        let transfer: Result<(), String> = (|| {
            stream
                .memcpy_htod(image, &mut self.input.slice_mut(0..image.len()))
                .ctx("geode upload")?;
            let mut launch = stream.launch_builder(&self.function);
            launch.arg(&self.input).arg(&mut self.output);
            // SAFETY: each thread handles one cell of one uploaded record. Both
            // allocations cover the launch and stay live through the stream wait.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (CELLS.div_ceil(128) as u32, count as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .ctx("geode kernel")?;
            stream
                .memcpy_dtoh(&self.output.slice(0..cells.len()), &mut cells[..])
                .ctx("geode readback")?;
            Ok(())
        })();
        // Event tracking is disabled on the shared context. Keep both host
        // buffers alive through the wait, including on a partial submission.
        let wait = stream.synchronize().ctx("geode wait");
        transfer?;
        wait?;
        if cells.iter().any(|&class| class > GEODE_OUTER) {
            return Err("invalid geode cell class".into());
        }
        Ok(cells)
    }

    fn run(mut self, inbox: mpsc::Receiver<Request>, max: usize, stats: &Stats) {
        let mut batch = Vec::with_capacity(max);
        let mut image = Vec::with_capacity(max * STRIDE);
        while let Ok(first) = inbox.recv() {
            if BROKEN.load(Relaxed) {
                break;
            }
            collect_batch(&inbox, &mut batch, first, max);
            image.clear();
            for request in &batch {
                image.extend_from_slice(&request.words);
            }
            let start = Instant::now();
            let result = self.classify(&image, batch.len());
            stats
                .launch_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
            match result {
                Ok(cells) => {
                    let cells = Arc::new(cells);
                    stats.batches.fetch_add(1, Relaxed);
                    stats.geodes.fetch_add(batch.len() as u64, Relaxed);
                    if batch.len() == 1 {
                        stats.singletons.fetch_add(1, Relaxed);
                    }
                    for (index, request) in batch.drain(..).enumerate() {
                        let _ = request.reply.send(GeodeClasses {
                            batch: Arc::clone(&cells),
                            index,
                        });
                    }
                }
                Err(error) => {
                    stats.failures.fetch_add(1, Relaxed);
                    BROKEN.store(true, Relaxed);
                    eprintln!("GPU geodes: {error}; using CPU for the rest of the process");
                    // Dropping the current and queued reply senders wakes all callers.
                    break;
                }
            }
        }
    }
}

pub struct GeodeGpu {
    sender: Option<mpsc::SyncSender<Request>>,
    worker: Option<JoinHandle<()>>,
    stats: Arc<Stats>,
    pub test: bool,
}

impl GeodeGpu {
    pub fn new(test: bool) -> Result<Self, String> {
        if BROKEN.load(Relaxed) {
            return Err("a previous geode batch failed".into());
        }
        let ctx = context()?;
        static ARCH: OnceLock<&'static str> = OnceLock::new();
        let (major, minor) = ctx.compute_capability().ctx("GPU compute capability")?;
        let arch = *ARCH.get_or_init(|| Box::leak(format!("sm_{major}{minor}").into_boxed_str()));
        let module = load_module(&ctx, SOURCE, arch).ctx("geode module")?;
        let function = module.load_function("kgeodes").ctx("kgeodes")?;
        let max = std::env::var("HORIZON_GPU_GEODES_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(MAX_BATCH)
            .clamp(1, MAX_BATCH);
        let stream = ctx.new_stream().ctx("geode stream")?;
        let input = stream.alloc_zeros::<i64>(max * STRIDE).ctx("geode input")?;
        let output = stream.alloc_zeros::<u8>(max * CELLS).ctx("geode output")?;
        stream.synchronize().ctx("geode buffer wait")?;
        let device = Worker {
            stream,
            function,
            input,
            output,
        };

        let (sender, inbox) = mpsc::sync_channel(max * 2);
        let stats = Arc::new(Stats::default());
        let worker_stats = Arc::clone(&stats);
        let worker = thread::Builder::new()
            .name("gpu-geodes".into())
            .spawn(move || device.run(inbox, max, &worker_stats))
            .ctx("geode worker")?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
            stats,
            test,
        })
    }

    pub fn classify(&self, shape: &GeodeGeometry<'_>, noise: &NoiseStack) -> Option<GeodeClasses> {
        self.stats.admitted.fetch_add(1, Relaxed);
        let start = Instant::now();
        let output = (|| {
            if BROKEN.load(Relaxed) {
                return None;
            }
            let words = encode(shape, noise)?;
            let (reply, result) = mpsc::channel();
            self.sender.as_ref()?.send(Request { words, reply }).ok()?;
            result.recv().ok()
        })();
        if output.is_none() {
            self.stats.fallbacks.fetch_add(1, Relaxed);
        } else {
            self.stats
                .request_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
        }
        output
    }

    pub fn note_comparison(&self, checked: usize, mismatches: usize) {
        self.stats.checked.fetch_add(checked as u64, Relaxed);
        self.stats.mismatches.fetch_add(mismatches as u64, Relaxed);
        if mismatches != 0 {
            self.stats.fallbacks.fetch_add(1, Relaxed);
        }
    }
}

impl Drop for GeodeGpu {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        let stats = &self.stats;
        let batches = stats.batches.load(Relaxed);
        let geodes = stats.geodes.load(Relaxed);
        eprintln!(
            "GPU geodes: {batches} batches, {geodes} GPU geodes, {} singletons, {} admitted, \
             {} CPU fallbacks, {} batches failed, {} cells checked, {} mismatches with the CPU; \
             {} us per batch (upload/kernel/readback), {} us per successful request",
            stats.singletons.load(Relaxed),
            stats.admitted.load(Relaxed),
            stats.fallbacks.load(Relaxed),
            stats.failures.load(Relaxed),
            stats.checked.load(Relaxed),
            stats.mismatches.load(Relaxed),
            stats.launch_ns.load(Relaxed) / batches.max(1) / 1_000,
            stats.request_ns.load(Relaxed) / geodes.max(1) / 1_000,
        );
    }
}

#[cfg(test)]
mod geodes_tests {
    use super::*;
    use horizon_core::{noise::NormalNoise, random::LegacyRandom};

    #[test]
    fn batch_cap_and_fifo() {
        let (send, recv) = mpsc::channel();
        for value in 0..10 {
            send.send(value).unwrap();
        }
        drop(send);
        let mut batch = Vec::new();
        collect_batch(&recv, &mut batch, recv.recv().unwrap(), 8);
        assert_eq!(batch, [0, 1, 2, 3, 4, 5, 6, 7]);
        batch.clear();
        collect_batch(&recv, &mut batch, recv.recv().unwrap(), 8);
        assert_eq!(batch, [8, 9]);
        assert!(recv.recv().is_err());
    }

    #[test]
    fn singleton_leaves_pending_requests_for_the_next_launch() {
        let (send, recv) = mpsc::channel();
        send.send(11).unwrap();
        send.send(12).unwrap();
        let mut batch = Vec::new();
        collect_batch(&recv, &mut batch, recv.recv().unwrap(), 1);
        assert_eq!(batch, [11]);
        assert_eq!(recv.try_recv().unwrap(), 12);
    }

    #[test]
    fn layout_uses_signed_coordinates_and_original_noise_bits() {
        let noise = NormalNoise::create_parity(-4, &[1.0])
            .unwrap()
            .create(&mut LegacyRandom::new(12345))
            .unwrap();
        let mut shape = GeodeGeometry {
            origin: [-30_000_000, -7, 30_000_000],
            offsets: [-16, 16],
            points: &[([-29_999_995, -2, 30_000_005], 2)],
            crack_points: &[],
            crack_offset: 2,
            multiplier: 0.05,
            thresholds: [1.0, 0.9, 0.8, 0.7, 0.6],
        };
        let packed = encode(&shape, &noise).unwrap();
        assert_eq!(
            &packed[..7],
            &[
                -30_000_000,
                -7,
                30_000_000,
                1,
                0,
                2,
                0.05_f64.to_bits() as i64
            ]
        );
        assert_eq!(
            &packed[POINTS..POINTS + 4],
            &[-29_999_995, -2, 30_000_005, 2]
        );
        for (i, layer) in noise.layers().iter().enumerate() {
            let base = NOISE + i * LAYER;
            let perm = layer.noise.as_perlin().unwrap().table().permutations();
            assert_eq!(packed[base] as u64, layer.frequency.to_bits());
            assert_eq!(packed[base + 1] as u32, layer.amplitude.to_bits());
            for (word, bytes) in packed[base + 5..base + LAYER]
                .iter()
                .zip(perm.as_chunks::<8>().0)
            {
                assert_eq!(word.to_le_bytes(), *bytes);
            }
        }
        shape.offsets = [-16, 17];
        assert!(encode(&shape, &noise).is_none());
        shape.offsets = [-16, 16];
        shape.origin[0] = i32::MIN;
        assert!(encode(&shape, &noise).is_none());
    }
}
