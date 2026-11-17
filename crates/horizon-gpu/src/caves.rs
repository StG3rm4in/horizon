//! Carver mask on the GPU (`HORIZON_GPU_CAVES`): see CAVES.md.
//!
//! The CPU plans the carvers of each origin once (the draws that do not depend on the target chunk)
//! and sends, at the start of a chunk, the elements that can reach it. The GPU traces the tunnels
//! and canyons and rasterizes their ellipsoids into the chunk's mask (`kcv_trace`), then, as soon
//! as the chunk's region is on the GPU, computes the substance of each carved cell with the aquifer
//! (`kcv_substance`, in the aquifer module). Everything runs on the batch's stream, off the
//! worker's critical path; the CPU reads the result when it carves.
use super::{Context, context, load_module};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const CAVE_ROOM: u32 = 0;
pub const CAVE_TUNNEL: u32 = 1;
pub const CAVE_CANYON: u32 = 2;
/// Mask words at most (256 columns of 384 cells: 1,536 words)
pub const CAVE_MASK_WORDS: usize = 1536;

/// A planned carver element for a target chunk, as `kcv_trace` reads it (16 words).
#[derive(Clone, Copy, Debug, Default)]
pub struct CaveItem {
    /// Tunnel: seed of its LegacyRandom; canyon: state of its random source after the width table
    pub seed: u64,
    pub position: [f64; 3],
    /// Tunnel: vertical_scale; canyon: y_scale
    pub scale: f64,
    /// Tunnel: radius multipliers of the run; room: its radii
    pub horizontal: f64,
    pub vertical: f64,
    /// Cave: floor_level
    pub floor: f64,
    pub kind: u32,
    /// Carver call (budget of 2^24 tested cells)
    pub call: u32,
    pub end: i32,
    /// Canyon: start of its table in the batch's widths
    pub width: u32,
    pub thickness: f32,
    pub yaw: f32,
    pub pitch: f32,
    /// Canyon, horizontal_radius_factor: 0 constant (`factor[0]`), 1 uniform
    pub factor_kind: u32,
    pub factor: [f32; 2],
    /// Canyon: vertical_radius_default_factor, vertical_radius_center_factor
    pub vertical_factors: [f32; 2],
}
impl CaveItem {
    fn words(&self) -> [u64; 16] {
        let f = |a: f32, b: f32| u64::from(a.to_bits()) | (u64::from(b.to_bits()) << 32);
        let u = |a: u32, b: u32| u64::from(a) | (u64::from(b) << 32);
        [
            self.seed,
            self.position[0].to_bits(),
            self.position[1].to_bits(),
            self.position[2].to_bits(),
            self.scale.to_bits(),
            self.horizontal.to_bits(),
            self.vertical.to_bits(),
            self.floor.to_bits(),
            u(self.kind, self.call),
            u(self.end as u32, self.width),
            f(self.thickness, self.yaw),
            u(self.pitch.to_bits(), self.factor_kind),
            f(self.factor[0], self.factor[1]),
            f(self.vertical_factors[0], self.vertical_factors[1]),
            0,
            0,
        ]
    }
}

/// A chunk's mask: CarvingMask bounds (ymin, ymax) and the generation minimum.
#[derive(Clone, Copy, Debug)]
pub struct CaveTarget {
    pub chunk: [i32; 2],
    pub min_y: i32,
    pub max_y: i32,
    pub generation_min_y: i32,
    /// Length of a canyon's width table (generation height)
    pub generation_height: i32,
}

#[derive(Default)]
struct CaveStats {
    batches: AtomicU64,
    elements: AtomicU64,
    cells: AtomicU64,
    max_cells: AtomicU64,
    ns_send: AtomicU64,
    ns_wait: AtomicU64,
    substances: AtomicU64,
    fallbacks: AtomicU64,
    errors: AtomicU64,
    mask_mismatches: AtomicU64,
    substance_mismatches: AtomicU64,
}

pub struct CaveGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<cudarc::driver::CudaModule>,
    k_trace: CudaFunction,
    sines: CudaSlice<f32>,
    pool: Mutex<Vec<CaveResources>>,
    /// Substance codes copied per batch (bytes), grown if a chunk has more
    cap_codes: AtomicU64,
    stats: CaveStats,
}
/// A single buffer per batch, in 64-bit words: elements (16 words each), widths (2 per word), then
/// the output: mask, counters, flags (zeroed), then the substance codes (8 per word). The upload
/// copies elements, widths and zeros at once; the readback copies output and codes at once.
struct CaveResources {
    st: Arc<CudaStream>,
    /// Capacity in words
    cap: usize,
    buffer: CudaSlice<u64>,
    h_input: PinnedHostSlice<u64>,
    h_output: PinnedHostSlice<u64>,
}

/// The batch of a chunk: sent by `CaveGpu::submit`, completed by the GPU aquifer when the chunk is
/// filled, read by `CaveGpu::wait` when carving, returned by `CaveGpu::finish`.
pub struct CaveJob {
    res: Option<CaveResources>,
    target: CaveTarget,
    nw: usize,
    nc: usize,
    /// Start of the output in the buffer
    e: usize,
    /// Code words the buffer can receive
    cw: usize,
    height: i32,
    /// Substance codes requested (0: no GPU aquifer for this chunk)
    codes: usize,
    expected: bool,
    ok: bool,
    /// The region whose aquifer is read, kept until the end of the batch's stream
    region: Option<Arc<dyn Any + Send + Sync>>,
}
impl Drop for CaveJob {
    fn drop(&mut self) {
        // The stream's copies write into these buffers: they are only released after them
        if let Some(r) = &self.res
            && !self.expected
        {
            let _ = r.st.synchronize();
        }
    }
}
impl std::fmt::Debug for CaveGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CaveGpu")
    }
}
impl Drop for CaveGpu {
    fn drop(&mut self) {
        let s = &self.stats;
        if let Some(n) = std::num::NonZero::new(s.batches.load(Relaxed)) {
            eprintln!(
                "GPU caves: {n} batches, {} elements and {} cells carved per batch (max {}), upload {} us, wait {} us per batch; {} with the GPU aquifer; {} chunks fell back to the CPU, {} batches failed; mismatches with the CPU (test mode): {} masks, {} substances",
                s.elements.load(Relaxed) / n,
                s.cells.load(Relaxed) / n,
                s.max_cells.load(Relaxed),
                s.ns_send.load(Relaxed) / n / 1000,
                s.ns_wait.load(Relaxed) / n / 1000,
                s.substances.load(Relaxed),
                s.fallbacks.load(Relaxed),
                s.errors.load(Relaxed),
                s.mask_mismatches.load(Relaxed),
                s.substance_mismatches.load(Relaxed),
            );
        }
    }
}

fn config(nb: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (nb as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl CaveGpu {
    /// `sine`: Horizon's sine table (`horizon_core::geometry::trig::sine_table`).
    pub fn new(sine: &[f32]) -> Result<Self, String> {
        if sine.len() != 65536 {
            return Err("sine table".into());
        }
        let ctx = context()?;
        let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
        let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
        let module = load_module(&ctx, include_str!("caves.cu"), arch).ctx("cave module")?;
        let k_trace = module.load_function("kcv_trace").ctx("kcv_trace")?;
        let st = ctx.new_stream().ctx("CUDA stream")?;
        let sines = st.clone_htod(sine).ctx("sine table")?;
        st.synchronize().ctx("sine table")?;
        Ok(Self {
            ctx,
            _module: module,
            k_trace,
            sines,
            pool: Mutex::new(Vec::new()),
            cap_codes: AtomicU64::new(16384),
            stats: CaveStats::default(),
        })
    }
    /// A chunk whose caves go through the CPU (failed batch, budget, missing substance).
    pub fn note_cpu_fallback(&self) {
        self.stats.fallbacks.fetch_add(1, Relaxed);
    }
    /// Test mode: a mask or substances that differ from the CPU.
    pub fn note_mismatch(&self, mask: bool) {
        if mask {
            self.stats.mask_mismatches.fetch_add(1, Relaxed);
        } else {
            self.stats.substance_mismatches.fetch_add(1, Relaxed);
        }
    }
    fn resource(&self, needed: usize) -> Result<CaveResources, String> {
        let mut pool = self.pool.lock().unwrap();
        if let Some(k) = pool.iter().position(|r| r.cap >= needed) {
            return Ok(pool.swap_remove(k));
        }
        let previous = pool.pop();
        drop(pool);
        let cap = needed
            .max(8192)
            .max(previous.as_ref().map_or(0, |r| r.cap))
            .next_power_of_two();
        let st = match previous {
            Some(r) => r.st.clone(),
            None => self.ctx.new_stream().ctx("CUDA stream")?,
        };
        let pinned =
            |n: usize| unsafe { self.ctx.alloc_pinned_with_flags::<u64>(n, 0) }.ctx("host buffer");
        Ok(CaveResources {
            buffer: st.alloc_zeros::<u64>(cap).ctx("cave buffer")?,
            h_input: pinned(cap)?,
            h_output: pinned(cap)?,
            st,
            cap,
        })
    }
    /// Sends the elements of a chunk (`calls` carver calls) and launches their trace on the batch's
    /// stream.
    pub fn submit(
        &self,
        target: CaveTarget,
        items: &[CaveItem],
        widths: &[f32],
        calls: usize,
    ) -> Result<CaveJob, String> {
        let t = Instant::now();
        let height = target.max_y - target.min_y + 1;
        if !(1..=384).contains(&height) {
            return Err("mask height".into());
        }
        let nw = (256 * height as usize).div_ceil(64);
        let (ni, nl) = (16 * items.len(), widths.len().div_ceil(2));
        let e = ni + nl;
        let head = nw + calls + 1;
        let cw = (self.cap_codes.load(Relaxed) as usize).div_ceil(8);
        let mut res = self.resource(e + head + cw)?;
        let st = res.st.clone();
        {
            let h = res.h_input.as_mut_slice().ctx("host buffer")?;
            for (m, it) in h[..ni].as_chunks_mut::<16>().0.iter_mut().zip(items) {
                m.copy_from_slice(&it.words());
            }
            for (m, w) in h[ni..e].iter_mut().zip(widths.chunks(2)) {
                *m = u64::from(w[0].to_bits())
                    | (u64::from(w.get(1).map_or(0, |v| v.to_bits())) << 32);
            }
            h[e..e + head].fill(0);
        }
        st.memcpy_htod(
            &res.h_input.as_slice().ctx("host buffer")?[..e + head],
            &mut res.buffer.slice_mut(0..e + head),
        )
        .ctx("element upload")?;
        if !items.is_empty() {
            let (nwi, nci, h) = (nw as i32, calls as i32, height);
            let [cx, cz] = target.chunk;
            let nwidth = target.generation_height;
            let input = res.buffer.slice(0..ni);
            let widths_d = res.buffer.slice(ni..e.max(ni + 1));
            let output = res.buffer.slice(e..e + head);
            let mut l = st.launch_builder(&self.k_trace);
            l.arg(&input).arg(&widths_d).arg(&self.sines).arg(&output);
            l.arg(&nwi)
                .arg(&nci)
                .arg(&cx)
                .arg(&cz)
                .arg(&target.min_y)
                .arg(&target.max_y)
                .arg(&h);
            l.arg(&target.generation_min_y).arg(&nwidth);
            unsafe { l.launch(config(items.len(), 128)) }.ctx("carver trace")?;
        }
        self.stats.batches.fetch_add(1, Relaxed);
        self.stats.elements.fetch_add(items.len() as u64, Relaxed);
        self.stats
            .ns_send
            .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        Ok(CaveJob {
            res: Some(res),
            target,
            nw,
            nc: calls,
            e,
            cw,
            height,
            codes: 0,
            expected: false,
            ok: true,
            region: None,
        })
    }
    /// Waits for the batch (once) and copies its mask if not done yet; `false` if the GPU failed.
    pub fn wait(&self, job: &mut CaveJob) -> bool {
        if job.expected {
            return job.ok;
        }
        let t = Instant::now();
        job.ok = (|| -> Result<(), String> {
            let res = job.res.as_mut().ok_or("batch without buffers")?;
            if job.codes == 0 {
                let (e, n) = (job.e, job.nw + job.nc + 1);
                res.st
                    .memcpy_dtoh(
                        &res.buffer.slice(e..e + n),
                        &mut res.h_output.as_mut_slice().ctx("host buffer")?[..n],
                    )
                    .ctx("mask readback")?;
            }
            res.st.synchronize().ctx("cave wait")
        })()
        .map_err(|e| {
            self.stats.errors.fetch_add(1, Relaxed);
            log::error!("GPU caves: chunk {:?}: {e}", job.target.chunk);
        })
        .is_ok();
        job.expected = true;
        job.region = None;
        self.stats
            .ns_wait
            .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        job.ok
    }
    /// The mask of a waited batch (CarvingMask words), or None if the GPU flagged an error or a
    /// carver call exceeds its budget (the CPU then raises its error itself).
    pub fn mask<'a>(&self, job: &'a CaveJob) -> Option<&'a [u64]> {
        if !job.expected || !job.ok {
            return None;
        }
        let s = job.res.as_ref()?.h_output.as_slice().ok()?;
        let (nw, nc) = (job.nw, job.nc);
        if s[nw + nc] != 0 || s[nw..nw + nc].iter().any(|&c| c > 1 << 24) {
            self.stats.errors.fetch_add(1, Relaxed);
            return None;
        }
        let m = &s[..nw];
        let n: u64 = m.iter().map(|w| u64::from(w.count_ones())).sum();
        self.stats.cells.fetch_add(n, Relaxed);
        self.stats.max_cells.fetch_max(n, Relaxed);
        Some(m)
    }
    /// The substances of the carved cells of a waited batch, in mask bit order (GPU aquifer codes:
    /// 1 air, 2 fluid, 3 lava, 8 kept block, + 4 scheduled update), or None if the GPU aquifer was
    /// not launched, the trace reported an error, or the mask has more cells than the copied
    /// codes. The codes follow the GPU's mask: the caller uses them only with that mask.
    pub fn codes<'a>(&self, job: &'a CaveJob, cells: usize) -> Option<&'a [i8]> {
        if !job.expected || !job.ok || job.codes == 0 {
            return None;
        }
        let s = job.res.as_ref()?.h_output.as_slice().ok()?;
        if s[job.nw + job.nc] != 0 {
            return None;
        }
        if cells > job.codes {
            self.cap_codes
                .fetch_max((cells as u64).next_power_of_two(), Relaxed);
            return None;
        }
        let head = job.nw + job.nc + 1;
        let words = &job.res.as_ref()?.h_output.as_slice().ok()?[head..head + job.cw];
        // The codes are the bytes of the words after the output
        let c = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const i8, cells) };
        if c.iter().any(|&v| v < 0) {
            return None;
        }
        self.stats.substances.fetch_add(1, Relaxed);
        Some(c)
    }
    /// Returns the buffers of a batch, after waiting for it if not done yet.
    pub fn finish(&self, mut job: CaveJob) {
        self.wait(&mut job);
        if job.ok
            && let Some(r) = job.res.take()
        {
            self.pool.lock().unwrap().push(r);
        }
    }
}

/// The aquifer of a region, for the substance of the carved cells of one of its chunks.
pub(crate) struct AquiferRegion<'a> {
    pub f: &'a CudaFunction,
    pub p: &'a CudaSlice<i32>,
    pub st: &'a CudaSlice<i32>,
    pub grid: [i32; 6],
    pub skip: i32,
    pub sea: i32,
}
impl CaveJob {
    /// Enqueues on the batch's stream, after its trace, the substance of its carved cells and the
    /// copy of the result. `region` keeps the aquifer's memory alive until the end of the stream.
    pub(crate) fn enqueue_substance(
        &mut self,
        aq: AquiferRegion<'_>,
        region: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), String> {
        if self.expected || self.codes != 0 {
            return Ok(());
        }
        let res = self.res.as_mut().ok_or("batch without buffers")?;
        let st = res.st.clone();
        let (e, head, cw) = (self.e, self.nw + self.nc + 1, self.cw);
        let cap = 8 * cw;
        let (nw, ci, h) = (self.nw as i32, cap as i32, self.height);
        let (x0, z0) = (self.target.chunk[0] * 16, self.target.chunk[1] * 16);
        let g = aq.grid;
        let mask = res.buffer.slice(e..e + self.nw);
        let codes = res.buffer.slice(e + head..e + head + cw);
        let mut l = st.launch_builder(aq.f);
        l.arg(&mask)
            .arg(&codes)
            .arg(&nw)
            .arg(&ci)
            .arg(&self.target.min_y)
            .arg(&h);
        l.arg(&x0).arg(&z0).arg(aq.p).arg(aq.st).arg(&aq.skip);
        l.arg(&g[0])
            .arg(&g[1])
            .arg(&g[2])
            .arg(&g[3])
            .arg(&g[4])
            .arg(&g[5])
            .arg(&aq.sea);
        unsafe { l.launch(config(self.nw.div_ceil(32), 256)) }.ctx("cave substance")?;
        st.memcpy_dtoh(
            &res.buffer.slice(e..e + head + cw),
            &mut res.h_output.as_mut_slice().ctx("host buffer")?[..head + cw],
        )
        .ctx("mask and substance readback")?;
        self.codes = cap;
        self.region = Some(region);
        Ok(())
    }
    pub fn target(&self) -> CaveTarget {
        self.target
    }
}
