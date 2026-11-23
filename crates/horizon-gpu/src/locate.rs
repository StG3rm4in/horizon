//! Biome searches on the GPU (`HORIZON_GPU_LOCATE`): the climate and the climate tree search of
//! a list of quarts at once, for `BiomeSource.findClosestBiome3d` (`/locate biome`, the mods'
//! searches) and the map previews (`MapItem.renderBiomePreviewMap`), all on the tick thread.
//!
//! Each of these walks its quarts in an order of its own and keeps the previous leaf of one
//! climate search from a quart to the next (`ClimateSearch::last`), so the GPU does not give the
//! biomes themselves. It gives, per quart, the six climate fields and the leaf that a search
//! without a previous leaf finds (`kbl_search`), and the caller replays its own order on the CPU
//! with `ClimateSearch::find_entry_from_first`, which answers what its own search would. Other
//! callers with an order of their own (the stronghold rings' squares, with their draws) can use
//! the same two calls. See locate.cu.
use super::{Context, codegen, context, load_module, priority_stream};
use crate::cuda::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice,
    PushKernelArg,
};
use horizon_core::density::graph::{DensityCompiler, FunctionId};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Flags of `locate.cu`: traversal stack too short, search without a result. The batch is then
/// left to the CPU, which searches every quart itself.
const E_STACK: i32 = 1;
const E_EMPTY: i32 = 2;

/// Quarts in one batch at most: 524,288, whose fields and leaves come back in 14.7 MB.
pub const LOCATE_BATCH: usize = 1 << 19;

/// Per quart of a batch, in the caller's order: the six climate fields as `ClimateSampler::sample`
/// computes them before quantizing (temperature, humidity, continentalness, erosion, depth,
/// weirdness), and the node of the first leaf at the smallest distance from the quantized target,
/// in `ClimateIndex::tree` numbering.
#[derive(Debug, Default)]
pub struct LocateSamples {
    pub fields: Vec<[f32; 6]>,
    pub first: Vec<u32>,
}

/// One batch's buffers, at the largest batch: the columns' quarts, the Y quarts, the fields and
/// the leaves on the GPU, the fields and the leaves in pinned memory for their readback.
struct Resources {
    st: Arc<CudaStream>,
    xz: CudaSlice<i32>,
    qy: CudaSlice<i32>,
    qf: CudaSlice<f32>,
    best: CudaSlice<i32>,
    err: CudaSlice<i32>,
    h_qf: PinnedHostSlice<f32>,
    h_best: PinnedHostSlice<i32>,
    h_err: PinnedHostSlice<i32>,
}

#[derive(Default)]
struct Stats {
    searches: AtomicU64,
    batches: AtomicU64,
    quarts: AtomicU64,
    ns: AtomicU64,
    ns_max: AtomicU64,
    errors: AtomicU64,
    fallbacks: AtomicU64,
    compared: AtomicU64,
    mismatches: AtomicU64,
}

pub struct LocateGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    k_climate: CudaFunction,
    k_search: CudaFunction,
    e: CudaSlice<i64>,
    l: CudaSlice<u32>,
    pool: Mutex<Vec<Resources>>,
    /// A CUDA error turns the searches back to the CPU for the session; a flag of the search
    /// (a tree too deep) only sends its batch there.
    broken: AtomicBool,
    stats: Stats,
}
impl std::fmt::Debug for LocateGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LocateGpu")
    }
}
impl Drop for LocateGpu {
    fn drop(&mut self) {
        let s = &self.stats;
        if let Some(n) = std::num::NonZero::new(s.batches.load(Relaxed)) {
            eprintln!(
                "GPU locate: {} searches, {n} batches of {} quarts in all, {} us per batch \
                 (max {}); {} searches by the CPU; {} batches failed; test mode: {} compared, \
                 {} mismatches",
                s.searches.load(Relaxed),
                s.quarts.load(Relaxed),
                s.ns.load(Relaxed) / n / 1000,
                s.ns_max.load(Relaxed) / 1000,
                s.fallbacks.load(Relaxed),
                s.errors.load(Relaxed),
                s.compared.load(Relaxed),
                s.mismatches.load(Relaxed),
            );
        }
    }
}

fn config(n: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl LocateGpu {
    /// `roots`: the six climate roots of the router (temperature, vegetation, continents,
    /// erosion, depth, ridges); `spaces` and `links`: the source's `ClimateIndex::tree`.
    pub fn new(
        compiler: &DensityCompiler,
        roots: [FunctionId; 6],
        seed: i64,
        spaces: &[[i64; 13]],
        links: &[(u32, u32)],
    ) -> Result<Self, String> {
        let n = links.len();
        if n == 0 || spaces.len() != n {
            return Err("climate tree".into());
        }
        // A branch's children must be nodes; a leaf's entry is only read by the CPU.
        if links
            .iter()
            .any(|&(a, k)| k != 0 && a as usize + k as usize > n)
        {
            return Err("climate tree links".into());
        }
        let src = codegen::generate_locate(compiler, roots, seed).map_err(|e| e.to_string())?;
        if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP_LOCATE") {
            let _ = std::fs::write(f, &src);
        }
        let ctx = context()?;
        let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
        let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
        let module = load_module(&ctx, &src, arch).ctx("locate module")?;
        let k_climate = module.load_function("kbl_climate").ctx("kbl_climate")?;
        let k_search = module.load_function("kbl_search").ctx("kbl_search")?;
        let st = ctx.new_stream().ctx("CUDA stream")?;
        let flat: Vec<i64> = spaces.iter().flatten().copied().collect();
        let e = st.clone_htod(&flat).ctx("climate tree")?;
        let flat: Vec<u32> = links.iter().flat_map(|&(a, k)| [a, k]).collect();
        let l = st.clone_htod(&flat).ctx("tree links")?;
        st.synchronize().ctx("climate tree")?;
        Ok(Self {
            ctx,
            _module: module,
            k_climate,
            k_search,
            e,
            l,
            pool: Mutex::new(Vec::new()),
            broken: AtomicBool::new(false),
            stats: Stats::default(),
        })
    }
    /// A search or a map preview answered from the GPU's batches.
    pub fn note_search(&self) {
        self.stats.searches.fetch_add(1, Relaxed);
    }
    /// A search the CPU made instead, after a failed batch.
    pub fn note_fallback(&self) {
        self.stats.fallbacks.fetch_add(1, Relaxed);
    }
    /// Test mode: an answer compared with the CPU's; returns the number compared so far.
    pub fn note_test(&self, gap: bool) -> u64 {
        if gap {
            self.stats.mismatches.fetch_add(1, Relaxed);
        }
        self.stats.compared.fetch_add(1, Relaxed) + 1
    }
    /// Mismatches with the CPU seen in test mode.
    pub fn mismatches(&self) -> u64 {
        self.stats.mismatches.load(Relaxed)
    }
    fn resource(&self) -> Result<Resources, String> {
        if let Some(r) = self.pool.lock().unwrap().pop() {
            return Ok(r);
        }
        // The tick thread waits for the answer: the critical path's priority.
        let st = priority_stream(&self.ctx)?;
        let n = LOCATE_BATCH;
        let pinned_i32 =
            |n: usize| unsafe { self.ctx.alloc_pinned_with_flags::<i32>(n, 0) }.ctx("host buffer");
        Ok(Resources {
            xz: st.alloc_zeros::<i32>(2 * n).ctx("columns")?,
            qy: st.alloc_zeros::<i32>(n).ctx("Y quarts")?,
            qf: st.alloc_zeros::<f32>(6 * n).ctx("climate fields")?,
            best: st.alloc_zeros::<i32>(n).ctx("leaves")?,
            err: st.alloc_zeros::<i32>(1).ctx("flags")?,
            h_qf: unsafe { self.ctx.alloc_pinned_with_flags::<f32>(6 * n, 0) }
                .ctx("host buffer")?,
            h_best: pinned_i32(n)?,
            h_err: pinned_i32(1)?,
            st,
        })
    }
    /// The fields and first leaves of the quarts of `columns` (quart x and z) at the Y quarts
    /// `layers`, column after column: the quart at `layers[j]` of column `c` is at
    /// `c * layers.len() + j` of `out`. Blocking; at most [`LOCATE_BATCH`] quarts.
    pub fn sample_columns(
        &self,
        columns: &[[i32; 2]],
        layers: &[i32],
        out: &mut LocateSamples,
    ) -> Result<(), String> {
        self.run(columns, layers, false, out)
    }
    /// The same for quarts of their own, in order.
    pub fn sample_quarts(
        &self,
        quarts: &[[i32; 3]],
        out: &mut LocateSamples,
    ) -> Result<(), String> {
        let columns: Vec<[i32; 2]> = quarts.iter().map(|q| [q[0], q[2]]).collect();
        let layers: Vec<i32> = quarts.iter().map(|q| q[1]).collect();
        self.run(&columns, &layers, true, out)
    }
    fn run(
        &self,
        columns: &[[i32; 2]],
        layers: &[i32],
        own: bool,
        out: &mut LocateSamples,
    ) -> Result<(), String> {
        out.fields.clear();
        out.first.clear();
        let nqy = if own { 1 } else { layers.len() };
        let n = columns.len().saturating_mul(nqy);
        if n == 0 {
            return Ok(());
        }
        if self.broken.load(Relaxed) {
            return Err("off after an earlier CUDA error".into());
        }
        if n > LOCATE_BATCH || layers.len() > LOCATE_BATCH {
            return Err(format!("{n} quarts in one batch"));
        }
        let t = Instant::now();
        let mut res = self.resource()?;
        // Err(true): a CUDA error; Err(false): a flag of the search.
        let result = (|| -> Result<(), (bool, String)> {
            let cuda = |e: String| (true, e);
            let st = res.st.clone();
            let xz: Vec<i32> = columns.iter().flatten().copied().collect();
            st.memcpy_htod(&xz[..], &mut res.xz.slice_mut(0..xz.len()))
                .ctx("column upload")
                .map_err(cuda)?;
            st.memcpy_htod(layers, &mut res.qy.slice_mut(0..layers.len()))
                .ctx("Y quart upload")
                .map_err(cuda)?;
            st.memset_zeros(&mut res.err).ctx("flags").map_err(cuda)?;
            let (ncol, nqy, own, ni) = (columns.len() as i32, nqy as i32, i32::from(own), n as i32);
            let mut b = st.launch_builder(&self.k_climate);
            b.arg(&res.xz)
                .arg(&ncol)
                .arg(&res.qy)
                .arg(&nqy)
                .arg(&own)
                .arg(&mut res.qf);
            unsafe { b.launch(config(columns.len(), 32)) }
                .ctx("climate")
                .map_err(cuda)?;
            let mut b = st.launch_builder(&self.k_search);
            b.arg(&ni).arg(&res.qf).arg(&mut res.best);
            b.arg(&self.e).arg(&self.l).arg(&mut res.err);
            unsafe { b.launch(config(n, 128)) }
                .ctx("search")
                .map_err(cuda)?;
            let h = res.h_qf.as_mut_slice().ctx("host buffer").map_err(cuda)?;
            st.memcpy_dtoh(&res.qf.slice(0..6 * n), &mut h[..6 * n])
                .ctx("field readback")
                .map_err(cuda)?;
            let h = res.h_best.as_mut_slice().ctx("host buffer").map_err(cuda)?;
            st.memcpy_dtoh(&res.best.slice(0..n), &mut h[..n])
                .ctx("leaf readback")
                .map_err(cuda)?;
            st.memcpy_dtoh(
                &res.err,
                res.h_err.as_mut_slice().ctx("host buffer").map_err(cuda)?,
            )
            .ctx("flag readback")
            .map_err(cuda)?;
            st.synchronize().ctx("locate wait").map_err(cuda)?;
            let err = res.h_err.as_slice().ctx("host buffer").map_err(cuda)?[0];
            if err & (E_STACK | E_EMPTY) != 0 {
                return Err((false, format!("climate search (flags {err})")));
            }
            let fields = &res.h_qf.as_slice().ctx("host buffer").map_err(cuda)?[..6 * n];
            out.fields.extend_from_slice(fields.as_chunks::<6>().0);
            let best = &res.h_best.as_slice().ctx("host buffer").map_err(cuda)?[..n];
            out.first.extend(best.iter().map(|&b| b as u32));
            Ok(())
        })();
        let ns = t.elapsed().as_nanos() as u64;
        self.stats.batches.fetch_add(1, Relaxed);
        self.stats.quarts.fetch_add(n as u64, Relaxed);
        self.stats.ns.fetch_add(ns, Relaxed);
        self.stats.ns_max.fetch_max(ns, Relaxed);
        match result {
            Ok(()) => {
                self.pool.lock().unwrap().push(res);
                Ok(())
            }
            Err((cuda, e)) => {
                self.stats.errors.fetch_add(1, Relaxed);
                out.fields.clear();
                out.first.clear();
                let _ = res.st.synchronize();
                if cuda {
                    self.broken.store(true, Relaxed);
                    log::error!("GPU locate: {e}; biome searches on the CPU from now on");
                } else {
                    self.pool.lock().unwrap().push(res);
                }
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// `cl_layers` as `codegen::generate_locate` declares it, with a body of its own: what
    /// locate.cu needs to compile without a density graph.
    const STUB: &str = "__device__ __noinline__ void cl_layers(int x, int z, const int *QY, \
        int nqy, float *V, long step) {\n  for (int j = 0; j < nqy; ++j)\n    for (int k = 0; k < \
        6; ++k) V[j * step + k] = (float)(x + z + QY[j] + k);\n}\n";

    /// locate.cu compiles with the crate's NVRTC options (skipped without the NVRTC library, as
    /// in a build without the `cuda` feature).
    #[test]
    fn the_locate_kernels_compile() {
        if crate::nvrtc_version().is_none() {
            eprintln!("skipping: no NVRTC library");
            return;
        }
        let source = format!("{STUB}{}", include_str!("locate.cu"));
        let options = crate::cuda::nvrtc::CompileOptions {
            arch: Some("compute_75"),
            options: crate::NVRTC_OPTIONS.iter().map(|&o| o.into()).collect(),
            ..Default::default()
        };
        crate::cuda::nvrtc::compile_ptx_with_opts(source, options).expect("locate.cu");
    }
}
