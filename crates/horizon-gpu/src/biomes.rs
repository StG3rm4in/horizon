//! Biomes on the GPU (`HORIZON_GPU_BIOMES`): see BIOMES.md.
//!
//! The GPU computes the biomes of a region of R x R chunks at once: `kbi_climate` samples the
//! climate column by column, `kbi_search` searches each quart in the climate tree without a
//! starting point, then `kbi_order` applies, chunk by chunk and in fill order, the previous search
//! result (`ClimateSearch::last`). The CPU builds each `BiomeChunk` from the region.
use super::{Context, codegen, context, priority_stream, ptx};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice,
    PushKernelArg,
};
use horizon_core::density::graph::{DensityCompiler, FunctionId};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The climate search tree (`ClimateIndex::tree`) and the biome id of each entry.
pub struct BiomeTree<'a> {
    /// Per node: the six minimums, the six maximums, the squared offset distance.
    pub spaces: &'a [[i64; 13]],
    /// Per node: first child and number of children, or entry and 0 for a leaf.
    pub links: &'a [(u32, u32)],
    /// The id of each entry; `u32::MAX` if its name has none (the chunk that picks it falls back to
    /// the CPU, which raises the error).
    pub entry_ids: &'a [u32],
}

/// Flags of `biomes.cu`: traversal stack too short, search without a result (the region falls back
/// to the CPU); entry without an id (only the chunks concerned fall back to the CPU).
const E_STACK: i32 = 1;
const E_EMPTY: i32 = 2;

/// Threads of a `kbi_order` block (BI_TILE in biomes.cu).
const TILE: u32 = 128;

struct Resources {
    st: Arc<CudaStream>,
    /// The six fields of each quart, by Y quart then column
    qf: CudaSlice<f32>,
    q: CudaSlice<i64>,
    best: CudaSlice<i32>,
    fmin: CudaSlice<i64>,
    out: CudaSlice<u32>,
    err: CudaSlice<i32>,
    h_out: PinnedHostSlice<u32>,
    h_err: PinnedHostSlice<i32>,
}

#[derive(Default)]
struct Stats {
    regions: AtomicU64,
    ns: AtomicU64,
    ns_max: AtomicU64,
    errors: AtomicU64,
    chunks: AtomicU64,
    fallbacks: AtomicU64,
    compared: AtomicU64,
    mismatches: AtomicU64,
}

pub struct BiomeGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    k_climate: CudaFunction,
    k_search: CudaFunction,
    k_order: CudaFunction,
    e: CudaSlice<i64>,
    l: CudaSlice<u32>,
    ids: CudaSlice<u32>,
    /// Side of a region in chunks
    r: i32,
    /// First Y quart and number of Y quarts of a chunk
    qy0: i32,
    nqy: i32,
    pool: Mutex<Vec<Resources>>,
    stats: Stats,
}
impl std::fmt::Debug for BiomeGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BiomeGpu")
    }
}
impl Drop for BiomeGpu {
    fn drop(&mut self) {
        let s = &self.stats;
        if let Some(n) = std::num::NonZero::new(s.regions.load(Relaxed)) {
            eprintln!(
                "GPU biomes: {n} regions of {r} x {r} chunks, {} us per region (max {}); {} chunks from the GPU, {} by the CPU; {} regions failed; test mode: {} chunks compared, {} mismatches",
                s.ns.load(Relaxed) / n / 1000,
                s.ns_max.load(Relaxed) / 1000,
                s.chunks.load(Relaxed),
                s.fallbacks.load(Relaxed),
                s.errors.load(Relaxed),
                s.compared.load(Relaxed),
                s.mismatches.load(Relaxed),
                r = self.r,
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

impl BiomeGpu {
    /// `roots`: the six climate roots of the router (temperature, vegetation, continents, erosion,
    /// depth, ridges); `min_y` and `height`: those of the `BiomeChunk`s; `region`: side of a
    /// region.
    pub fn new(
        compiler: &DensityCompiler,
        roots: [FunctionId; 6],
        seed: i64,
        tree: &BiomeTree,
        min_y: i32,
        height: i32,
        region: i32,
    ) -> Result<Self, String> {
        let (n, ne) = (tree.links.len(), tree.entry_ids.len());
        if n == 0 || tree.spaces.len() != n {
            return Err("climate tree".into());
        }
        for &(a, k) in tree.links {
            let ok = if k == 0 {
                (a as usize) < ne
            } else {
                a as usize + k as usize <= n
            };
            if !ok {
                return Err("climate tree links".into());
            }
        }
        if min_y % 16 != 0 || height <= 0 || height % 16 != 0 || !(1..=32).contains(&region) {
            return Err("chunk height or region side".into());
        }
        let src = codegen::generate_biomes(compiler, roots, seed).map_err(|e| e.to_string())?;
        if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP_BIOMES") {
            let _ = std::fs::write(f, &src);
        }
        let ctx = context()?;
        let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
        let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
        let module = ctx.load_module(ptx(&src, arch)?).ctx("biome module")?;
        let k_climate = module.load_function("kbi_climate").ctx("kbi_climate")?;
        let k_search = module.load_function("kbi_search").ctx("kbi_search")?;
        let k_order = module.load_function("kbi_order").ctx("kbi_order")?;
        let st = ctx.new_stream().ctx("CUDA stream")?;
        let flat: Vec<i64> = tree.spaces.iter().flatten().copied().collect();
        let e = st.clone_htod(&flat).ctx("climate tree")?;
        let links: Vec<u32> = tree.links.iter().flat_map(|&(a, k)| [a, k]).collect();
        let l = st.clone_htod(&links).ctx("tree links")?;
        let ids = st.clone_htod(tree.entry_ids).ctx("biome ids")?;
        st.synchronize().ctx("climate tree")?;
        Ok(Self {
            ctx,
            _module: module,
            k_climate,
            k_search,
            k_order,
            e,
            l,
            ids,
            r: region,
            qy0: min_y >> 2,
            nqy: height >> 2,
            pool: Mutex::new(Vec::new()),
            stats: Stats::default(),
        })
    }
    /// Side of a region in chunks.
    pub fn region_side(&self) -> i32 {
        self.r
    }
    /// Quarts of a chunk: 16 per Y quart.
    pub fn quarters_per_chunk(&self) -> usize {
        16 * self.nqy as usize
    }
    /// A chunk built from the GPU, or falling back to the CPU (blending with an old world, entry
    /// without an id, failed region).
    pub fn note_chunk(&self, gpu: bool) {
        if gpu {
            self.stats.chunks.fetch_add(1, Relaxed);
        } else {
            self.stats.fallbacks.fetch_add(1, Relaxed);
        }
    }
    /// Test mode: a chunk compared with the CPU; returns the number of chunks compared.
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
    fn resource(&self, n: usize) -> Result<Resources, String> {
        if let Some(r) = self.pool.lock().unwrap().pop() {
            return Ok(r);
        }
        let st = priority_stream(&self.ctx)?;
        Ok(Resources {
            qf: st.alloc_zeros::<f32>(6 * n).ctx("climate fields")?,
            q: st.alloc_zeros::<i64>(6 * n).ctx("climate values")?,
            best: st.alloc_zeros::<i32>(n).ctx("feuilles")?,
            fmin: st.alloc_zeros::<i64>(n).ctx("distances")?,
            out: st.alloc_zeros::<u32>(n).ctx("biomes")?,
            err: st.alloc_zeros::<i32>(1).ctx("flags")?,
            h_out: unsafe { self.ctx.alloc_pinned_with_flags::<u32>(n, 0) }.ctx("host buffer")?,
            h_err: unsafe { self.ctx.alloc_pinned_with_flags::<i32>(1, 0) }.ctx("host buffer")?,
            st,
        })
    }
    /// The biomes of the R x R chunks from chunk (`cx0`, `cz0`): for each chunk (z * R + x), the id
    /// of its quarts in `BiomeChunk::fill` order (`u32::MAX`: entry without an id). Blocking.
    pub fn region(&self, cx0: i32, cz0: i32) -> Result<Vec<u32>, String> {
        let t = Instant::now();
        let n = (self.r * self.r) as usize * self.quarters_per_chunk();
        let mut res = self.resource(n)?;
        let result = (|| -> Result<Vec<u32>, String> {
            let st = res.st.clone();
            st.memset_zeros(&mut res.err).ctx("flags")?;
            let (r, qy0, nqy) = (self.r, self.qy0, self.nqy);
            let mut b = st.launch_builder(&self.k_climate);
            b.arg(&cx0)
                .arg(&cz0)
                .arg(&r)
                .arg(&qy0)
                .arg(&nqy)
                .arg(&mut res.qf);
            unsafe { b.launch(config((r * r * 16) as usize, 32)) }.ctx("climate")?;
            let mut b = st.launch_builder(&self.k_search);
            b.arg(&r)
                .arg(&nqy)
                .arg(&res.qf)
                .arg(&mut res.q)
                .arg(&mut res.best)
                .arg(&mut res.fmin);
            b.arg(&self.e).arg(&self.l).arg(&mut res.err);
            unsafe { b.launch(config(n, 128)) }.ctx("search")?;
            let mut b = st.launch_builder(&self.k_order);
            b.arg(&r)
                .arg(&nqy)
                .arg(&res.q)
                .arg(&res.best)
                .arg(&res.fmin)
                .arg(&self.e)
                .arg(&self.l);
            b.arg(&self.ids).arg(&mut res.out).arg(&mut res.err);
            let blocks = LaunchConfig {
                grid_dim: ((r * r) as u32, 1, 1),
                block_dim: (TILE, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(blocks) }.ctx("search order")?;
            st.memcpy_dtoh(
                &res.out,
                &mut res.h_out.as_mut_slice().ctx("host buffer")?[..n],
            )
            .ctx("biome readback")?;
            st.memcpy_dtoh(&res.err, res.h_err.as_mut_slice().ctx("host buffer")?)
                .ctx("flag readback")?;
            st.synchronize().ctx("biome wait")?;
            let err = res.h_err.as_slice().ctx("host buffer")?[0];
            if err & (E_STACK | E_EMPTY) != 0 {
                return Err(format!("climate search (flags {err})"));
            }
            Ok(res.h_out.as_slice().ctx("host buffer")?[..n].to_vec())
        })();
        let ns = t.elapsed().as_nanos() as u64;
        self.stats.regions.fetch_add(1, Relaxed);
        self.stats.ns.fetch_add(ns, Relaxed);
        self.stats.ns_max.fetch_max(ns, Relaxed);
        match result {
            Ok(v) => {
                self.pool.lock().unwrap().push(res);
                Ok(v)
            }
            Err(e) => {
                self.stats.errors.fetch_add(1, Relaxed);
                let _ = res.st.synchronize();
                Err(e)
            }
        }
    }
}
