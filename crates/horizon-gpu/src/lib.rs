//! World generation on the GPU (CUDA), bit-identical to Horizon's CPU code: final_density and the
//! aquifers (this module), the surface rules, ore veins, carvers (`caves`), light (`light`), the
//! sections' palettes and the multi-noise biomes (`biomes`).
//!
//! The GPU computes the final_density of a region of R x R chunks at once (8 x 8 by default) and
//! keeps it in GPU memory; each chunk a worker then asks for is copied (393 KB for 16 x 384 x 16)
//! into the `fill_noise` buffer, or, with the aquifers on the GPU, comes back as block codes.
//!
//! Enabled by the `HORIZON_GPU_DENSITY` environment variable:
//! - `1`: compilation in the background, the CPU serves the chunks meanwhile;
//! - `wait`: compilation before the first chunk (parity tests).
//!
//! The other stages have their own variables (`HORIZON_GPU_SURFACE`, `HORIZON_GPU_ORES`,
//! `HORIZON_GPU_CAVES`, `HORIZON_GPU_LIGHT`, `HORIZON_GPU_SECTIONS`, `HORIZON_GPU_BIOMES`), whose
//! `test` value (`HORIZON_GPU_SURFACE_TEST` for the surface) compares every chunk with the CPU.
//!
//! Settings: `HORIZON_GPU_SYNC` (waiting: block by default, spin, yield, auto),
//! `HORIZON_GPU_PRIORITY=0` (no priority for the critical path streams), `HORIZON_GPU_REGION` (side
//! of a region in chunks, 8), `HORIZON_GPU_REGIONS_MAX` (regions kept in GPU memory, 128),
//! `HORIZON_GPU_CACHE` (directory of the compiled PTX), `HORIZON_GPU_DEVICE` (the CUDA device, 0),
//! `HORIZON_GPU_MEMORY_MB` (GPU memory of final_density's region cache; half of the memory free at
//! startup by default, a region of the overworld takes about 33 MB). The surface
//! (`HORIZON_GPU_SURFACE_REGIONS`, 48 regions) and the light (`HORIZON_GPU_LIGHT_CACHE`, slots of
//! sections) have caches of their own. Out of GPU memory, the region cache halves (down to 8
//! regions) and the chunk goes to the CPU; any other CUDA error, or no memory even for 8 regions,
//! turns the stage off for the session.
//!
//! Build: the `cuda` feature (on by default in horizon-world, horizon-client-render and the game)
//! compiles the CUDA backend. It needs no CUDA toolkit: the bindings are those of CUDA 13.4, and the
//! driver and NVRTC libraries (CUDA 11 to 13) are loaded at run time. Without the feature (the
//! server build with `--no-default-features`, unless it adds `--features cuda`), `cuda_off.rs`
//! stands in for cudarc and every stage reports CUDA as absent.
//!
//! Checking and measuring: `tools/gpu_worldgen.py` (probe, verify, pregen, compare-worlds), see
//! BENCHMARKS.md in this crate.
//!
//! At startup, a self-test compares the final_density of three chunks with the CPU; at the
//! slightest difference or CUDA error, final_density and the aquifers stay on the CPU. The
//! self-test covers final_density only: the other stages are checked by their `test` values and
//! the parity harness, not at startup.
mod biomes;
mod caves;
mod codegen;
mod light;

// The CUDA API: cudarc with the `cuda` feature, else a stand-in that reports CUDA as absent.
#[cfg(feature = "cuda")]
use cudarc as cuda;
#[cfg(not(feature = "cuda"))]
mod cuda_off;
#[cfg(not(feature = "cuda"))]
use cuda_off as cuda;

pub use biomes::{BiomeGpu, BiomeTree};
pub use light::{LightGpu, LightInput, LightJob, LightSection};

pub use caves::{
    CAVE_CANYON, CAVE_MASK_WORDS, CAVE_ROOM, CAVE_TUNNEL, CaveGpu, CaveItem, CaveJob, CaveTarget,
};

use crate::cuda::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice,
    PushKernelArg,
};
use crate::cuda::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use horizon_core::density::graph::{DensityCompiler, FunctionId};
use horizon_core::density::{Beardifier, DensitySampler, SamplerContext, TerrainAdjustment};
use horizon_core::noise::DensityVolume;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

trait Context<T> {
    fn ctx(self, what: &str) -> Result<T, String>;
}
impl<T, E: std::fmt::Debug> Context<T> for Result<T, E> {
    fn ctx(self, what: &str) -> Result<T, String> {
        self.map_err(|e| format!("{what}: {e:?}"))
    }
}

fn env_or(name: &str, fallback: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// The process's CUDA context, shared by all dimensions.
fn context() -> Result<Arc<CudaContext>, String> {
    static CTX: OnceLock<Result<Arc<CudaContext>, String>> = OnceLock::new();
    CTX.get_or_init(|| {
        if !cfg!(feature = "cuda") {
            return Err("built without CUDA (the cuda feature of horizon-gpu)".into());
        }
        // SAFETY: only tries to open the library; cudarc would panic on a missing one
        if !unsafe { crate::cuda::driver::sys::is_culib_present() } {
            return Err("CUDA driver library not found".into());
        }
        // HORIZON_GPU_DEVICE: the CUDA device, 0 by default
        let device = match std::env::var("HORIZON_GPU_DEVICE") {
            Ok(v) => v
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("HORIZON_GPU_DEVICE={v}: not a device number"))?,
            Err(_) => 0,
        };
        let ctx = CudaContext::new(device)
            .map_err(|e| format!("CUDA device {device} (HORIZON_GPU_DEVICE): {e:?}"))?;
        log::info!(
            "GPU: CUDA device {device}, {} ({} MB)",
            ctx.name().unwrap_or_else(|_| "?".into()),
            ctx.total_mem().map_or(0, |b| b >> 20)
        );
        // Waiting for a computation to finish: HORIZON_GPU_SYNC = block (default: the thread
        // sleeps), spin, yield or auto (CUDA heuristic, often spin: the core spins idle while
        // waiting).
        // With light on the GPU, the CPU is no longer the bottleneck of /pregen: sleeping gives
        // back 4 to 6 % of the CPU per chunk without changing the throughput (GPU_WAIT.md).
        use crate::cuda::driver::sys::CUctx_flags::*;
        let wait = match std::env::var("HORIZON_GPU_SYNC").as_deref() {
            Ok("spin") => Some(CU_CTX_SCHED_SPIN),
            Ok("yield") => Some(CU_CTX_SCHED_YIELD),
            Ok("auto") => None,
            _ => Some(CU_CTX_SCHED_BLOCKING_SYNC),
        };
        if let Some(f) = wait {
            ctx.set_flags(f).ctx("wait mode")?;
        }
        // Synchronization handled here: each computation and each copy waits for the end of its
        // stream.
        unsafe { ctx.disable_event_tracking() };
        Ok(ctx)
    })
    .clone()
}

/// u32 words of a section in the output of `SurfaceGpu::enable_sections` (sections.cu, SC_STRIDE):
/// a 5-word header, 128 states, 128 counters, 1024 data words.
pub const SECTION_WORDS: usize = 5 + 2 * 128 + 1024;
/// Post-processing marks per list at most in this output (beyond that: raw states, CPU).
pub const SECTION_MARKS: usize = 2048;

/// A critical path stream (a worker waits for its result: filling, surface, region): the GPU's
/// highest priority, to go ahead of the asynchronous batches (light, caves, ores), created at the
/// default priority. `HORIZON_GPU_PRIORITY=0` puts them all at the same one.
fn priority_stream(ctx: &Arc<CudaContext>) -> Result<Arc<CudaStream>, String> {
    static PRIORITY: OnceLock<i32> = OnceLock::new();
    let p = *PRIORITY.get_or_init(|| {
        if std::env::var("HORIZON_GPU_PRIORITY").as_deref() == Ok("0")
            || ctx.bind_to_thread().is_err()
        {
            return 0;
        }
        crate::cuda::driver::result::stream::get_priority_range().map_or(0, |(_, high)| high)
    });
    ctx.new_stream_with_priority(p).ctx("CUDA stream")
}

/// The aquifers of a dimension, to decide each block on the GPU too.
#[derive(Clone, Copy, Debug)]
pub struct AquiferSetup {
    /// Roots in AquiferField::ALL order (barrier, flood, spread, lava, exclusion, surface level);
    /// None for a dimension without aquifers (the Nether, the End), where every block that is not
    /// solid takes the global fluid picker's substance (`DisabledAquifer`).
    pub roots: Option<[FunctionId; 6]>,
    /// State (lo, hi) of the aquifers' positional random factory (Xoroshiro).
    pub positional: (u64, u64),
    pub sea_level: i32,
}

/// final_density of a dimension on the GPU.
pub struct GpuFinalDensity {
    shared: Arc<Shared>,
}

struct Shared {
    name: String,
    engine: OnceLock<Option<Engine>>,
}

impl std::fmt::Debug for GpuFinalDensity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = match self.shared.engine.get() {
            None => "en preparation",
            Some(None) => "disabled",
            Some(Some(m)) if m.broken.load(Relaxed) => "broken",
            Some(Some(_)) => "ready",
        };
        write!(f, "GpuFinalDensity({}, {state})", self.shared.name)
    }
}

impl GpuFinalDensity {
    /// `None` if `HORIZON_GPU_DENSITY` is unset or 0, or if a graph node has no CUDA translation.
    /// `cpu` is the reference sampler of the self-test. A legacy random source (the Nether, the
    /// End) only changes how the compiler draws its noises, which the GPU reads from it.
    pub fn new(
        compiler: &DensityCompiler,
        root: FunctionId,
        seed: i64,
        legacy: bool,
        cpu: Arc<dyn DensitySampler>,
        min_y: i32,
        height: i32,
        name: &str,
        aquifer: Option<AquiferSetup>,
    ) -> Option<Self> {
        let mode = std::env::var("HORIZON_GPU_DENSITY").ok()?;
        if mode.is_empty() || mode == "0" {
            return None;
        }
        let _ = legacy;
        let r = env_or("HORIZON_GPU_REGION", 8).clamp(1, 32);
        let capacity = env_or("HORIZON_GPU_REGIONS_MAX", 128).max(1) as usize;
        let t = Instant::now();
        let prog = match codegen::generate(compiler, root, seed, [16 * r, height, 16 * r]) {
            Ok(p) => p,
            Err(e) => {
                let msg = format!(
                    "GPU final_density ({name}): no CUDA translation ({e}), staying on the CPU"
                );
                log::info!("{msg}");
                eprintln!("{msg}");
                return None;
            }
        };
        log::info!(
            "GPU final_density ({name}): {} lines of CUDA generated in {:.0} ms, regions of {r}x{r} chunks ({:.0} MB of GPU memory each)",
            prog.lines,
            t.elapsed().as_secs_f64() * 1e3,
            prog.size as f64 * 4e-6
        );
        if !prog.beard_last {
            log::info!(
                "GPU final_density ({name}): the beardifier is not added last, chunks with structures stay on the CPU"
            );
        }
        if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP") {
            let _ = std::fs::write(f, &prog.src);
        }
        // Aquifers: HORIZON_GPU_AQUIFERS=0 leaves them on the CPU
        let aquifer_prog = aquifer
            .filter(|_| std::env::var("HORIZON_GPU_AQUIFERS").map_or(true, |v| v != "0"))
            .and_then(|a| match codegen::generate_aquifer(compiler, a.roots, seed) {
                Ok(src) => Some((a, src)),
                Err(e) => {
                    log::info!("GPU final_density ({name}): aquifers without a CUDA translation ({e}), staying on the CPU");
                    None
                }
            });
        let shared = Arc::new(Shared {
            name: name.to_owned(),
            engine: OnceLock::new(),
        });
        let p = shared.clone();
        let work = move || {
            let t = Instant::now();
            let m = Engine::start(prog, aquifer_prog, r, capacity, min_y, height)
                .and_then(|m| m.auto_test(&*cpu).map(|()| m));
            match m {
                Ok(m) => {
                    let msg = format!(
                        "GPU final_density ({}): ready in {:.1} s, self-test identical to the CPU{}",
                        p.name,
                        t.elapsed().as_secs_f64(),
                        if m.aq.is_some() {
                            ", aquifers on the GPU"
                        } else {
                            ""
                        }
                    );
                    log::info!("{msg}");
                    eprintln!("{msg}");
                    let _ = p.engine.set(Some(m));
                }
                Err(e) => {
                    log::error!(
                        "GPU final_density ({}): disabled, everything stays on the CPU: {e}",
                        p.name
                    );
                    eprintln!(
                        "GPU final_density ({}): disabled, everything stays on the CPU: {e}",
                        p.name
                    );
                    let _ = p.engine.set(None);
                }
            }
        };
        if mode == "wait" {
            work();
        } else if let Err(e) = std::thread::Builder::new()
            .name("horizon-gpu".into())
            .spawn(work)
        {
            log::error!("GPU final_density ({name}): compilation thread: {e}");
            return None;
        }
        Some(Self { shared })
    }

    /// Writes the chunk's final density into `out` (16 x height x 16, `fill_noise` order).
    /// `false`: GPU not ready yet, disabled or failed; the CPU must compute this chunk.
    pub fn fill_chunk(&self, chunk_x: i32, chunk_z: i32, out: &mut [f32]) -> bool {
        match self.shared.engine.get() {
            Some(Some(m)) => m.chunk(chunk_x, chunk_z, out),
            _ => false,
        }
    }

    /// Codes of each block of the chunk (`fill_noise` order): 0 density > 0, 1 air, 2 default
    /// fluid, 3 lava, 8 default block decided by the aquifer's pressure, +4 scheduled fluid update.
    /// Returns the chunk's skip_sampling_above_y, or `None` (no GPU, no aquifers, or failed).
    /// `caves`: the chunk's carver batch, whose cells the GPU aquifer then computes too.
    pub fn chunk_blocks(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        out: &mut [i8],
        caves: Option<&mut CaveJob>,
    ) -> Option<i32> {
        match self.shared.engine.get() {
            Some(Some(m)) if m.aq.is_some() => m.chunk_codes(chunk_x, chunk_z, out, None, caves),
            _ => None,
        }
    }

    /// Like `chunk_blocks`, for a chunk whose terrain adapts to structures: the beardifier is added
    /// to the final density on the GPU before the aquifer decision. `None` for the terrain kernels
    /// of structure libraries (mods), which stay on the CPU.
    pub fn chunk_blocks_beard(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        beard: &Beardifier,
        out: &mut [i8],
        caves: Option<&mut CaveJob>,
    ) -> Option<i32> {
        match self.shared.engine.get() {
            Some(Some(m)) if m.aq.is_some() => {
                m.chunk_codes_beard(chunk_x, chunk_z, beard, out, None, caves)
            }
            _ => None,
        }
    }

    /// `chunk_blocks` (`chunk_blocks_beard` with `beard`), then the chunk's surface on the same
    /// stream: its codes do not go back through the host. Returns the chunk's skip and the outcome
    /// of the surface: `Some(true)` final states in `states` (see `SurfaceGpu::chunk_states`),
    /// `Some(false)` a condition that cannot be evaluated, `None` surface not computed (the chunk
    /// then goes through the CPU surface).
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_blocks_surface(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        beard: Option<&Beardifier>,
        out: &mut [i8],
        sg: &SurfaceGpu,
        biomes: &[u16],
        states: &mut [u32],
        mut caves: Option<&mut CaveJob>,
        mut sections: Option<&mut Vec<u32>>,
    ) -> (Option<i32>, Option<bool>) {
        let with_sections = sections.is_some();
        let Some(Some(m)) = self.shared.engine.get() else {
            return (None, None);
        };
        if m.aq.is_none() {
            return (None, None);
        }
        let simple = |out: &mut [i8], caves: Option<&mut CaveJob>| match beard {
            Some(b) => m.chunk_codes_beard(chunk_x, chunk_z, b, out, None, caves),
            None => m.chunk_codes(chunk_x, chunk_z, out, None, caves),
        };
        if states.len() != out.len()
            || out.len() != 256 * sg.height as usize
            || biomes.len() != sg.biome_count()
        {
            return (simple(out, caves), None);
        }
        let Ok(mut res) = sg.resource() else {
            return (simple(out, caves), None);
        };
        let t = Instant::now();
        let mut issue: Option<Result<Arc<RegionSurface>, String>> = None;
        let mut follow_up = |st: &Arc<CudaStream>, cd: &CudaSlice<i8>| {
            issue = Some(sg.enqueue(
                &mut res,
                st,
                Some(cd),
                chunk_x,
                chunk_z,
                biomes,
                with_sections,
            ));
        };
        let skip = match beard {
            Some(b) => {
                m.chunk_codes_beard(chunk_x, chunk_z, b, out, Some(&mut follow_up), caves.take())
            }
            None => m.chunk_codes(chunk_x, chunk_z, out, Some(&mut follow_up), caves.take()),
        };
        // The codes' stream was waited for by chunk_codes: the states have arrived
        let surface = match (skip, issue) {
            (Some(_), Some(Ok(_region))) => {
                let readback = sg.read_back(&mut res, states, sections.take());
                sg.put_back(res, readback.is_ok(), t);
                readback.ok()
            }
            (_, Some(Err(e))) => {
                static REPORTED: OnceLock<()> = OnceLock::new();
                if REPORTED.set(()).is_ok() {
                    log::error!("GPU surface: chunk {chunk_x} {chunk_z}: {e}");
                }
                None
            }
            _ => None,
        };
        (skip, surface)
    }

    /// A chunk whose blocks the CPU recomputed: the air limit of its aquifer differed from the GPU.
    pub fn note_aquifer_mismatch(&self) {
        if let Some(Some(m)) = self.shared.engine.get() {
            m.stats.mismatches.fetch_add(1, Relaxed);
        }
    }

    /// A chunk computed on the CPU because its terrain depends on structures or old chunks.
    pub fn note_cpu_fallback(&self) {
        if let Some(Some(m)) = self.shared.engine.get() {
            m.stats.fallbacks.fetch_add(1, Relaxed);
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        if let Some(Some(m)) = self.engine.get()
            && m.stats.chunks.load(Relaxed) > 0
        {
            eprintln!("{}", m.summary());
        }
    }
}

/// What the GPU keeps of a region: final density, and block codes if the aquifers are there.
struct RegionData {
    mem: CudaSlice<f32>,
    codes: Option<CudaSlice<i8>>,
    /// Positions and statuses of the aquifer cells, to redo the chunks with structures
    cells: Option<Cells>,
    /// skip_sampling_above_y of each chunk of the region (cz * r + cx)
    skips: Vec<i32>,
}

struct Cells {
    p: CudaSlice<i32>,
    st: CudaSlice<i32>,
    /// cx0, cy0, cz0, ncx, ncy, ncz
    grid: [i32; 6],
}

#[derive(Default)]
struct Region {
    data: OnceLock<Result<RegionData, String>>,
    served: AtomicU32,
}

struct Cache {
    lookup: HashMap<(i32, i32), (Arc<Region>, u64)>,
    clock: u64,
    capacity: usize,
}

/// Stream and buffers of a worker for copying a chunk, returned to the pool after use.
struct Resources {
    stream: Arc<CudaStream>,
    buf: CudaSlice<f32>,
    host: PinnedHostSlice<f32>,
    buf_codes: CudaSlice<i8>,
    host_codes: PinnedHostSlice<i8>,
}

#[derive(Default)]
struct Stats {
    chunks: AtomicU64,
    ns_copy: AtomicU64,
    regions: AtomicU64,
    ns_regions: AtomicU64,
    evictions: AtomicU64,
    fallbacks: AtomicU64,
    /// Chunks filled from the GPU's aquifer codes
    chunks_aq: AtomicU64,
    /// Chunks where the CPU aquifer's air limit differed from the GPU
    mismatches: AtomicU64,
    /// Chunks with structures (beardifier) done on the GPU
    chunks_beard: AtomicU64,
    /// Times the region cache halved for lack of GPU memory
    shrinks: AtomicU64,
}

/// The aquifer kernels and their constants.
struct Aquifers {
    _module: Arc<crate::cuda::driver::CudaModule>,
    surface: CudaFunction,
    cells: CudaFunction,
    skip: CudaFunction,
    blocks: CudaFunction,
    chunk: CudaFunction,
    beard: CudaFunction,
    /// Substance of the carved cells (caves)
    caves: CudaFunction,
    /// Vanilla's beard kernel (24 x 24 x 24)
    kernel: CudaSlice<f32>,
    positional: (u64, u64),
    sea: i32,
    /// No aquifers: the skip height of every chunk is below every block, so `aq_substance` takes
    /// the global fluid picker without reading any cell.
    disabled: bool,
}

type Launch = (CudaFunction, usize, usize);

struct Engine {
    ctx: Arc<CudaContext>,
    /// Axis tables: (kernel, grid, threads)
    f_ax: Vec<Launch>,
    /// Grids in launch order: (grid, 2D pass, 3D kernel, points)
    f_cv: Vec<(usize, Option<(CudaFunction, usize)>, CudaFunction, usize)>,
    f_chunk: CudaFunction,
    aq: Option<Aquifers>,
    grid_step: Vec<[i32; 3]>,
    size: usize,
    r: i32,
    min_y: i32,
    n_chunk: usize,
    /// final_density adds the beardifier last (`codegen::beard_added_last`)
    beard_last: bool,
    cache: Mutex<Cache>,
    pool: Mutex<Vec<Resources>>,
    stats: Stats,
    broken: AtomicBool,
}

fn config(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// 64-bit FNV-1a: stable name of the cached PTX.
fn footprint(octets: &[u8]) -> u64 {
    octets.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// NVRTC's options: no fused multiply-add, no flush to zero, IEEE division and square root.
const NVRTC_OPTIONS: [&str; 4] = [
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
];

/// The directory of the compiled PTX: `HORIZON_GPU_CACHE`, else `horizon-gpu` in the user's cache
/// directory (`XDG_CACHE_HOME` or `~/.cache`; `%LOCALAPPDATA%` on Windows).
fn ptx_cache_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("HORIZON_GPU_CACHE") {
        return Some(PathBuf::from(d));
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|d| PathBuf::from(d).join(".cache")))
    };
    base.map(|d| d.join("horizon-gpu"))
}

/// NVRTC's version, part of the PTX cache key; None without its library (cudarc would panic).
fn nvrtc_version() -> Option<(i32, i32)> {
    static VERSION: OnceLock<Option<(i32, i32)>> = OnceLock::new();
    *VERSION.get_or_init(|| {
        use crate::cuda::nvrtc::sys;
        // SAFETY: the library is checked before its symbol is loaded; the out-pointers are live
        unsafe {
            if !sys::is_culib_present() {
                return None;
            }
            let (mut major, mut minor) = (0, 0);
            (sys::nvrtcVersion(&mut major, &mut minor) == sys::nvrtcResult::NVRTC_SUCCESS)
                .then_some((major, minor))
        }
    })
}

/// The module of `src`, from the PTX disk cache if possible (NVRTC takes about 12 s). The key
/// covers the source, the architecture, NVRTC's version and its options; a cached PTX the driver
/// refuses (damaged, or left by another toolchain) is compiled again and replaced.
fn load_module(
    ctx: &Arc<CudaContext>,
    src: &str,
    arch: &'static str,
) -> Result<Arc<CudaModule>, String> {
    let version = nvrtc_version();
    let dir = ptx_cache_dir();
    let file = match (version, &dir) {
        (Some((major, minor)), Some(d)) => {
            let options = NVRTC_OPTIONS.join(" ");
            let key = footprint(format!("{arch}\n{major}.{minor}\n{options}\n{src}").as_bytes());
            Some(d.join(format!("final_density-{key:016x}.ptx")))
        }
        _ => None,
    };
    if let Some(f) = &file
        && let Ok(text) = std::fs::read_to_string(f)
    {
        match ctx.load_module(Ptx::from_src(text)) {
            Ok(module) => return Ok(module),
            Err(e) => log::warn!(
                "GPU: cached PTX {} refused ({e:?}), compiling it again",
                f.display()
            ),
        }
    }
    if version.is_none() {
        return Err("NVRTC library not found".into());
    }
    let options = CompileOptions {
        arch: Some(arch),
        options: NVRTC_OPTIONS.iter().map(|&o| o.into()).collect(),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(src, options).ctx("compilation NVRTC")?;
    let text = ptx.to_src();
    let module = ctx.load_module(ptx).ctx("module load")?;
    // Written whole under a unique temporary name, then renamed: no reader sees half a file
    if let (Some(d), Some(f)) = (&dir, &file) {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let n = SEQUENCE.fetch_add(1, Relaxed);
        let tmp = f.with_extension(format!("ptx.{}.{n}.tmp", std::process::id()));
        let written = std::fs::create_dir_all(d)
            .and_then(|()| std::fs::write(&tmp, &text))
            .and_then(|()| std::fs::rename(&tmp, f));
        if written.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    Ok(module)
}

impl Engine {
    fn start(
        prog: codegen::Program,
        aquifer: Option<(AquiferSetup, String)>,
        r: i32,
        capacity: usize,
        min_y: i32,
        height: i32,
    ) -> Result<Self, String> {
        let ctx = context()?;
        // The region cache within a budget: HORIZON_GPU_MEMORY_MB, else half of the GPU memory free
        // now (the other stages and dimensions take theirs from what is left). A region holds its
        // densities and its block codes: about 33 MB for 8 x 8 chunks of the overworld.
        let region_bytes = prog.size * 4 + (r * r) as usize * 256 * height as usize;
        let budget = match std::env::var("HORIZON_GPU_MEMORY_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            Some(mb) => mb << 20,
            None => ctx.mem_get_info().map_or(usize::MAX, |(free, _)| free / 2),
        };
        let capacity = capacity.min((budget / region_bytes.max(1)).max(1));
        log::info!(
            "GPU final_density: up to {capacity} regions of {:.0} MB in GPU memory (budget {} MB)",
            region_bytes as f64 / 1e6,
            budget.min(usize::MAX >> 1) >> 20
        );
        let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
        let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
        let module = load_module(&ctx, &prog.src, arch).ctx("module load")?;
        let f = |name: &str| module.load_function(name).ctx(name);
        let mut f_ax = Vec::new();
        for &(k, gr, nb) in &prog.axes {
            f_ax.push((f(&format!("kax_{k}"))?, gr, nb));
        }
        let mut f_cv = Vec::new();
        for k in (1..prog.grids.len()).chain(std::iter::once(0)) {
            let gr = prog.grids[k];
            let k2d = if prog.edges[k] > 0 {
                Some((f(&format!("k2d_{k}"))?, gr.columns() * prog.edges[k]))
            } else {
                None
            };
            f_cv.push((k, k2d, f(&format!("k3d_{k}"))?, gr.n()));
        }
        let aq = match aquifer {
            Some((a, src)) => {
                let m = load_module(&ctx, &src, arch).ctx("aquifer module")?;
                let g = |name: &str| m.load_function(name).ctx(name);
                let st = ctx.new_stream().ctx("CUDA stream")?;
                let kernel = st
                    .clone_htod(&horizon_core::density::beard_kernel()[..])
                    .ctx("beard kernel")?;
                st.synchronize().ctx("beard kernel")?;
                Some(Aquifers {
                    beard: g("kbeard_chunk")?,
                    caves: g("kcv_substance")?,
                    kernel,
                    surface: g("kaq_surface")?,
                    cells: g("kaq_cells")?,
                    skip: g("kaq_skip")?,
                    blocks: g("kaq_blocks_region")?,
                    chunk: g("kaq_chunk")?,
                    _module: m,
                    positional: a.positional,
                    sea: a.sea_level,
                    disabled: a.roots.is_none(),
                })
            }
            None => None,
        };
        Ok(Self {
            f_chunk: f("kchunk")?,
            aq,
            ctx,
            f_ax,
            f_cv,
            grid_step: prog.grids.iter().map(|g| g.step).collect(),
            size: prog.size,
            r,
            min_y,
            n_chunk: 256 * height as usize,
            beard_last: prog.beard_last,
            cache: Mutex::new(Cache {
                lookup: HashMap::new(),
                clock: 0,
                capacity,
            }),
            pool: Mutex::new(Vec::new()),
            stats: Stats::default(),
            broken: AtomicBool::new(false),
        })
    }

    /// Three chunks (some with negative coordinates) compared bit for bit with the CPU sampler.
    fn auto_test(&self, cpu: &dyn DensitySampler) -> Result<(), String> {
        let h = (self.n_chunk / 256) as i32;
        for (cx, cz) in [(0, 0), (self.r - 1, 1), (-1, -1)] {
            let mut gpu = vec![0.0f32; self.n_chunk];
            let mut reference = vec![0.0f32; self.n_chunk];
            self.inner_chunk(cx, cz, &mut gpu)?;
            let volume = DensityVolume::new([16, h, 16], [cx * 16, self.min_y, cz * 16], [1; 3])
                .ctx("volume")?;
            cpu.sample_volume(&mut SamplerContext::cached(), &mut reference, volume)
                .ctx("reference CPU")?;
            let diff = gpu
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if diff > 0 {
                let ex: Vec<String> = [0usize, 1, 100, 384, 50000]
                    .iter()
                    .map(|&i| format!("[{i}] gpu {} cpu {}", gpu[i], reference[i]))
                    .collect();
                return Err(format!(
                    "self-test: {diff} densities differ from the CPU in chunk {cx} {cz} ({})",
                    ex.join(", ")
                ));
            }
        }
        Ok(())
    }

    fn chunk(&self, cx: i32, cz: i32, out: &mut [f32]) -> bool {
        // Outside the world border (30 M blocks), the CPU keeps control
        if self.broken.load(Relaxed)
            || out.len() != self.n_chunk
            || cx.unsigned_abs() > 1_875_000
            || cz.unsigned_abs() > 1_875_000
        {
            return false;
        }
        match self.inner_chunk(cx, cz, out) {
            Ok(()) => {
                let n = self.stats.chunks.load(Relaxed);
                if n.is_power_of_two() && n >= 1024 || n.is_multiple_of(65536) {
                    self.journal();
                }
                true
            }
            Err(e) => {
                self.failed(&e, (cx.div_euclid(self.r), cz.div_euclid(self.r)));
                false
            }
        }
    }

    fn journal(&self) {
        log::info!("{}", self.summary());
    }

    fn summary(&self) -> String {
        let s = &self.stats;
        let (n, reg) = (s.chunks.load(Relaxed), s.regions.load(Relaxed));
        format!(
            "GPU final_density: {n} chunks, {} of them with aquifers (copy {:.0} us on average), {reg} regions computed ({:.2} ms on average), {} evictions, {} chunks on the CPU (structures, blending), {} aquifer mismatches, {} chunks with structures on the GPU, {} cache shrinks (out of GPU memory)",
            s.chunks_aq.load(Relaxed),
            s.ns_copy.load(Relaxed) as f64 / 1e3 / n.max(1) as f64,
            s.ns_regions.load(Relaxed) as f64 / 1e6 / reg.max(1) as f64,
            s.evictions.load(Relaxed),
            s.fallbacks.load(Relaxed),
            s.mismatches.load(Relaxed),
            s.chunks_beard.load(Relaxed),
            s.shrinks.load(Relaxed)
        )
    }

    /// A chunk's GPU work failed. Out of GPU memory, the region cache halves (down to
    /// `MIN_REGIONS`) and drops the region, whose next chunks compute it again with more room; only
    /// this chunk goes to the CPU. Any other error, or a cache already at its floor, turns
    /// final_density off for the session: with fewer regions the workers would compute the same
    /// regions over and over, slower than the CPU.
    fn failed(&self, e: &str, key: (i32, i32)) {
        const MIN_REGIONS: usize = 8;
        if e.contains("OUT_OF_MEMORY") {
            let mut c = self.cache.lock().unwrap();
            c.lookup.remove(&key);
            if c.capacity > MIN_REGIONS {
                let old = c.capacity;
                c.capacity = (old / 2).max(MIN_REGIONS);
                while c.lookup.len() > c.capacity {
                    let Some(oldest) = c
                        .lookup
                        .iter()
                        .min_by_key(|(_, (_, t))| *t)
                        .map(|(k, _)| *k)
                    else {
                        break;
                    };
                    // Released when the last worker that reads it is done
                    c.lookup.remove(&oldest);
                    self.stats.evictions.fetch_add(1, Relaxed);
                }
                let new = c.capacity;
                drop(c);
                self.stats.shrinks.fetch_add(1, Relaxed);
                log::warn!(
                    "GPU final_density: out of GPU memory, region cache {old} -> {new} regions; this chunk goes to the CPU"
                );
                return;
            }
        }
        if !self.broken.swap(true, Relaxed) {
            log::error!("GPU final_density disabled, back to the CPU: {e}");
        }
    }

    fn inner_chunk(&self, cx: i32, cz: i32, out: &mut [f32]) -> Result<(), String> {
        let r = self.r;
        let (rx, rz) = (cx.div_euclid(r), cz.div_euclid(r));
        let (lx, lz) = (cx - rx * r, cz - rz * r);
        let region = self.region((rx, rz));
        let mut res = self.resource()?;
        let result = self.copy_out(&region, rx, rz, lx, lz, &mut res, out);
        self.pool.lock().unwrap().push(res);
        if result.is_ok() && region.served.fetch_add(1, Relaxed) + 1 == (r * r) as u32 {
            // Every chunk of the region has been served: its GPU memory is released
            let mut c = self.cache.lock().unwrap();
            if c.lookup
                .get(&(rx, rz))
                .is_some_and(|(e, _)| Arc::ptr_eq(e, &region))
            {
                c.lookup.remove(&(rx, rz));
            }
        }
        result
    }

    fn copy_out(
        &self,
        region: &Region,
        rx: i32,
        rz: i32,
        lx: i32,
        lz: i32,
        res: &mut Resources,
        out: &mut [f32],
    ) -> Result<(), String> {
        // The first worker that asks for the region computes it; the others wait here
        let data = self.data(region, rx, rz, res)?;
        let t = Instant::now();
        let mut b = res.stream.launch_builder(&self.f_chunk);
        b.arg(&data.mem).arg(&mut res.buf).arg(&lx).arg(&lz);
        unsafe { b.launch(config(self.n_chunk)) }.ctx("chunk copy")?;
        res.stream
            .memcpy_dtoh(&res.buf, &mut res.host)
            .ctx("chunk readback")?;
        // Without event tracking, cudarc does not record the event of the pinned buffer and
        // `as_slice` would not wait for the end of the copy: wait for the stream itself.
        res.stream.synchronize().ctx("readback wait")?;
        out.copy_from_slice(res.host.as_slice().ctx("readback read")?);
        self.stats.chunks.fetch_add(1, Relaxed);
        self.stats
            .ns_copy
            .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        Ok(())
    }

    /// The region's data; the first worker that asks for it computes it, the others wait.
    fn data<'r>(
        &self,
        region: &'r Region,
        rx: i32,
        rz: i32,
        res: &Resources,
    ) -> Result<&'r RegionData, String> {
        let d = region.data.get_or_init(|| {
            let t = Instant::now();
            let m = self.compute(rx, rz, &res.stream);
            self.stats.regions.fetch_add(1, Relaxed);
            self.stats
                .ns_regions
                .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            m.map_err(|e| format!("region {rx} {rz}: {e}"))
        });
        d.as_ref().map_err(|e| e.clone())
    }

    /// Block codes of a chunk and its skip_sampling_above_y.
    fn chunk_codes(
        &self,
        cx: i32,
        cz: i32,
        out: &mut [i8],
        mut follow_up: Option<&mut dyn FnMut(&Arc<CudaStream>, &CudaSlice<i8>)>,
        caves: Option<&mut CaveJob>,
    ) -> Option<i32> {
        if self.broken.load(Relaxed)
            || out.len() != self.n_chunk
            || cx.unsigned_abs() > 1_875_000
            || cz.unsigned_abs() > 1_875_000
        {
            return None;
        }
        let r = self.r;
        let (rx, rz) = (cx.div_euclid(r), cz.div_euclid(r));
        let (lx, lz) = (cx - rx * r, cz - rz * r);
        let region = self.region((rx, rz));
        let result = self.resource().and_then(|mut res| {
            let rr = self.copy_codes(&region, rx, rz, lx, lz, &mut res, out, follow_up.take());
            self.pool.lock().unwrap().push(res);
            rr
        });
        match result {
            Ok(skip) => {
                if let Some(j) = caves {
                    self.caves_substance(&region, lx, lz, j);
                }
                if region.served.fetch_add(1, Relaxed) + 1 == (r * r) as u32 {
                    let mut c = self.cache.lock().unwrap();
                    if c.lookup
                        .get(&(rx, rz))
                        .is_some_and(|(e, _)| Arc::ptr_eq(e, &region))
                    {
                        c.lookup.remove(&(rx, rz));
                    }
                }
                let n = self.stats.chunks.load(Relaxed);
                if n.is_power_of_two() && n >= 1024 || n.is_multiple_of(65536) {
                    self.journal();
                }
                Some(skip)
            }
            Err(e) => {
                self.failed(&e, (rx, rz));
                None
            }
        }
    }

    fn chunk_codes_beard(
        &self,
        cx: i32,
        cz: i32,
        beard: &Beardifier,
        out: &mut [i8],
        mut follow_up: Option<&mut dyn FnMut(&Arc<CudaStream>, &CudaSlice<i8>)>,
        caves: Option<&mut CaveJob>,
    ) -> Option<i32> {
        static NO_BEARD: OnceLock<bool> = OnceLock::new();
        if *NO_BEARD.get_or_init(|| std::env::var("HORIZON_GPU_BEARD").as_deref() == Ok("0")) {
            return None;
        }
        let (pieces, junctions, kp, kj, zone) = beard.parts();
        if !self.beard_last
            || !kp.is_empty()
            || !kj.is_empty()
            || self.broken.load(Relaxed)
            || out.len() != self.n_chunk
            || cx.unsigned_abs() > 1_875_000
            || cz.unsigned_abs() > 1_875_000
        {
            return None;
        }
        let pc: Vec<i32> = pieces
            .iter()
            .flat_map(|p| {
                let (a, b) = (p.bounds.minimum(), p.bounds.maximum());
                let adj = match p.adjustment {
                    TerrainAdjustment::None => 0,
                    TerrainAdjustment::Bury => 1,
                    TerrainAdjustment::BeardThin => 2,
                    TerrainAdjustment::BeardBox => 3,
                    TerrainAdjustment::Encapsulate => 4,
                };
                [
                    a[0],
                    a[1],
                    a[2],
                    b[0],
                    b[1],
                    b[2],
                    adj,
                    p.ground_level_delta,
                ]
            })
            .collect();
        let jc: Vec<i32> = junctions
            .iter()
            .flat_map(|j| [j.source_x, j.source_ground_y, j.source_z])
            .collect();
        let z: Vec<i32> = zone.map_or(vec![0; 6], |b| {
            let (a, c) = (b.minimum(), b.maximum());
            vec![a[0], a[1], a[2], c[0], c[1], c[2]]
        });
        let r = self.r;
        let (rx, rz) = (cx.div_euclid(r), cz.div_euclid(r));
        let (lx, lz) = (cx - rx * r, cz - rz * r);
        let region = self.region((rx, rz));
        let result = self.resource().and_then(|mut res| {
            let rr = (|| -> Result<i32, String> {
                let data = self.data(&region, rx, rz, &res)?;
                let (aq, cell_grid) = match (&self.aq, &data.cells) {
                    (Some(a), Some(c)) => (a, c),
                    _ => return Err("region without aquifer cells".into()),
                };
                let t = Instant::now();
                let st = res.stream.clone();
                // Empty arrays are not allocated: at least one element
                let pd = st
                    .clone_htod(if pc.is_empty() { &[0i32][..] } else { &pc[..] })
                    .ctx("pieces")?;
                let jd = st
                    .clone_htod(if jc.is_empty() { &[0i32][..] } else { &jc[..] })
                    .ctx("junctions")?;
                let zd = st.clone_htod(&z[..]).ctx("zone")?;
                let skip = data.skips[(lz * self.r + lx) as usize];
                let (np, nj, has_zone) = (
                    (pc.len() / 8) as i32,
                    (jc.len() / 3) as i32,
                    i32::from(zone.is_some()),
                );
                let (gx, h) = (16 * self.r, (self.n_chunk / 256) as i32);
                let (x0, z0) = (rx * self.r * 16, rz * self.r * 16);
                let gr = cell_grid.grid;
                let mut b = st.launch_builder(&aq.beard);
                b.arg(&data.mem)
                    .arg(&mut res.buf_codes)
                    .arg(&pd)
                    .arg(&np)
                    .arg(&jd)
                    .arg(&nj)
                    .arg(&aq.kernel)
                    .arg(&zd)
                    .arg(&has_zone);
                b.arg(&cell_grid.p)
                    .arg(&cell_grid.st)
                    .arg(&skip)
                    .arg(&lx)
                    .arg(&lz)
                    .arg(&x0)
                    .arg(&z0)
                    .arg(&gx)
                    .arg(&self.min_y)
                    .arg(&h);
                b.arg(&gr[0])
                    .arg(&gr[1])
                    .arg(&gr[2])
                    .arg(&gr[3])
                    .arg(&gr[4])
                    .arg(&gr[5])
                    .arg(&aq.sea);
                unsafe { b.launch(config(self.n_chunk)) }.ctx("chunk a structures")?;
                // The chunk's surface right after, on the same stream
                if let Some(s) = follow_up.take() {
                    s(&st, &res.buf_codes);
                }
                st.memcpy_dtoh(&res.buf_codes, &mut res.host_codes)
                    .ctx("code readback")?;
                st.synchronize().ctx("code wait")?;
                drop((pd, jd, zd));
                // Copy then check: the codes the GPU just wrote are read once from memory, the
                // check rereads them in the cache. On error, `out` is zeroed.
                out.copy_from_slice(res.host_codes.as_slice().ctx("code read")?);
                if out.contains(&-1) {
                    out.fill(0);
                    return Err("aquifer cell outside the grid".into());
                }
                self.stats.chunks.fetch_add(1, Relaxed);
                self.stats.chunks_beard.fetch_add(1, Relaxed);
                self.stats
                    .ns_copy
                    .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                Ok(skip)
            })();
            self.pool.lock().unwrap().push(res);
            rr
        });
        match result {
            Ok(skip) => {
                if let Some(j) = caves {
                    self.caves_substance(&region, lx, lz, j);
                }
                if region.served.fetch_add(1, Relaxed) + 1 == (r * r) as u32 {
                    let mut c = self.cache.lock().unwrap();
                    if c.lookup
                        .get(&(rx, rz))
                        .is_some_and(|(e, _)| Arc::ptr_eq(e, &region))
                    {
                        c.lookup.remove(&(rx, rz));
                    }
                }
                Some(skip)
            }
            Err(e) => {
                self.failed(&e, (rx, rz));
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_codes(
        &self,
        region: &Region,
        rx: i32,
        rz: i32,
        lx: i32,
        lz: i32,
        res: &mut Resources,
        out: &mut [i8],
        follow_up: Option<&mut dyn FnMut(&Arc<CudaStream>, &CudaSlice<i8>)>,
    ) -> Result<i32, String> {
        let data = self.data(region, rx, rz, res)?;
        let (aq, codes) = match (&self.aq, &data.codes) {
            (Some(a), Some(c)) => (a, c),
            _ => return Err("region without aquifer codes".into()),
        };
        let t = Instant::now();
        let (gx, h) = (16 * self.r, (self.n_chunk / 256) as i32);
        let mut b = res.stream.launch_builder(&aq.chunk);
        b.arg(codes)
            .arg(&mut res.buf_codes)
            .arg(&lx)
            .arg(&lz)
            .arg(&gx)
            .arg(&h);
        unsafe { b.launch(config(self.n_chunk)) }.ctx("code copy")?;
        // The chunk's surface right after, on the same stream
        if let Some(s) = follow_up {
            s(&res.stream, &res.buf_codes);
        }
        res.stream
            .memcpy_dtoh(&res.buf_codes, &mut res.host_codes)
            .ctx("code readback")?;
        res.stream.synchronize().ctx("code wait")?;
        // Copy then check (chunk_codes_beard): one read from memory, the check in the cache.
        out.copy_from_slice(res.host_codes.as_slice().ctx("code read")?);
        if out.contains(&-1) {
            out.fill(0);
            return Err("aquifer cell outside the grid".into());
        }
        self.stats.chunks.fetch_add(1, Relaxed);
        self.stats.chunks_aq.fetch_add(1, Relaxed);
        self.stats
            .ns_copy
            .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        Ok(data.skips[(lz * self.r + lx) as usize])
    }

    /// Enqueues on a cave batch's stream the substance of its cells, with the region's aquifer.
    fn caves_substance(&self, region: &Arc<Region>, lx: i32, lz: i32, job: &mut CaveJob) {
        let (Some(aq), Some(Ok(d))) = (&self.aq, region.data.get()) else {
            return;
        };
        let Some(cell_grid) = &d.cells else {
            return;
        };
        let skip = d.skips[(lz * self.r + lx) as usize];
        let guard: Arc<dyn std::any::Any + Send + Sync> = region.clone();
        let aqr = caves::AquiferRegion {
            f: &aq.caves,
            p: &cell_grid.p,
            st: &cell_grid.st,
            grid: cell_grid.grid,
            skip,
            sea: aq.sea,
        };
        if let Err(e) = job.enqueue_substance(aqr, guard) {
            static REPORTED: OnceLock<()> = OnceLock::new();
            if REPORTED.set(()).is_ok() {
                log::error!("GPU caves: substance: {e}");
            }
        }
    }

    fn region(&self, key: (i32, i32)) -> Arc<Region> {
        let mut c = self.cache.lock().unwrap();
        c.clock += 1;
        let h = c.clock;
        if let Some((r, t)) = c.lookup.get_mut(&key) {
            *t = h;
            return r.clone();
        }
        if c.lookup.len() >= c.capacity
            && let Some(oldest) = c
                .lookup
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| *k)
        {
            // Released when the last worker that reads it is done
            c.lookup.remove(&oldest);
            self.stats.evictions.fetch_add(1, Relaxed);
        }
        let r = Arc::new(Region::default());
        c.lookup.insert(key, (r.clone(), h));
        r
    }

    fn resource(&self) -> Result<Resources, String> {
        if let Some(r) = self.pool.lock().unwrap().pop() {
            return Ok(r);
        }
        let stream = priority_stream(&self.ctx)?;
        let buf = stream.alloc_zeros::<f32>(self.n_chunk).ctx("GPU buffer")?;
        // Ordinary pinned memory: cudarc's default (write-combined) is slow to read back
        let host = unsafe { self.ctx.alloc_pinned_with_flags::<f32>(self.n_chunk, 0) }
            .ctx("host buffer")?;
        let buf_codes = stream
            .alloc_zeros::<i8>(self.n_chunk)
            .ctx("GPU code buffer")?;
        let host_codes = unsafe { self.ctx.alloc_pinned_with_flags::<i8>(self.n_chunk, 0) }
            .ctx("host code buffer")?;
        Ok(Resources {
            stream,
            buf,
            host,
            buf_codes,
            host_codes,
        })
    }

    /// The whole region on stream `st`: axis tables, corner grids, block grid.
    fn compute(&self, rx: i32, rz: i32, st: &Arc<CudaStream>) -> Result<RegionData, String> {
        let min = [rx * self.r * 16, self.min_y, rz * self.r * 16];
        let mins: Vec<[i32; 3]> = self
            .grid_step
            .iter()
            .enumerate()
            .map(|(k, step)| {
                if k == 0 {
                    min
                } else {
                    std::array::from_fn(|i| min[i].div_euclid(step[i]) * step[i])
                }
            })
            .collect();
        let mut mem = st.alloc_zeros::<f32>(self.size).ctx("region memory")?;
        for (f, gr, nb) in &self.f_ax {
            let mut b = st.launch_builder(f);
            b.arg(&mut mem)
                .arg(&mins[*gr][0])
                .arg(&mins[*gr][1])
                .arg(&mins[*gr][2]);
            unsafe { b.launch(config(*nb)) }.ctx("axis kernel")?;
        }
        for (k, k2d, k3d, n3d) in &self.f_cv {
            if let Some((f, nb)) = k2d {
                let mut b = st.launch_builder(f);
                b.arg(&mut mem)
                    .arg(&mins[*k][0])
                    .arg(&mins[*k][1])
                    .arg(&mins[*k][2]);
                unsafe { b.launch(config(*nb)) }.ctx("2D pass")?;
            }
            let mut b = st.launch_builder(k3d);
            b.arg(&mut mem)
                .arg(&mins[*k][0])
                .arg(&mins[*k][1])
                .arg(&mins[*k][2]);
            unsafe { b.launch(config(*n3d)) }.ctx("3D grid")?;
        }
        let (codes, skips, cells) = match &self.aq {
            Some(aq) => {
                let (c, s, cell_grid) = self.aquifers(aq, &mem, min, st)?;
                (Some(c), s, Some(cell_grid))
            }
            None => {
                st.synchronize().ctx("region computation")?;
                (None, Vec::new(), None)
            }
        };
        Ok(RegionData {
            mem,
            codes,
            cells,
            skips,
        })
    }

    /// Block codes of the region (same grids as aquifer_blocks, over the whole region).
    fn aquifers(
        &self,
        aq: &Aquifers,
        mem: &CudaSlice<f32>,
        min: [i32; 3],
        st: &Arc<CudaStream>,
    ) -> Result<(CudaSlice<i8>, Vec<i32>, Cells), String> {
        let h = (self.n_chunk / 256) as i32;
        let gx = 16 * self.r;
        let (x0, z0) = (min[0], min[2]);
        let (x1, z1) = (x0 + gx - 1, z0 + gx - 1);
        let (y0, y1) = (self.min_y, self.min_y + h - 1);
        let (cx0, cy0, cz0) = ((x0 - 5) >> 4, (y0 + 1).div_euclid(12) - 1, (z0 - 5) >> 4);
        let (cx1, cy1, cz1) = (
            ((x1 - 5) >> 4) + 1,
            (y1 + 1).div_euclid(12) + 1,
            ((z1 - 5) >> 4) + 1,
        );
        let (ncx, ncy, ncz) = (cx1 - cx0 + 1, cy1 - cy0 + 1, cz1 - cz0 + 1);
        let (qx0, qz0) = ((cx0 * 16 - 48) >> 2, (cz0 * 16 - 16) >> 2);
        let (qx1, qz1) = ((cx1 * 16 + 9 + 16) >> 2, (cz1 * 16 + 9 + 16) >> 2);
        let (nqx, nqz) = (qx1 - qx0 + 1, qz1 - qz0 + 1);
        let nch = self.r * self.r;
        let (chx0, chz0) = (x0 >> 4, z0 >> 4);
        if aq.disabled {
            // Every block above the skip height takes the global fluid picker: no cell is read.
            let skips = vec![i32::MIN; nch as usize];
            let skip = st.clone_htod(&skips).ctx("skips")?;
            let p = st.alloc_zeros::<i32>(3).ctx("positions")?;
            let stt = st.alloc_zeros::<i32>(2).ctx("statuses")?;
            let mut codes = st
                .alloc_zeros::<i8>(self.n_chunk * nch as usize)
                .ctx("codes")?;
            let zero = 0i32;
            let mut l = st.launch_builder(&aq.blocks);
            l.arg(mem)
                .arg(&mut codes)
                .arg(&p)
                .arg(&stt)
                .arg(&skip)
                .arg(&x0)
                .arg(&z0)
                .arg(&gx)
                .arg(&gx)
                .arg(&self.r);
            l.arg(&self.min_y)
                .arg(&h)
                .arg(&zero)
                .arg(&zero)
                .arg(&zero)
                .arg(&zero)
                .arg(&zero)
                .arg(&zero)
                .arg(&aq.sea);
            unsafe { l.launch(config(self.n_chunk * nch as usize)) }.ctx("blocks")?;
            st.synchronize().ctx("region aquifers")?;
            return Ok((
                codes,
                skips,
                Cells {
                    p,
                    st: stt,
                    grid: [0; 6],
                },
            ));
        }
        let mut s = st.alloc_zeros::<i32>((nqx * nqz) as usize).ctx("surface")?;
        let mut p = st
            .alloc_zeros::<i32>((3 * ncx * ncy * ncz) as usize)
            .ctx("positions")?;
        let mut stt = st
            .alloc_zeros::<i32>((2 * ncx * ncy * ncz) as usize)
            .ctx("statuses")?;
        let mut skip = st.alloc_zeros::<i32>(nch as usize).ctx("skips")?;
        let mut codes = st
            .alloc_zeros::<i8>(self.n_chunk * nch as usize)
            .ctx("codes")?;
        let (lo, hi) = aq.positional;
        {
            let mut l = st.launch_builder(&aq.surface);
            l.arg(&mut s).arg(&qx0).arg(&qz0).arg(&nqx).arg(&nqz);
            unsafe { l.launch(config((nqx * nqz) as usize)) }.ctx("surface")?;
        }
        {
            let mut l = st.launch_builder(&aq.cells);
            l.arg(&mut p)
                .arg(&mut stt)
                .arg(&s)
                .arg(&qx0)
                .arg(&qz0)
                .arg(&nqx)
                .arg(&cx0)
                .arg(&cy0)
                .arg(&cz0);
            l.arg(&ncx)
                .arg(&ncy)
                .arg(&ncz)
                .arg(&lo)
                .arg(&hi)
                .arg(&aq.sea);
            unsafe { l.launch(config((ncx * ncy * ncz) as usize)) }.ctx("cells")?;
        }
        {
            let mut l = st.launch_builder(&aq.skip);
            l.arg(&mut skip)
                .arg(&s)
                .arg(&qx0)
                .arg(&qz0)
                .arg(&nqx)
                .arg(&chx0)
                .arg(&chz0)
                .arg(&self.r)
                .arg(&nch);
            l.arg(&self.min_y).arg(&h);
            unsafe { l.launch(config(nch as usize)) }.ctx("skips")?;
        }
        {
            let mut l = st.launch_builder(&aq.blocks);
            l.arg(mem)
                .arg(&mut codes)
                .arg(&p)
                .arg(&stt)
                .arg(&skip)
                .arg(&x0)
                .arg(&z0)
                .arg(&gx)
                .arg(&gx)
                .arg(&self.r);
            l.arg(&self.min_y)
                .arg(&h)
                .arg(&cx0)
                .arg(&cy0)
                .arg(&cz0)
                .arg(&ncx)
                .arg(&ncy)
                .arg(&ncz)
                .arg(&aq.sea);
            unsafe { l.launch(config(self.n_chunk * nch as usize)) }.ctx("blocks")?;
        }
        st.synchronize().ctx("region aquifers")?;
        let skips = st.clone_dtoh(&skip).ctx("skip readback")?;
        Ok((
            codes,
            skips,
            Cells {
                p,
                st: stt,
                grid: [cx0, cy0, cz0, ncx, ncy, ncz],
            },
        ))
    }
}

/// Evaluates graph roots at a list of points on the GPU, exactly like Horizon's
/// `DensitySampler::sample_value` (checks and benchmark).
/// Result: one list of values per root, in point order.
pub fn evaluate_points(
    compiler: &DensityCompiler,
    seed: i64,
    roots: &[FunctionId],
    points: &[[i32; 3]],
) -> Result<Vec<Vec<f32>>, String> {
    let src = codegen::generate_points(compiler, roots, seed).map_err(|e| e.to_string())?;
    if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP") {
        let _ = std::fs::write(f, &src);
    }
    let ctx = context()?;
    let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
    let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
    let module = load_module(&ctx, &src, arch).ctx("module load")?;
    let st = ctx.new_stream().ctx("CUDA stream")?;
    let flat: Vec<i32> = points.iter().flatten().copied().collect();
    let p = st.clone_htod(&flat).ctx("point upload")?;
    let n = points.len();
    let mut outputs = Vec::new();
    for k in 0..roots.len() {
        let f = module
            .load_function(&format!("kpt_{k}"))
            .ctx("point kernel")?;
        let mut out = st.alloc_zeros::<f32>(n.max(1)).ctx("output")?;
        let ni = n as i32;
        let mut b = st.launch_builder(&f);
        b.arg(&p).arg(&mut out).arg(&ni);
        unsafe { b.launch(config(n.max(1))) }.ctx("evaluation")?;
        st.synchronize().ctx("evaluation")?;
        let mut v = st.clone_dtoh(&out).ctx("readback")?;
        v.truncate(n);
        outputs.push(v);
    }
    Ok(outputs)
}

/// Parameters of an aquifer computation on the GPU (check milestone): a rectangle of chunks.
pub struct AquiferBatch<'a> {
    /// Roots in AquiferField::ALL order.
    pub roots: [FunctionId; 6],
    /// State (lo, hi) of the aquifers' positional random factory.
    pub positional: (u64, u64),
    pub sea_level: i32,
    pub min_y: i32,
    pub height: i32,
    /// First chunk (x, z) and number of chunks (x, z).
    pub chunk0: (i32, i32),
    pub chunks: (i32, i32),
    /// Final density of each chunk one after the other (chunk = z * nx + x), in `fill_noise` order.
    pub density: &'a [f32],
}

/// Codes of each block (0 default block, 1 air, 2 default fluid, 3 lava, +4 scheduled update, -1
/// outside the grid) and skip_sampling_above_y of each chunk.
pub fn aquifer_blocks(
    compiler: &DensityCompiler,
    seed: i64,
    b: &AquiferBatch,
) -> Result<(Vec<i8>, Vec<i32>), String> {
    let src =
        codegen::generate_aquifer(compiler, Some(b.roots), seed).map_err(|e| e.to_string())?;
    if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP") {
        let _ = std::fs::write(f, &src);
    }
    let ctx = context()?;
    let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
    let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
    let module = load_module(&ctx, &src, arch).ctx("module load")?;
    let st = ctx.new_stream().ctx("CUDA stream")?;
    let (nchx, nchz) = b.chunks;
    let nch = nchx * nchz;
    let (x0, z0) = (b.chunk0.0 * 16, b.chunk0.1 * 16);
    let (x1, z1) = ((b.chunk0.0 + nchx) * 16 - 1, (b.chunk0.1 + nchz) * 16 - 1);
    let (y0, y1) = (b.min_y, b.min_y + b.height - 1);
    // Cell grid (NoiseBasedAquifer::new over the whole rectangle)
    let (cx0, cy0, cz0) = ((x0 - 5) >> 4, (y0 + 1).div_euclid(12) - 1, (z0 - 5) >> 4);
    let (cx1, cy1, cz1) = (
        ((x1 - 5) >> 4) + 1,
        (y1 + 1).div_euclid(12) + 1,
        ((z1 - 5) >> 4) + 1,
    );
    let (ncx, ncy, ncz) = (cx1 - cx0 + 1, cy1 - cy0 + 1, cz1 - cz0 + 1);
    // Surface columns: compute_fluid offsets (x from -48 to +16, z from -16 to +16) around the
    // cells
    let (qx0, qz0) = ((cx0 * 16 - 48) >> 2, (cz0 * 16 - 16) >> 2);
    let (qx1, qz1) = ((cx1 * 16 + 9 + 16) >> 2, (cz1 * 16 + 9 + 16) >> 2);
    let (nqx, nqz) = (qx1 - qx0 + 1, qz1 - qz0 + 1);
    let mut s = st.alloc_zeros::<i32>((nqx * nqz) as usize).ctx("surface")?;
    let mut p = st
        .alloc_zeros::<i32>((3 * ncx * ncy * ncz) as usize)
        .ctx("positions")?;
    let mut stt = st
        .alloc_zeros::<i32>((2 * ncx * ncy * ncz) as usize)
        .ctx("statuses")?;
    let mut skip = st.alloc_zeros::<i32>(nch as usize).ctx("skips")?;
    let dens = st.clone_htod(b.density).ctx("densities")?;
    let mut codes = st.alloc_zeros::<i8>(b.density.len()).ctx("codes")?;
    let f = |n: &str| module.load_function(n).ctx(n);
    let (lo, hi) = b.positional;
    {
        let k = f("kaq_surface")?;
        let mut l = st.launch_builder(&k);
        l.arg(&mut s).arg(&qx0).arg(&qz0).arg(&nqx).arg(&nqz);
        unsafe { l.launch(config((nqx * nqz) as usize)) }.ctx("surface")?;
    }
    {
        let k = f("kaq_cells")?;
        let mut l = st.launch_builder(&k);
        l.arg(&mut p)
            .arg(&mut stt)
            .arg(&s)
            .arg(&qx0)
            .arg(&qz0)
            .arg(&nqx)
            .arg(&cx0)
            .arg(&cy0)
            .arg(&cz0);
        l.arg(&ncx)
            .arg(&ncy)
            .arg(&ncz)
            .arg(&lo)
            .arg(&hi)
            .arg(&b.sea_level);
        unsafe { l.launch(config((ncx * ncy * ncz) as usize)) }.ctx("cells")?;
    }
    {
        let k = f("kaq_skip")?;
        let mut l = st.launch_builder(&k);
        l.arg(&mut skip)
            .arg(&s)
            .arg(&qx0)
            .arg(&qz0)
            .arg(&nqx)
            .arg(&b.chunk0.0)
            .arg(&b.chunk0.1)
            .arg(&nchx)
            .arg(&nch);
        l.arg(&b.min_y).arg(&b.height);
        unsafe { l.launch(config(nch as usize)) }.ctx("skips")?;
    }
    {
        let k = f("kaq_blocks")?;
        let mut l = st.launch_builder(&k);
        l.arg(&dens)
            .arg(&mut codes)
            .arg(&p)
            .arg(&stt)
            .arg(&skip)
            .arg(&b.chunk0.0)
            .arg(&b.chunk0.1)
            .arg(&nchx)
            .arg(&nch);
        l.arg(&b.min_y)
            .arg(&b.height)
            .arg(&cx0)
            .arg(&cy0)
            .arg(&cz0)
            .arg(&ncx)
            .arg(&ncy)
            .arg(&ncz)
            .arg(&b.sea_level);
        unsafe { l.launch(config(b.density.len())) }.ctx("blocks")?;
    }
    st.synchronize().ctx("aquifers")?;
    Ok((
        st.clone_dtoh(&codes).ctx("code readback")?,
        st.clone_dtoh(&skip).ctx("skip readback")?,
    ))
}

/// Evaluates noises (`Noise::get3`) at a list of points on the GPU (surface checks).
pub fn evaluate_noises(
    compiler: &DensityCompiler,
    noises: &[&dyn horizon_core::noise::Noise],
    points: &[[f64; 3]],
) -> Result<Vec<Vec<f32>>, String> {
    let src = codegen::generate_noises(compiler, noises).map_err(|e| e.to_string())?;
    let ctx = context()?;
    let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
    let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
    let module = load_module(&ctx, &src, arch).ctx("module load")?;
    let st = ctx.new_stream().ctx("CUDA stream")?;
    let flat: Vec<f64> = points.iter().flatten().copied().collect();
    let p = st.clone_htod(&flat).ctx("point upload")?;
    let n = points.len();
    let mut outputs = Vec::new();
    for k in 0..noises.len() {
        let f = module
            .load_function(&format!("knz_{k}"))
            .ctx("noise kernel")?;
        let mut out = st.alloc_zeros::<f32>(n.max(1)).ctx("output")?;
        let ni = n as i32;
        let mut b = st.launch_builder(&f);
        b.arg(&p).arg(&mut out).arg(&ni);
        unsafe { b.launch(config(n.max(1))) }.ctx("evaluation")?;
        st.synchronize().ctx("evaluation")?;
        let mut v = st.clone_dtoh(&out).ctx("readback")?;
        v.truncate(n);
        outputs.push(v);
    }
    Ok(outputs)
}

/// Draws of a Xoroshiro positional random factory (lo, hi) at each position: first next_float,
/// first next_double, three successive next_float.
pub fn positional_draws(
    compiler: &DensityCompiler,
    positional: (u64, u64),
    points: &[[i32; 3]],
) -> Result<(Vec<f32>, Vec<f64>, Vec<f32>), String> {
    let src = codegen::generate_noises(compiler, &[]).map_err(|e| e.to_string())?;
    let ctx = context()?;
    let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
    let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
    let module = load_module(&ctx, &src, arch).ctx("module load")?;
    let st = ctx.new_stream().ctx("CUDA stream")?;
    let flat: Vec<i32> = points.iter().flatten().copied().collect();
    let p = st.clone_htod(&flat).ctx("point upload")?;
    let n = points.len().max(1);
    let mut f = st.alloc_zeros::<f32>(n).ctx("output")?;
    let mut d = st.alloc_zeros::<f64>(n).ctx("output")?;
    let mut f3 = st.alloc_zeros::<f32>(3 * n).ctx("output")?;
    let k = module.load_function("kdraws").ctx("kdraws")?;
    let (ni, lo, hi) = (points.len() as i32, positional.0, positional.1);
    let mut b = st.launch_builder(&k);
    b.arg(&p)
        .arg(&mut f)
        .arg(&mut d)
        .arg(&mut f3)
        .arg(&ni)
        .arg(&lo)
        .arg(&hi);
    unsafe { b.launch(config(n)) }.ctx("draws")?;
    st.synchronize().ctx("draws")?;
    Ok((
        st.clone_dtoh(&f).ctx("readback")?,
        st.clone_dtoh(&d).ctx("readback")?,
        st.clone_dtoh(&f3).ctx("readback")?,
    ))
}

/// S2 milestone of the surface: quarts picked by the biome zoom (seed already obfuscated) and
/// height-adjusted temperatures for each (base temperature, frozen modifier).
#[allow(clippy::type_complexity)]
pub fn surface_s2_check(
    compiler: &DensityCompiler,
    zoom_seed: i64,
    points: &[[i32; 3]],
    biomes: &[(f32, bool)],
    sea_level: i32,
    height: &horizon_core::noise::SimplexNoise,
    freeze: &horizon_core::noise::NoiseStack,
    info: &horizon_core::noise::SimplexNoise,
) -> Result<(Vec<[i32; 3]>, Vec<Vec<f32>>), String> {
    let src =
        codegen::generate_surface_s2(compiler, height, freeze, info).map_err(|e| e.to_string())?;
    let ctx = context()?;
    let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
    let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
    let module = load_module(&ctx, &src, arch).ctx("module load")?;
    let st = ctx.new_stream().ctx("CUDA stream")?;
    let flat: Vec<i32> = points.iter().flatten().copied().collect();
    let p = st.clone_htod(&flat).ctx("point upload")?;
    let n = points.len();
    let mut q = st.alloc_zeros::<i32>(3 * n.max(1)).ctx("quarters")?;
    {
        let k = module.load_function("ksf_quarters").ctx("ksf_quarters")?;
        let ni = n as i32;
        let mut b = st.launch_builder(&k);
        b.arg(&p).arg(&mut q).arg(&ni).arg(&zoom_seed);
        unsafe { b.launch(config(n.max(1))) }.ctx("zoom")?;
    }
    let bases: Vec<f32> = biomes.iter().map(|b| b.0).collect();
    let frozen_flags: Vec<i32> = biomes.iter().map(|b| i32::from(b.1)).collect();
    let bd = st.clone_htod(&bases).ctx("bases")?;
    let gd = st.clone_htod(&frozen_flags).ctx("frozen_flags")?;
    let mut t = st
        .alloc_zeros::<f32>((n * biomes.len()).max(1))
        .ctx("temperatures")?;
    {
        let k = module
            .load_function("ksf_temperatures")
            .ctx("ksf_temperatures")?;
        let (ni, nb) = (n as i32, biomes.len() as i32);
        let mut b = st.launch_builder(&k);
        b.arg(&p)
            .arg(&bd)
            .arg(&gd)
            .arg(&nb)
            .arg(&mut t)
            .arg(&ni)
            .arg(&sea_level);
        unsafe { b.launch(config((n * biomes.len()).max(1))) }.ctx("temperatures")?;
    }
    st.synchronize().ctx("S2 milestone")?;
    let qv = st.clone_dtoh(&q).ctx("readback")?;
    let tv = st.clone_dtoh(&t).ctx("readback")?;
    let quarters = (0..n)
        .map(|i| [qv[3 * i], qv[3 * i + 1], qv[3 * i + 2]])
        .collect();
    let temperatures = (0..biomes.len())
        .map(|k| tv[k * n..(k + 1) * n].to_vec())
        .collect();
    Ok((quarters, temperatures))
}

/// Surface rule in neutral form (translated from horizon-world's `MaterialRule`).
#[derive(Clone, Debug)]
pub enum SurfaceRule {
    Empty,
    Block(u32),
    Sequence(Vec<usize>),
    Condition {
        condition: usize,
        rule: usize,
    },
    Bandlands,
    /// `positive_y`: the y outside of which the density is never positive (nothing is placed
    /// there).
    OreVein {
        blocks: [u32; 3],
        raw_chance: f32,
        density: FunctionId,
        richness: FunctionId,
        gap: FunctionId,
        positive_y: [i32; 2],
    },
}
/// A positional random factory the surface draws from, as a kernel rebuilds it.
#[derive(Clone, Copy, Debug)]
pub enum Positional {
    /// `XoroshiroPositionalRandomFactory`: its (lo, hi) state.
    Xoroshiro(u64, u64),
    /// `LegacyPositionalRandomFactory` (the Nether, the End): its seed.
    Legacy(i64),
}

/// Surface condition in neutral form (anchors already resolved, names already translated to
/// indices).
#[derive(Clone, Debug)]
pub enum SurfaceCondition {
    Hole,
    Steep,
    Biomes(Vec<u32>),
    Temperature,
    AbovePreliminary,
    Noise {
        noise: usize,
        min: f64,
        max: f64,
        is_3d: bool,
    },
    YAbove {
        anchor: i32,
        multiplier: i32,
        add_stone: bool,
    },
    Water {
        offset: i32,
        multiplier: i32,
        add_stone: bool,
    },
    StoneDepth {
        offset: i32,
        add_surface: bool,
        secondary_range: i32,
        ceiling: bool,
    },
    Gradient {
        low: i32,
        high: i32,
        positional: Positional,
    },
    Not(usize),
}
/// Everything a world's surface uses (S3 milestone).
pub struct SurfaceSetup<'a> {
    pub rules: Vec<SurfaceRule>,
    pub conditions: Vec<SurfaceCondition>,
    pub root: usize,
    pub noises: Vec<&'a dyn horizon_core::noise::Noise>,
    pub surface_noise: &'a dyn horizon_core::noise::Noise,
    pub secondary_noise: &'a dyn horizon_core::noise::Noise,
    pub clay_noise: &'a dyn horizon_core::noise::Noise,
    pub bands: [u32; 192],
    pub root_positional: Positional,
    pub ore_positional: (u64, u64),
    pub surface_level: FunctionId,
    /// Per biome: base temperature and frozen modifier.
    pub temperatures: Vec<(f32, bool)>,
    pub temperature_noises: (
        &'a horizon_core::noise::SimplexNoise,
        &'a horizon_core::noise::NoiseStack,
        &'a horizon_core::noise::SimplexNoise,
    ),
    pub zoom_seed: i64,
    pub sea_level: i32,
    pub min_y: i32,
    pub height: i32,
    /// The level whose biomes `chunk_states` is given; the noise may cover less of it (the
    /// Nether's noise covers 128 of its 256 blocks).
    pub biome_min_y: i32,
    pub biome_height: i32,
    /// Eroded badlands pillars and icebergs (S4 milestone); `None`: columns without extensions.
    pub extensions: Option<SurfaceExtensions<'a>>,
    /// The fill states per code (& 3): default block, air, default fluid, lava.
    pub fill_states: [u32; 4],
    /// The fluid states the rule can write (swamp water): post-processing marks.
    pub fluid_states: Vec<u32>,
}
/// The surface extensions (Horizon's `material_extensions.rs`).
pub struct SurfaceExtensions<'a> {
    pub eroded_badlands: Option<u32>,
    /// At most two biomes (frozen_ocean, deep_frozen_ocean).
    pub frozen_oceans: Vec<u32>,
    /// badlands_surface, badlands_pillar, badlands_pillar_roof, iceberg_surface, iceberg_pillar,
    /// iceberg_pillar_roof.
    pub noises: [&'a dyn horizon_core::noise::Noise; 6],
    pub default_block: u32,
    pub packed_ice: u32,
    pub snow: u32,
}

/// A program of grids (veins, preliminary surface) loaded on the GPU.
struct Volume {
    f_ax: Vec<Launch>,
    f_cv: Vec<(usize, Option<(CudaFunction, usize)>, CudaFunction, usize)>,
    step: Vec<[i32; 3]>,
    size: usize,
}
impl Volume {
    fn load(
        ctx: &Arc<CudaContext>,
        arch: &'static str,
        prog: codegen::Program,
    ) -> Result<Self, String> {
        let module = load_module(ctx, &prog.src, arch).ctx("vein module")?;
        let f = |name: &str| module.load_function(name).ctx(name);
        let mut f_ax = Vec::new();
        for &(k, gr, nb) in &prog.axes {
            f_ax.push((f(&format!("kax_{k}"))?, gr, nb));
        }
        let mut f_cv = Vec::new();
        for k in (1..prog.grids.len()).chain(std::iter::once(0)) {
            let gr = prog.grids[k];
            let k2d = if prog.edges[k] > 0 {
                Some((f(&format!("k2d_{k}"))?, gr.columns() * prog.edges[k]))
            } else {
                None
            };
            f_cv.push((k, k2d, f(&format!("k3d_{k}"))?, gr.n()));
        }
        Ok(Self {
            f_ax,
            f_cv,
            step: prog.grids.iter().map(|g| g.step).collect(),
            size: prog.size,
        })
    }
    /// All the program's grids at origin `min`; the block grid is at the start of `mem`, at ((z *
    /// CGX) + x) * CGY + y.
    fn grid(
        &self,
        st: &Arc<CudaStream>,
        mem: &mut CudaSlice<f32>,
        min: [i32; 3],
    ) -> Result<(), String> {
        let mins: Vec<[i32; 3]> = self
            .step
            .iter()
            .enumerate()
            .map(|(k, step)| {
                if k == 0 {
                    min
                } else {
                    std::array::from_fn(|i| min[i].div_euclid(step[i]) * step[i])
                }
            })
            .collect();
        st.memset_zeros(mem).ctx("program memory")?;
        for (f, gr, nb) in &self.f_ax {
            let mut b = st.launch_builder(f);
            b.arg(&mut *mem)
                .arg(&mins[*gr][0])
                .arg(&mins[*gr][1])
                .arg(&mins[*gr][2]);
            unsafe { b.launch(config(*nb)) }.ctx("axis kernel")?;
        }
        for (k, k2d, k3d, n3d) in &self.f_cv {
            if let Some((f, nb)) = k2d {
                let mut b = st.launch_builder(f);
                b.arg(&mut *mem)
                    .arg(&mins[*k][0])
                    .arg(&mins[*k][1])
                    .arg(&mins[*k][2]);
                unsafe { b.launch(config(*nb)) }.ctx("2D pass")?;
            }
            let mut b = st.launch_builder(k3d);
            b.arg(&mut *mem)
                .arg(&mins[*k][0])
                .arg(&mins[*k][1])
                .arg(&mins[*k][2]);
            unsafe { b.launch(config(*n3d)) }.ctx("3D grid")?;
        }
        Ok(())
    }
}

/// The buffers of a surface call, kept from one chunk to the next (one set per worker at a time).
struct SurfaceResources {
    st: Arc<CudaStream>,
    pl: CudaSlice<f32>,
    cd: CudaSlice<i8>,
    bd: CudaSlice<u16>,
    /// Iceberg blocks (state + 1).
    ice: CudaSlice<u32>,
    /// Per block: above, below, water; per column: 10 integers (ksf_heads) and the secondary noise.
    meta: CudaSlice<i32>,
    coli: CudaSlice<i32>,
    cold: CudaSlice<f64>,
    end: CudaSlice<u32>,
    err: CudaSlice<i32>,
    host: crate::cuda::driver::PinnedHostSlice<u32>,
    host_err: crate::cuda::driver::PinnedHostSlice<i32>,
    /// Compact sections, heightmaps, marks and error (`enable_sections`)
    sec: CudaSlice<u32>,
    host_sec: crate::cuda::driver::PinnedHostSlice<u32>,
    /// Marks of each column (ksec_heights for ksec_marks)
    nm: CudaSlice<u32>,
}
/// Building the compact sections of the fill (`SurfaceGpu::enable_sections`).
struct SectionsGpu {
    _module: Arc<crate::cuda::driver::CudaModule>,
    k_sections: CudaFunction,
    k_heights: CudaFunction,
    k_marks: CudaFunction,
    flags: CudaSlice<u8>,
    nflags: u32,
    air: u32,
    test: bool,
}

#[derive(Default)]
struct SurfaceStats {
    chunks: AtomicU64,
    ns: AtomicU64,
    fallbacks: AtomicU64,
    /// Compact sections: chunks placed, fallen back to raw states, mismatches with the CPU (test
    /// mode)
    sections: AtomicU64,
    sections_fallbacks: AtomicU64,
    sections_mismatches: AtomicU64,
    /// HORIZON_GPU_SURFACE_PROFILE: veins, preliminary surface, columns, blocks, uploads and
    /// readback (ns)
    phases: [AtomicU64; 5],
}

/// A world's surface on the GPU: final states of a chunk after filling and surface.
pub struct SurfaceGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<crate::cuda::driver::CudaModule>,
    k_heads: CudaFunction,
    k_descent: CudaFunction,
    k_blocks: CudaFunction,
    volumes: Vec<Volume>,
    level: Volume,
    height: i32,
    biome_height: i32,
    pool: Mutex<Vec<SurfaceResources>>,
    stats: SurfaceStats,
    /// Per region of LEVEL_R x LEVEL_R chunks, computed once for its chunks; the oldest are dropped
    /// beyond `capacity`.
    regions: Mutex<Regions>,
    capacity: usize,
    /// The vein grids of a region: starting y and height.
    vlo: i32,
    vh: i32,
    sections: OnceLock<Option<SectionsGpu>>,
}
/// The preliminary surface of a region (y = 0 plane, by z then x, on the host) and the densities of
/// its veins (on the GPU: one grid per root, at ((z * SF_VR) + x) * SF_VH + y - SF_VLO).
struct RegionSurface {
    plan: Vec<f32>,
    veins: CudaSlice<f32>,
}
type RegionCell = Arc<std::sync::OnceLock<Result<Arc<RegionSurface>, String>>>;
#[derive(Default)]
struct Regions {
    lookup: std::collections::HashMap<(i32, i32), RegionCell>,
    order: std::collections::VecDeque<(i32, i32)>,
}
impl Drop for SurfaceGpu {
    fn drop(&mut self) {
        if let Some(n) = std::num::NonZero::new(self.stats.chunks.load(Relaxed)) {
            eprintln!(
                "GPU surface: {n} chunks ({} us per call on average), {} CPU fallbacks; compact sections: {} chunks, {} through raw states, {} mismatches with the CPU (test mode)",
                self.stats.ns.load(Relaxed) / n / 1000,
                self.stats.fallbacks.load(Relaxed),
                self.stats.sections.load(Relaxed),
                self.stats.sections_fallbacks.load(Relaxed),
                self.stats.sections_mismatches.load(Relaxed),
            );
            if std::env::var_os("HORIZON_GPU_SURFACE_PROFILE").is_some() {
                let p = self
                    .stats
                    .phases
                    .each_ref()
                    .map(|a| a.load(Relaxed) / n / 1000);
                eprintln!(
                    "GPU surface: per chunk, veins {} us, preliminary surface {} us, columns {} us, blocks {} us, uploads and readback {} us",
                    p[0], p[1], p[2], p[3], p[4]
                );
            }
        }
    }
}
impl std::fmt::Debug for SurfaceGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SurfaceGpu({} veins)", self.volumes.len())
    }
}
impl SurfaceGpu {
    pub fn new(
        compiler: &DensityCompiler,
        setup: &SurfaceSetup,
        seed: i64,
    ) -> Result<Self, String> {
        let prog = codegen::generate_surface(compiler, setup, seed).map_err(|e| e.to_string())?;
        if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP_SURFACE") {
            let f = std::path::PathBuf::from(f);
            let _ = std::fs::write(&f, &prog.src);
            for (k, v) in prog.volumes.iter().enumerate() {
                let _ = std::fs::write(f.with_extension(format!("vein{k}.cu")), &v.src);
            }
            let _ = std::fs::write(f.with_extension("level.cu"), &prog.level.src);
        }
        let ctx = context()?;
        let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
        let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
        let module = load_module(&ctx, &prog.src, arch).ctx("surface module")?;
        let k_heads = module.load_function("ksf_heads").ctx("ksf_heads")?;
        let k_descent = module.load_function("ksf_descent").ctx("ksf_descent")?;
        let k_blocks = module.load_function("ksf_blocks").ctx("ksf_blocks")?;
        let mut volumes = Vec::new();
        for v in prog.volumes {
            volumes.push(Volume::load(&ctx, arch, v)?);
        }
        let level = Volume::load(&ctx, arch, prog.level)?;
        let capacity = std::env::var("HORIZON_GPU_SURFACE_REGIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(48);
        Ok(Self {
            ctx,
            _module: module,
            k_heads,
            k_descent,
            k_blocks,
            volumes,
            level,
            height: setup.height,
            biome_height: setup.biome_height,
            pool: Mutex::new(Vec::new()),
            stats: SurfaceStats::default(),
            regions: Mutex::new(Regions::default()),
            capacity,
            vlo: prog.veins_y.0,
            vh: prog.veins_y.1,
            sections: OnceLock::new(),
        })
    }
    /// HORIZON_GPU_SECTIONS: the GPU also builds the compact sections of the fill, its heightmaps
    /// and its marks (sections.cu), which `fill_noise_sections` places as they are. `flags`: per
    /// state, 1 air, 2 fluid, 4 blocks motion (`TerrainChunk::section_flags`); `test`: the states
    /// come back too, for comparison. Once per dimension; false if the module does not load.
    pub fn enable_sections(&self, flags: &[u8], air: u32, test: bool) -> bool {
        self.sections
            .get_or_init(|| {
                let r = (|| -> Result<SectionsGpu, String> {
                    let (major, min) = self
                        .ctx
                        .compute_capability()
                        .ctx("GPU compute capability")?;
                    let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
                    let module = load_module(&self.ctx, include_str!("sections.cu"), arch)
                        .ctx("section module")?;
                    let st = self.ctx.new_stream().ctx("CUDA stream")?;
                    let flags_d = st.clone_htod(flags).ctx("state flags")?;
                    st.synchronize().ctx("state flags")?;
                    Ok(SectionsGpu {
                        k_sections: module.load_function("ksec_sections").ctx("ksec_sections")?,
                        k_heights: module.load_function("ksec_heights").ctx("ksec_heights")?,
                        k_marks: module.load_function("ksec_marks").ctx("ksec_marks")?,
                        _module: module,
                        flags: flags_d,
                        nflags: flags.len() as u32,
                        air,
                        test,
                    })
                })();
                r.map_err(|e| log::error!("GPU sections: no sections on the GPU: {e}"))
                    .ok()
            })
            .is_some()
    }
    /// A chunk whose GPU compact sections are placed (`ok`) or go through the raw states.
    pub fn note_sections(&self, ok: bool) {
        if ok {
            self.stats.sections.fetch_add(1, Relaxed);
        } else {
            self.stats.sections_fallbacks.fetch_add(1, Relaxed);
        }
    }
    /// Test mode: compact sections that differ from the CPU's.
    pub fn note_sections_mismatch(&self) {
        self.stats.sections_mismatches.fetch_add(1, Relaxed);
    }
    /// u32 words of a chunk's section output: `SECTION_WORDS` per section, 512 heights (ocean then
    /// surface), 2 counters and 2 x `SECTION_MARKS` marks, then the error word.
    pub fn sections_len(&self) -> usize {
        (self.height as usize / 16) * SECTION_WORDS + 512 + 2 + 2 * SECTION_MARKS + 1
    }
    /// Region (rx, rz): preliminary surface and veins, computed at the first chunk that asks for
    /// it.
    fn region(&self, st: &Arc<CudaStream>, rx: i32, rz: i32) -> Result<Arc<RegionSurface>, String> {
        let cell = {
            let mut p = self.regions.lock().unwrap();
            match p.lookup.get(&(rx, rz)) {
                Some(c) => c.clone(),
                None => {
                    let c: RegionCell = Arc::new(std::sync::OnceLock::new());
                    p.lookup.insert((rx, rz), c.clone());
                    p.order.push_back((rx, rz));
                    // Released when the last chunk that reads it is done
                    while p.order.len() > self.capacity {
                        if let Some(v) = p.order.pop_front() {
                            p.lookup.remove(&v);
                        }
                    }
                    c
                }
            }
        };
        cell.get_or_init(|| {
            let vr = 16 * codegen::LEVEL_R;
            let n = (vr * vr) as usize;
            let mut mem = st
                .alloc_zeros::<f32>(self.level.size)
                .ctx("preliminary surface memory")?;
            self.level.grid(st, &mut mem, [rx * vr, 0, rz * vr])?;
            let plan = st
                .clone_dtoh(&mem.slice(0..n))
                .ctx("preliminary surface readback")?;
            let vn = n * self.vh as usize;
            let mut veins = st
                .alloc_zeros::<f32>(vn * self.volumes.len().max(1))
                .ctx("region veins")?;
            for (k, v) in self.volumes.iter().enumerate() {
                let mut m = st.alloc_zeros::<f32>(v.size).ctx("vein memory")?;
                v.grid(st, &mut m, [rx * vr, self.vlo, rz * vr])?;
                st.memcpy_dtod(&m.slice(0..vn), &mut veins.slice_mut(k * vn..(k + 1) * vn))
                    .ctx("vein copy")?;
            }
            // Read afterwards from the other workers' streams
            st.synchronize().ctx("surface region")?;
            Ok(Arc::new(RegionSurface { plan, veins }))
        })
        .clone()
    }
    /// Number of biomes `chunk_states` expects: the quarts of the 3 x 3 chunks around the chunk.
    pub fn biome_count(&self) -> usize {
        9 * 16 * (self.biome_height as usize / 4)
    }
    /// The chunk went through the CPU.
    pub fn note_cpu_fallback(&self) {
        self.stats.fallbacks.fetch_add(1, Relaxed);
    }
    fn resource(&self) -> Result<SurfaceResources, String> {
        if let Some(r) = self.pool.lock().unwrap().pop() {
            return Ok(r);
        }
        let n = 256 * self.height as usize;
        let st = priority_stream(&self.ctx)?;
        let res = SurfaceResources {
            pl: st.alloc_zeros::<f32>(256).ctx("preliminary surface")?,
            cd: st.alloc_zeros::<i8>(n).ctx("codes")?,
            bd: st.alloc_zeros::<u16>(self.biome_count()).ctx("biomes")?,
            ice: st.alloc_zeros::<u32>(n).ctx("icebergs")?,
            meta: st.alloc_zeros::<i32>(3 * n).ctx("descent")?,
            coli: st.alloc_zeros::<i32>(10 * 256).ctx("columns")?,
            cold: st.alloc_zeros::<f64>(256).ctx("columns")?,
            end: st.alloc_zeros::<u32>(n).ctx("final states")?,
            err: st.alloc_zeros::<i32>(1).ctx("error")?,
            // Ordinary pinned memory (cudarc's default, write-combined, is slow to read back)
            host: unsafe { self.ctx.alloc_pinned_with_flags::<u32>(n, 0) }.ctx("host buffer")?,
            host_err: unsafe { self.ctx.alloc_pinned_with_flags::<i32>(1, 0) }
                .ctx("host buffer")?,
            sec: st.alloc_zeros::<u32>(self.sections_len()).ctx("sections")?,
            nm: st.alloc_zeros::<u32>(512).ctx("column marks")?,
            host_sec: unsafe {
                self.ctx
                    .alloc_pinned_with_flags::<u32>(self.sections_len(), 0)
            }
            .ctx("host buffer")?,
            st,
        };
        // The buffers are allocated and zeroed on their own stream, but `chunk_blocks_surface` fills
        // and reads them on the stream of the chunk's codes, which does not wait for this one (event
        // tracking is off): the zeroing could land after the upload of the biomes and of the
        // preliminary surface, and the chunk got biome 0 (badlands) and a surface at y 0.
        res.st.synchronize().ctx("surface buffers")?;
        Ok(res)
    }
    /// The chunk's final states (fill then surface, extensions included) in fill_noise order: the
    /// state on 30 bits, bit 31 when the fill schedules a fluid update (code + 4 on water or lava),
    /// bit 30 when the surface writes a fluid state there. `codes`: the codes of the GPU fill;
    /// `biomes`: the quarts of the 3 x 3 chunks (dz then dx from -1 to +1, each chunk by sections).
    /// `Ok(false)`: a condition could not be evaluated, the chunk must go through the CPU.
    /// `GpuFinalDensity::chunk_blocks_surface` does the same without sending the codes back to the
    /// GPU.
    pub fn chunk_states(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        codes: &[i8],
        biomes: &[u16],
        out: &mut [u32],
    ) -> Result<bool, String> {
        let n = 256 * self.height as usize;
        if codes.len() != n || out.len() != n || biomes.len() != self.biome_count() {
            return Err("unexpected sizes".into());
        }
        let t = Instant::now();
        let mut res = self.resource()?;
        let st = res.st.clone();
        let r = (|| {
            st.memcpy_htod(codes, &mut res.cd).ctx("codes")?;
            let _region = self.enqueue(&mut res, &st, None, chunk_x, chunk_z, biomes, false)?;
            st.synchronize().ctx("surface wait")?;
            self.read_back(&mut res, out, None)
        })();
        self.put_back(res, r.is_ok(), t);
        r
    }
    /// The chunk's surface queued on `st`, after the kernel that writes its codes there (`cd`,
    /// otherwise the codes already uploaded into `res`). The states are read by `read_back` once
    /// `st` is waited for; the returned region must stay alive until then.
    fn enqueue(
        &self,
        res: &mut SurfaceResources,
        st: &Arc<CudaStream>,
        cd: Option<&CudaSlice<i8>>,
        chunk_x: i32,
        chunk_z: i32,
        biomes: &[u16],
        sections: bool,
    ) -> Result<Arc<RegionSurface>, String> {
        let n = 256 * self.height as usize;
        let profile = cd.is_none() && std::env::var_os("HORIZON_GPU_SURFACE_PROFILE").is_some();
        let mut t = Instant::now();
        let mut phase = |k: usize| -> Result<(), String> {
            if profile {
                st.synchronize().ctx("profile")?;
                self.stats.phases[k].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                t = Instant::now();
            }
            Ok(())
        };
        let r = codegen::LEVEL_R;
        let (rx, rz) = (chunk_x.div_euclid(r), chunk_z.div_euclid(r));
        let region = self.region(st, rx, rz)?;
        phase(0)?;
        let (ox, oz) = ((chunk_x - rx * r) * 16, (chunk_z - rz * r) * 16);
        let mut pl = [0f32; 256];
        for lz in 0..16 {
            for lx in 0..16 {
                pl[(lz * 16 + lx) as usize] = region.plan[((oz + lz) * 16 * r + ox + lx) as usize];
            }
        }
        st.memcpy_htod(&pl[..], &mut res.pl)
            .ctx("preliminary surface")?;
        phase(1)?;
        st.memcpy_htod(biomes, &mut res.bd).ctx("biomes")?;
        st.memset_zeros(&mut res.err).ctx("error")?;
        phase(4)?;
        st.memset_zeros(&mut res.ice).ctx("icebergs")?;
        st.memset_zeros(&mut res.meta).ctx("descent")?;
        let cd: &CudaSlice<i8> = match cd {
            Some(c) => c,
            None => &res.cd,
        };
        let mut b = st.launch_builder(&self.k_heads);
        b.arg(cd)
            .arg(&res.bd)
            .arg(&res.pl)
            .arg(&mut res.ice)
            .arg(&mut res.coli)
            .arg(&mut res.cold)
            .arg(&mut res.err)
            .arg(&chunk_x)
            .arg(&chunk_z);
        // One warp per column
        unsafe { b.launch(config(256 * 32)) }.ctx("surface columns")?;
        let mut b = st.launch_builder(&self.k_descent);
        b.arg(cd).arg(&mut res.coli).arg(&mut res.meta);
        unsafe { b.launch(config(256 * 32)) }.ctx("surface descent")?;
        phase(2)?;
        let mut b = st.launch_builder(&self.k_blocks);
        b.arg(cd)
            .arg(&res.bd)
            .arg(&region.veins)
            .arg(&res.ice)
            .arg(&res.meta)
            .arg(&res.coli)
            .arg(&res.cold)
            .arg(&mut res.end)
            .arg(&mut res.err)
            .arg(&chunk_x)
            .arg(&chunk_z)
            .arg(&ox)
            .arg(&oz);
        unsafe { b.launch(config(n)) }.ctx("surface blocks")?;
        phase(3)?;
        let sec = if sections {
            self.sections.get().and_then(Option::as_ref)
        } else {
            None
        };
        if let Some(sg) = sec {
            let ns = self.height as usize / 16;
            let (a, b) = (ns * SECTION_WORDS, ns * SECTION_WORDS + 512);
            let end_mq = b + 2 + 2 * SECTION_MARKS;
            st.memset_zeros(&mut res.sec.slice_mut(end_mq..end_mq + 1))
                .ctx("section error")?;
            let h = self.height;
            let cap = SECTION_MARKS as i32;
            {
                let hm = res.sec.slice(a..b);
                let mut l = st.launch_builder(&sg.k_heights);
                l.arg(&res.end)
                    .arg(&sg.flags)
                    .arg(&sg.nflags)
                    .arg(&sg.air)
                    .arg(&h)
                    .arg(&hm)
                    .arg(&mut res.nm);
                unsafe {
                    l.launch(LaunchConfig {
                        grid_dim: (256, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .ctx("section heights")?;
            }
            {
                let mq = res.sec.slice(b..end_mq);
                let er = res.sec.slice(end_mq..end_mq + 1);
                let mut l = st.launch_builder(&sg.k_marks);
                l.arg(&res.end)
                    .arg(&sg.air)
                    .arg(&h)
                    .arg(&res.nm)
                    .arg(&mq)
                    .arg(&cap)
                    .arg(&er);
                unsafe {
                    l.launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .ctx("section marks")?;
            }
            {
                let out = res.sec.slice(0..a);
                let er = res.sec.slice(end_mq..end_mq + 1);
                let mut l = st.launch_builder(&sg.k_sections);
                l.arg(&res.end)
                    .arg(&sg.flags)
                    .arg(&sg.nflags)
                    .arg(&sg.air)
                    .arg(&h)
                    .arg(&out)
                    .arg(&er);
                unsafe {
                    l.launch(LaunchConfig {
                        grid_dim: (ns as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .ctx("sections")?;
            }
            st.memcpy_dtoh(&res.sec, &mut res.host_sec)
                .ctx("section readback")?;
            if sg.test {
                st.memcpy_dtoh(&res.end, &mut res.host)
                    .ctx("state readback")?;
            }
        } else {
            st.memcpy_dtoh(&res.end, &mut res.host)
                .ctx("state readback")?;
        }
        st.memcpy_dtoh(&res.err, &mut res.host_err)
            .ctx("error readback")?;
        if profile {
            st.synchronize().ctx("profile")?;
            self.stats.phases[4].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        }
        Ok(region)
    }
    /// The states of a queued surface, its stream waited for (event tracking off: pinned memory
    /// does not wait by itself).
    fn read_back(
        &self,
        res: &mut SurfaceResources,
        out: &mut [u32],
        sections: Option<&mut Vec<u32>>,
    ) -> Result<bool, String> {
        let sec = self.sections.get().and_then(Option::as_ref);
        match (sections, sec) {
            (Some(v), Some(sg)) => {
                let readback = res.host_sec.as_slice().ctx("section read")?;
                if readback[readback.len() - 1] == 0 {
                    v.clear();
                    v.extend_from_slice(readback);
                    if sg.test {
                        out.copy_from_slice(res.host.as_slice().ctx("state read")?);
                    }
                } else {
                    // Too many marks or palettes: the raw states, for the CPU
                    v.clear();
                    res.st
                        .memcpy_dtoh(&res.end, &mut res.host)
                        .ctx("state readback")?;
                    res.st.synchronize().ctx("state readback")?;
                    out.copy_from_slice(res.host.as_slice().ctx("state read")?);
                }
            }
            (Some(v), None) => {
                v.clear();
                out.copy_from_slice(res.host.as_slice().ctx("state read")?);
            }
            (None, _) => out.copy_from_slice(res.host.as_slice().ctx("state read")?),
        }
        Ok(res.host_err.as_slice().ctx("error read")?[0] == 0)
    }
    /// Returns the buffers (not those of a failed call, whose stream may be broken).
    fn put_back(&self, res: SurfaceResources, ok: bool, t: Instant) {
        if ok {
            self.pool.lock().unwrap().push(res);
            self.stats.chunks.fetch_add(1, Relaxed);
            self.stats
                .ns
                .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        }
    }
}

/// An ore blob (OreFeature.doPlace) to compute: its `count` spheres `[x, y, z, r]` (before pruning)
/// from `first` in the batch's spheres, and its box (low corner, side in X and Z, height).
#[derive(Clone, Copy, Debug)]
pub struct OreBlob {
    pub first: u32,
    pub count: u32,
    pub start: [i32; 3],
    pub size_xz: i32,
    pub size_y: i32,
}

#[derive(Default)]
struct OreStats {
    batches: AtomicU64,
    blobs: AtomicU64,
    ns_send: AtomicU64,
    ns_wait: AtomicU64,
    fallbacks: AtomicU64,
    errors: AtomicU64,
    mismatches: AtomicU64,
}

/// The cells of the ore blobs on the GPU (`HORIZON_GPU_ORES`): the sphere pruning of
/// OreFeature.doPlace, then the mask of the cells of each row (x, y) of the blob's box.
/// A chunk's batch is sent at the start of its decoration and read at the ore step.
pub struct OreGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<crate::cuda::driver::CudaModule>,
    k_blobs: CudaFunction,
    pool: Mutex<Vec<OreResources>>,
    stats: OreStats,
}
/// The buffers of a batch, in 32-bit words: as input the blobs (8 words each) then the spheres (8
/// words each), as output the errors (1 word per blob) then the rows.
struct OreResources {
    st: Arc<CudaStream>,
    /// Capacities: blobs, spheres, rows
    cap: [usize; 3],
    input: CudaSlice<u32>,
    output: CudaSlice<u32>,
    h_input: PinnedHostSlice<u32>,
    h_output: PinnedHostSlice<u32>,
}
/// A batch of blobs sent to the GPU, to read after `OreGpu::wait` and to return through
/// `OreGpu::finish`.
pub struct OreJob {
    res: Option<OreResources>,
    /// First row of each blob, and the total at the end
    rows: Vec<usize>,
    expected: bool,
    ok: bool,
    /// The readback, once waited for (address and length): `as_slice` is a driver call
    view: (usize, usize),
}
impl Drop for OreJob {
    fn drop(&mut self) {
        // The stream's copies write into these buffers: they are only released after them
        if let Some(r) = &self.res
            && !self.expected
        {
            let _ = r.st.synchronize();
        }
    }
}
impl std::fmt::Debug for OreGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OreGpu")
    }
}
impl Drop for OreGpu {
    fn drop(&mut self) {
        if let Some(n) = std::num::NonZero::new(self.stats.batches.load(Relaxed)) {
            eprintln!(
                "GPU ores: {n} batches, {} blobs per batch, upload {} us, wait {} us per batch; {} features fell back to the CPU, {} blobs failed, {} mismatches with the CPU (test mode)",
                self.stats.blobs.load(Relaxed) / n,
                self.stats.ns_send.load(Relaxed) / n / 1000,
                self.stats.ns_wait.load(Relaxed) / n / 1000,
                self.stats.fallbacks.load(Relaxed),
                self.stats.errors.load(Relaxed),
                self.stats.mismatches.load(Relaxed),
            );
        }
    }
}
impl OreGpu {
    pub fn new() -> Result<Self, String> {
        let ctx = context()?;
        let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
        let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
        let module = load_module(&ctx, include_str!("ores.cu"), arch).ctx("ore module")?;
        let k_blobs = module.load_function("kore_blobs").ctx("kore_blobs")?;
        Ok(Self {
            ctx,
            _module: module,
            k_blobs,
            pool: Mutex::new(Vec::new()),
            stats: OreStats::default(),
        })
    }
    /// A planned feature fell back to the CPU.
    pub fn note_cpu_fallback(&self) {
        self.stats.fallbacks.fetch_add(1, Relaxed);
    }
    /// Test mode: a blob whose cells differ from the CPU's.
    pub fn note_mismatch(&self) {
        self.stats.mismatches.fetch_add(1, Relaxed);
    }
    fn resource(&self, needed: [usize; 3]) -> Result<OreResources, String> {
        let mut pool = self.pool.lock().unwrap();
        if let Some(k) = pool
            .iter()
            .position(|r| (0..3).all(|i| r.cap[i] >= needed[i]))
        {
            return Ok(pool.swap_remove(k));
        }
        // Too small (or none): a bigger one replaces one of them
        let previous = pool.pop();
        drop(pool);
        let cap: [usize; 3] = std::array::from_fn(|i| {
            let fallback = [256, 8192, 65536][i];
            needed[i]
                .max(fallback)
                .max(previous.as_ref().map_or(0, |r| r.cap[i]))
                .next_power_of_two()
        });
        let st = match previous {
            Some(r) => r.st.clone(),
            None => self.ctx.new_stream().ctx("CUDA stream")?,
        };
        let [b, s, m] = cap;
        Ok(OreResources {
            input: st.alloc_zeros::<u32>(8 * b + 8 * s).ctx("blobs")?,
            output: st.alloc_zeros::<u32>(b + m).ctx("rows")?,
            h_input: unsafe { self.ctx.alloc_pinned_with_flags::<u32>(8 * b + 8 * s, 0) }
                .ctx("host buffer")?,
            h_output: unsafe { self.ctx.alloc_pinned_with_flags::<u32>(b + m, 0) }
                .ctx("host buffer")?,
            st,
            cap,
        })
    }
    /// Sends the blobs of a chunk; the computation and the readback happen on a separate stream.
    pub fn submit(&self, blobs: &[OreBlob], spheres: &[[f64; 4]]) -> Result<OreJob, String> {
        let t = Instant::now();
        let nb = blobs.len();
        let mut rows = Vec::with_capacity(nb + 1);
        let mut total = 0usize;
        for b in blobs {
            if b.size_xz <= 0
                || b.size_xz > 32
                || b.size_y <= 0
                || b.size_y > 255
                || b.first as usize + b.count as usize > spheres.len()
            {
                return Err("blob hors limites".into());
            }
            rows.push(total);
            total += (b.size_xz * b.size_y) as usize;
        }
        rows.push(total);
        if nb == 0 {
            return Ok(OreJob {
                res: None,
                rows,
                expected: true,
                ok: true,
                view: (0, 0),
            });
        }
        let mut res = self.resource([nb, spheres.len(), total])?;
        let (n_input, n_output) = (8 * nb + 8 * spheres.len(), nb + total);
        {
            let h = res.h_input.as_mut_slice().ctx("host buffer")?;
            let (heads, rest) = h.split_at_mut(4 * nb);
            let (starts, sp) = rest.split_at_mut(4 * nb);
            for (i, b) in blobs.iter().enumerate() {
                heads[4 * i..4 * i + 4].copy_from_slice(&[
                    b.first,
                    b.count,
                    rows[i] as u32,
                    (b.size_xz | (b.size_y << 8)) as u32,
                ]);
                starts[4 * i..4 * i + 4].copy_from_slice(&[
                    b.start[0] as u32,
                    b.start[1] as u32,
                    b.start[2] as u32,
                    0,
                ]);
            }
            // The doubles in little-endian, as on the GPU
            for (words, v) in sp
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(spheres.iter().flatten())
            {
                let b = v.to_bits();
                words[0] = b as u32;
                words[1] = (b >> 32) as u32;
            }
        }
        let st = res.st.clone();
        st.memcpy_htod(
            &res.h_input.as_slice().ctx("host buffer")?[..n_input],
            &mut res.input.slice_mut(0..n_input),
        )
        .ctx("blob upload")?;
        let nbi = nb as i32;
        let mut l = st.launch_builder(&self.k_blobs);
        l.arg(&res.input).arg(&mut res.output).arg(&nbi);
        let cfg = LaunchConfig {
            grid_dim: (nb as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { l.launch(cfg) }.ctx("ore blobs")?;
        st.memcpy_dtoh(
            &res.output.slice(0..n_output),
            &mut res.h_output.as_mut_slice().ctx("host buffer")?[..n_output],
        )
        .ctx("row readback")?;
        self.stats.batches.fetch_add(1, Relaxed);
        self.stats.blobs.fetch_add(nb as u64, Relaxed);
        self.stats
            .ns_send
            .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        Ok(OreJob {
            res: Some(res),
            rows,
            expected: false,
            ok: true,
            view: (0, 0),
        })
    }
    /// Waits for the batch (once); `false` if the GPU failed (the batch then falls back to the
    /// CPU).
    pub fn wait(&self, job: &mut OreJob) -> bool {
        if !job.expected {
            let t = Instant::now();
            job.ok = job.res.as_ref().is_some_and(|r| r.st.synchronize().is_ok());
            if job.ok
                && let Some(s) = job.res.as_ref().and_then(|r| r.h_output.as_slice().ok())
            {
                job.view = (s.as_ptr() as usize, s.len());
            } else {
                job.ok = false;
            }
            job.expected = true;
            self.stats
                .ns_wait
                .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        }
        job.ok
    }
    /// The rows of blob `i` of a waited batch (row `x * size_y + y`, bit z), or None if the blob
    /// failed (a cell outside its box) or the batch was not waited for.
    pub fn rows<'a>(&self, job: &'a OreJob, i: usize) -> Option<&'a [u32]> {
        if !job.expected || !job.ok {
            return None;
        }
        job.res.as_ref()?;
        // The batch's buffer, read by `wait`: it lives as long as the batch
        let output = unsafe { std::slice::from_raw_parts(job.view.0 as *const u32, job.view.1) };
        let nb = job.rows.len() - 1;
        if *output.get(i)? != 0 {
            self.stats.errors.fetch_add(1, Relaxed);
            return None;
        }
        output.get(nb + job.rows[i]..nb + *job.rows.get(i + 1)?)
    }
    /// Returns the buffers of a batch, after waiting for it if not done yet.
    pub fn finish(&self, mut job: OreJob) {
        self.wait(&mut job);
        if job.ok
            && let Some(r) = job.res.take()
        {
            self.pool.lock().unwrap().push(r);
        }
    }
}
