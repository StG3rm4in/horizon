//! Translation of Horizon's compiled graph (DensityCompiler) to CUDA: straight-line code, one
//! variable per node, in topological order, for each grid (2D pass of the Y-free values, corner
//! grids of the interpolations, block grid). Every float operation goes through `_rn` intrinsics
//! (no fused multiply-add) to reproduce the CPU bit for bit.
#![allow(dead_code, clippy::all)]
use horizon_core::density::graph::{
    DensityCompiler, FunctionId, GraphBinary, GraphNode, GraphUnary,
};
use horizon_core::density::{
    CubicSpline, MultipointSpline, ShiftNoiseKind, TilingMode, WorldField,
};
use horizon_core::noise::{Noise, PerlinNoise};
#[allow(unused_imports)]
use horizon_core::random::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

type R<T> = Result<T, Box<dyn std::error::Error>>;
const BASE: &str = include_str!("base.cu");

/// Copy of one chunk of a region's block grid to a contiguous buffer, in `fill_noise` order: column
/// (x + z * 16), then increasing Y.
const EXCERPT: &str = r#"
extern "C" __global__ void kchunk(const float *MEM, float *OUT, int cx, int cz) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 256L * CGY) return;
  int lc = (int)(i / CGY), iy = (int)(i % CGY);
  int ix = cx * 16 + (lc & 15), iz = cz * 16 + (lc >> 4);
  OUT[i] = MEM[OUTPUT_0 + ((long)iz * CGX + ix) * CGY + iy];
}
"#;

const OPS: &str = r#"
__device__ __forceinline__ float qnan() { return __int_as_float(0x7fc00000); }
__device__ __forceinline__ float jmin(float a, float b) {
    if (isnan(a) || isnan(b)) return qnan();
    if (a == 0.0f && b == 0.0f) return __int_as_float(__float_as_int(a) | __float_as_int(b));
    return a <= b ? a : b;
}
__device__ __forceinline__ float jmax(float a, float b) {
    if (isnan(a) || isnan(b)) return qnan();
    if (a == 0.0f && b == 0.0f) return __int_as_float(__float_as_int(a) & __float_as_int(b));
    return a >= b ? a : b;
}
__device__ __forceinline__ float clampj(float v, float lo, float hi) { return v < lo ? lo : jmin(v, hi); }
// Java (int)Math.floor(double): saturating, NaN gives 0
__device__ __forceinline__ int floor_sat(double v) {
    if (isnan(v)) return 0;
    double f = floor(v);
    if (f >= 2147483647.0) return 2147483647;
    if (f <= -2147483648.0) return (-2147483647 - 1);
    return (int)f;
}
__device__ __forceinline__ float squeeze(float v) {
    float c = clampj(v, -1.0f, 1.0f);
    return __fsub_rn(__fdiv_rn(c, 2.0f), __fdiv_rn(__fmul_rn(__fmul_rn(c, c), c), 24.0f));
}
__device__ __forceinline__ float leaky(float v, float f) { return v > 0.0f ? v : __fmul_rn(v, f); }
__device__ __forceinline__ float signj(float v) { return (v == 0.0f || isnan(v)) ? v : (v > 0.0f ? 1.0f : -1.0f); }
__device__ __forceinline__ float lerpi(float a, float f, float s) { return __fadd_rn(f, __fmul_rn(a, __fsub_rn(s, f))); }
__device__ __forceinline__ int fdiv(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }

// EndIslandSampler::density: (height(x / 8, z / 8) - 8) / 128, the outer End islands of a simplex
// noise; divisions and remainders truncate toward zero, as in Java.
__device__ __noinline__ float end_islands(const unsigned char *P, const double *O, int bx, int bz) {
  int x = bx / 8, z = bz / 8;
  int cx = x / 2, cz = z / 2, lx = x % 2, lz = z % 2;
  float height = -100.0f;
  for (int ox = -12; ox <= 12; ++ox) {
    for (int oz = -12; oz <= 12; ++oz) {
      long long ix = (long long)(int)((unsigned)cx + (unsigned)ox);
      long long iz = (long long)(int)((unsigned)cz + (unsigned)oz);
      if (ix * ix + iz * iz <= 4096LL || sf_simplex2(P, O, (double)ix, (double)iz) >= -0.9f) continue;
      float scale = __fadd_rn(fmodf(__fadd_rn(__fmul_rn(fabsf((float)ix), 3439.0f), __fmul_rn(fabsf((float)iz), 147.0f)), 13.0f), 9.0f);
      float dx = (float)(lx - ox * 2), dz = (float)(lz - oz * 2);
      double d = __dsqrt_rn((double)__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dz, dz)));
      float candidate = __fsub_rn(100.0f, __fmul_rn(__double2float_rn(d), scale));
      candidate = jmin(jmax(candidate, -100.0f), 80.0f);
      height = jmax(height, candidate);
    }
  }
  return __fdiv_rn(__fsub_rn(height, 8.0f), 128.0f);
}

// DistanceMetric::compute3 on float deltas, in its order of operations
__device__ __forceinline__ float dist_squared(float x, float y, float z) {
  return __fadd_rn(__fadd_rn(__fmul_rn(x, x), __fmul_rn(y, y)), __fmul_rn(z, z));
}
__device__ __forceinline__ float dist_euclid(float x, float y, float z) {
  return __double2float_rn(__dsqrt_rn((double)dist_squared(x, y, z)));
}
__device__ __forceinline__ float dist_manhattan(float x, float y, float z) {
  return __fadd_rn(__fadd_rn(fabsf(x), fabsf(y)), fabsf(z));
}
__device__ __forceinline__ float dist_chebyshev(float x, float y, float z) {
  return jmax(jmax(fabsf(x), fabsf(y)), fabsf(z));
}
"#;

fn f32lit(v: f32) -> String {
    format!("__int_as_float((int)0x{:08x}u)", v.to_bits())
}
fn f64lit(v: f64) -> String {
    format!("__longlong_as_double((long long)0x{:016x}ull)", v.to_bits())
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Grid {
    pub sizes: [i32; 3],
    pub step: [i32; 3],
}
impl Grid {
    pub fn n(&self) -> usize {
        self.sizes.iter().map(|&t| t as usize).product()
    }
    pub fn columns(&self) -> usize {
        (self.sizes[0] * self.sizes[2]) as usize
    }
}

/// Straight-line code of a kernel: one variable per node already computed.
#[derive(Default)]
struct Ssa {
    code: String,
    vars: HashMap<FunctionId, String>,
}

struct Gen<'a> {
    c: &'a DensityCompiler,
    defs: String,
    tables: Vec<([u8; 256], [f64; 3])>,
    index_tables: HashMap<usize, usize>,
    points: HashMap<usize, String>,
    noises: HashMap<FunctionId, String>,
    /// Root spline (graph node): CUDA function and full order of its coordinates. A reused spline
    /// must receive exactly the same coordinate array as when it was generated.
    splines: HashMap<FunctionId, (String, Vec<FunctionId>)>,
    nb_splines: usize,
    grids: Vec<Grid>,
    roots: Vec<FunctionId>,
    interps: HashMap<FunctionId, (usize, [i32; 3])>,
    edges: Vec<Vec<FunctionId>>,
    external: Vec<(FunctionId, usize)>,
    nv: usize,
    count: BTreeMap<&'static str, usize>,
    seed: i64,
    /// Noises kept alive for the whole generation: the tables are identified by their address,
    /// which a freed object could let another noise reuse.
    kept: Vec<Box<dyn std::any::Any>>,
    /// Volume noises of the 3D kernels, with their double part precomputed per axis:
    /// (grid, layers (table, xz scale, y scale, amplitude, fudge)).
    axes: Vec<(usize, Vec<(usize, f64, f64, f32, Option<f64>)>)>,
    index_axes: HashMap<(String, usize), String>,
    /// Point mode: Horizon's `sample_value` (one point at a time) instead of `sample_volume`.
    /// Noises through `get3`, binary operations with their shortcuts, and java_min/java_max.
    point_mode: bool,
    /// `float pfN(int x, int y, int z)` functions already generated in point mode.
    point_functions: HashMap<FunctionId, String>,
    /// In point mode, during the loop of a `find_top_surface` descent: the Y-free subgraphs read by
    /// a node with Y, computed once before the loop (`hK` variables).
    hoisted: Option<Vec<FunctionId>>,
    n_descents: usize,
}

/// A precomputed layer: for each axis, cell (integer bits), position and smoothing.
fn layer_step(t: [i32; 3]) -> usize {
    3 * (t[0] + t[1] + t[2]) as usize
}

impl<'a> Gen<'a> {
    fn new(c: &'a DensityCompiler, main_grid: Grid, root: Option<FunctionId>, seed: i64) -> Self {
        Self {
            axes: Vec::new(),
            index_axes: HashMap::new(),
            kept: Vec::new(),
            seed,
            c,
            defs: String::new(),
            tables: Vec::new(),
            index_tables: HashMap::new(),
            points: HashMap::new(),
            noises: HashMap::new(),
            splines: HashMap::new(),
            nb_splines: 0,
            grids: vec![main_grid],
            roots: root.into_iter().collect(),
            interps: HashMap::new(),
            edges: vec![Vec::new()],
            external: Vec::new(),
            nv: 0,
            count: BTreeMap::new(),
            point_mode: false,
            point_functions: HashMap::new(),
            hoisted: None,
            n_descents: 0,
        }
    }
    /// `__device__ float sdN(int x, int z, float upper)`: FindTopSurfaceSampler's descent from the
    /// upper bound, with the density in point mode. Its Y-free subgraphs are computed once before
    /// the loop: they do not change from one cell of the column to the next.
    fn surface_descent(&mut self, density: FunctionId, lower: i32, height: i32) -> R<String> {
        let previous_mode = std::mem::replace(&mut self.point_mode, true);
        let previous_hoisted = self.hoisted.replace(Vec::new());
        let mut loop_ssa = Ssa::default();
        let v = self.value(&mut loop_ssa, density, 0, true);
        let hoisted = std::mem::replace(&mut self.hoisted, previous_hoisted).unwrap_or_default();
        let r = v.and_then(|v| {
            let mut before = Ssa::default();
            for (k, id) in hoisted.iter().enumerate() {
                let hv = self.value(&mut before, *id, 0, false)?;
                writeln!(before.code, "  float h{k} = {hv};")?;
            }
            Ok((v, before.code))
        });
        self.point_mode = previous_mode;
        let (v, prelude) = r?;
        let name = format!("sd{}", self.n_descents);
        self.n_descents += 1;
        writeln!(
            self.defs,
            "__device__ __noinline__ float {name}(int x, int z, float upper) {{\n  int y = 0;\n{prelude}  int sy = (int)((unsigned)floor_sat((double)__fdiv_rn(upper, {h})) * {height}u);\n  if (sy <= {lower}) return (float)({lower});\n  while (sy >= {lower}) {{\n    int y = sy;\n{loop_ssa}    if ({v} > 0.0f) return (float)sy;\n    sy -= {height};\n  }}\n  return (float)({lower});\n}}",
            h = f32lit(height as f32),
            loop_ssa = loop_ssa.code
        )?;
        Ok(name)
    }
    /// `__device__ void NAME(int x, int z, int qy0, int nqy, float *V, long step)`: the roots in
    /// point mode (like `sample_value`) at each Y quart j of a column, at y = (qy0 + j) << 2, into
    /// V[j * step + k] for root k. As in `surface_descent`, the Y-free subgraphs are computed once
    /// before the loop, and shared between the roots (the same node, only once).
    fn column_points(&mut self, name: &str, roots: &[FunctionId]) -> R<String> {
        let previous_mode = std::mem::replace(&mut self.point_mode, true);
        let previous_hoisted = self.hoisted.replace(Vec::new());
        let mut loop_ssa = Ssa::default();
        let mut vs = Vec::new();
        let mut error = None;
        for &r in roots {
            match self.value(&mut loop_ssa, r, 0, true) {
                Ok(v) => vs.push(v),
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
        let hoisted = std::mem::replace(&mut self.hoisted, previous_hoisted).unwrap_or_default();
        let r = match error {
            Some(e) => Err(e),
            None => (|| -> R<String> {
                let mut before = Ssa::default();
                for (k, id) in hoisted.iter().enumerate() {
                    let hv = self.value(&mut before, *id, 0, false)?;
                    writeln!(before.code, "  float h{k} = {hv};")?;
                }
                Ok(before.code)
            })(),
        };
        self.point_mode = previous_mode;
        let prelude = r?;
        let mut writes = String::new();
        for (k, v) in vs.iter().enumerate() {
            writeln!(writes, "    V[j * step + {k}] = {v};")?;
        }
        Ok(format!(
            "__device__ __noinline__ void {name}(int x, int z, int qy0, int nqy, float *V, long step) {{\n  int y = 0;\n{prelude}  for (int j = 0; j < nqy; ++j) {{\n    int y = (int)((unsigned)(qy0 + j) << 2);\n{loop_ssa}{writes}  }}\n}}",
            loop_ssa = loop_ssa.code
        ))
    }
    /// `__device__ float pfN(int x, int y, int z)`: the value of `id` at one point, exactly like
    /// Horizon's `DensitySampler::sample_value`.
    fn point_function(&mut self, id: FunctionId) -> R<String> {
        if let Some(name) = self.point_functions.get(&id) {
            return Ok(name.clone());
        }
        let previous = std::mem::replace(&mut self.point_mode, true);
        // A separate function does not see the variables hoisted by its calling descent
        let previous_hoisted = self.hoisted.take();
        let mut ssa = Ssa::default();
        let v = self.value(&mut ssa, id, 0, false);
        self.hoisted = previous_hoisted;
        self.point_mode = previous;
        let v = v?;
        let name = format!("pf{}", self.point_functions.len());
        writeln!(
            self.defs,
            "__device__ __noinline__ float {name}(int x, int y, int z) {{\n{}  return {v};\n}}",
            ssa.code
        )?;
        self.point_functions.insert(id, name.clone());
        Ok(name)
    }
    fn node(&self, id: FunctionId) -> R<GraphNode> {
        Ok(self
            .c
            .graph()
            .node(id)
            .ok_or("bound sampler: not supported")?
            .clone())
    }
    fn y_free(&self, id: FunctionId) -> R<bool> {
        Ok(self.c.graph().domain_axes(id)? & 2 == 0)
    }
    /// DensityCompiler::constant after optimization: references are unrolled.
    fn constant(&self, mut id: FunctionId) -> Option<f32> {
        loop {
            match self.c.graph().node(id)? {
                GraphNode::Constant(v) => return Some(*v),
                GraphNode::Reference { input, .. } => id = *input,
                _ => return None,
            }
        }
    }
    /// Whether the subgraph of `id` evaluates the same in point mode as in a volume: only
    /// constants, references, caches, unary operations (binary ones with a constant operand,
    /// minimum and maximum excepted), clamps, gradients, distances and slices.
    fn pointwise(&self, id: FunctionId) -> R<bool> {
        let mut todo = vec![id];
        while let Some(id) = todo.pop() {
            match self.c.graph().node(id).ok_or("missing node")? {
                GraphNode::Constant(_)
                | GraphNode::Gradient { .. }
                | GraphNode::DistanceToPoint { .. } => {}
                GraphNode::Reference { input, .. }
                | GraphNode::Cache(input)
                | GraphNode::Slice { input, .. }
                | GraphNode::Unary { input, .. }
                | GraphNode::Clamp { input, .. } => todo.push(*input),
                // A constant operand makes it a unary operation in both modes (compile_binary)
                GraphNode::Binary { kind, left, right }
                    if !matches!(kind, GraphBinary::Minimum | GraphBinary::Maximum)
                        && (self.constant(*left).is_some() || self.constant(*right).is_some()) =>
                {
                    todo.push(*left);
                    todo.push(*right);
                }
                _ => return Ok(false),
            }
        }
        Ok(true)
    }
    fn var(&mut self) -> String {
        self.nv += 1;
        format!("v{}", self.nv)
    }

    // ---------- noises ----------
    fn table(&mut self, p: &PerlinNoise) -> usize {
        self.table_of(p.table())
    }
    fn table_of(&mut self, t: &horizon_core::noise::GradientTable) -> usize {
        let key = t as *const _ as usize;
        if let Some(&i) = self.index_tables.get(&key) {
            return i;
        }
        self.tables.push((*t.permutations(), *t.offsets()));
        self.index_tables.insert(key, self.tables.len() - 1);
        self.tables.len() - 1
    }
    fn layers(
        &mut self,
        n: &dyn Noise,
        xz: f64,
        y: f64,
        amp: f32,
        out: &mut Vec<(usize, f64, f64, f32, Option<f64>)>,
    ) -> R<()> {
        if let Some(p) = n.as_perlin() {
            out.push((self.table(p), xz, y, amp, None));
        } else if let Some(s) = n.as_smeared() {
            out.push((
                self.table_of(s.table()),
                xz,
                y,
                amp,
                Some(s.fudge_y_scale()),
            ));
        } else if let Some(s) = n.as_stack() {
            for l in s.layers() {
                self.layers(
                    &*l.noise,
                    xz * l.frequency,
                    y * l.frequency,
                    amp * l.amplitude,
                    out,
                )?;
            }
        } else {
            return Err("unsupported noise".into());
        }
        Ok(())
    }
    /// CUDA function `float name(int x, int y, int z)`: volume sum of a noise (add_to_volume path).
    fn fn_volume(
        &mut self,
        id: FunctionId,
        n: &dyn Noise,
        xz: f64,
        y: f64,
        amp: f32,
        axes: [&str; 3],
        times4: bool,
    ) -> R<String> {
        if let Some(name) = self.noises.get(&id) {
            return Ok(name.clone());
        }
        let mut c = Vec::new();
        self.layers(n, xz, y, amp, &mut c)?;
        let name = format!("noise{}", self.noises.len());
        let mut s = format!(
            "__device__ __noinline__ float {name}(int x, int y, int z) {{\n  float acc = 0.0f;\n"
        );
        for (t, sxz, sy, a, fudge) in c {
            let call = match fudge {
                None => format!(
                    "perlin_vol(PERM[{t}], OFF[{t}], {}, {}, {}, {}, {}, {})",
                    axes[0],
                    f64lit(sxz),
                    axes[1],
                    f64lit(sy),
                    axes[2],
                    f64lit(sxz)
                ),
                Some(f) => format!(
                    "perlin_vol_smeared(PERM[{t}], OFF[{t}], {}, {}, {}, {}, {}, {}, {})",
                    axes[0],
                    f64lit(sxz),
                    axes[1],
                    f64lit(sy),
                    axes[2],
                    f64lit(sxz),
                    f64lit(f)
                ),
            };
            writeln!(
                s,
                "  acc = __fadd_rn(acc, __fmul_rn({}, {call}));",
                f32lit(a)
            )?;
        }
        writeln!(
            s,
            "  return {};\n}}",
            if times4 {
                format!("__fmul_rn(acc, {})", f32lit(4.0))
            } else {
                "acc".into()
            }
        )?;
        self.defs.push_str(&s);
        self.noises.insert(id, name.clone());
        Ok(name)
    }
    /// Volume noise of a 3D kernel (x, y, z axes of grid g): the double part comes from the
    /// per-axis tables, computed once per zone by the `kax_N` kernel; only float work is left per
    /// point.
    fn fn_volume_axes(
        &mut self,
        key: String,
        n: &dyn Noise,
        xz: f64,
        y: f64,
        amp: f32,
        g: usize,
    ) -> R<String> {
        if let Some(name) = self.index_axes.get(&(key.clone(), g)) {
            return Ok(name.clone());
        }
        let mut c = Vec::new();
        self.layers(n, xz, y, amp, &mut c)?;
        let r = self.axes.len();
        let t = self.grids[g].sizes;
        let [sx, sy, sz] = t;
        let step = layer_step(t);
        let name = format!("baxes{r}");
        let tables = c
            .iter()
            .map(|l| l.0.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let amps = c.iter().map(|l| f32lit(l.3)).collect::<Vec<_>>().join(", ");
        let nl = c.len();
        writeln!(
            self.defs,
            "__device__ __noinline__ float {name}(const float *MEM, int ix, int iy, int iz) {{\n  const int T[{nl}] = {{{tables}}};\n  const float A[{nl}] = {{{amps}}};\n  float acc = 0.0f;\n  for (int l = 0; l < {nl}; ++l) {{\n    const float *B = MEM + AX_{r} + (long)l * {step};\n    const float *X = B, *Y = B + 3 * {sx}, *Z = B + 3 * {sx} + 3 * {sy};\n    float v = perlin_core(PERM[T[l]], __float_as_int(X[ix]), X[{sx} + ix], X[2 * {sx} + ix], __float_as_int(Y[iy]), Y[{sy} + iy], Y[2 * {sy} + iy], __float_as_int(Z[iz]), Z[{sz} + iz], Z[2 * {sz} + iz]);\n    acc = __fadd_rn(acc, __fmul_rn(A[l], v));\n  }}\n  return acc;\n}}"
        )?;
        self.axes.push((g, c));
        self.index_axes.insert((key, g), name.clone());
        Ok(name)
    }

    /// Kernel that precomputes an axis table: one thread per (layer, axis, index), in double,
    /// exactly like PerlinNoise::add_volume (wrap, shift, floor_i32, position, smoothing, Y fudge).
    fn kernel_axes(&self, r: usize) -> R<String> {
        let (g, c) = &self.axes[r];
        let gr = self.grids[*g];
        let [sx, sy, sz] = gr.sizes;
        let [px, py, pz] = gr.step;
        let nl = c.len();
        let step = layer_step(gr.sizes);
        let list = |f: &dyn Fn(&(usize, f64, f64, f32, Option<f64>)) -> String| {
            c.iter().map(f).collect::<Vec<_>>().join(", ")
        };
        let t = list(&|l| l.0.to_string());
        let sxz = list(&|l| f64lit(l.1));
        let syy = list(&|l| f64lit(l.2));
        let fud = list(&|l| f64lit(l.4.unwrap_or(0.0)));
        let smeared = list(&|l| {
            if l.4.is_some() {
                "1".into()
            } else {
                "0".into()
            }
        });
        Ok(format!(
            "extern \"C\" __global__ void kax_{r}(float *MEM, int MX, int MY, int MZ) {{\n  const int T[{nl}] = {{{t}}};\n  const double SXZ[{nl}] = {{{sxz}}};\n  const double SY[{nl}] = {{{syy}}};\n  const double FUD[{nl}] = {{{fud}}};\n  const int SMEARED[{nl}] = {{{smeared}}};\n  long k = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (k >= (long){nl} * ({sx} + {sy} + {sz})) return;\n  int l = (int)(k / ({sx} + {sy} + {sz})), i = (int)(k % ({sx} + {sy} + {sz}));\n  int a = i < {sx} ? 0 : (i < {sx} + {sy} ? 1 : 2);\n  int j = a == 0 ? i : (a == 1 ? i - {sx} : i - {sx} - {sy});\n  int n = a == 0 ? {sx} : (a == 1 ? {sy} : {sz});\n  int c = a == 0 ? MX + j * {px} : (a == 1 ? MY + j * {py} : MZ + j * {pz});\n  double e = a == 1 ? SY[l] : SXZ[l];\n  double input = __dmul_rn((double)c, e);\n  double h = __dadd_rn(wrapd(input), OFF[T[l]][a]);\n  int cell = floor_i32(h);\n  double dd = __dsub_rn(h, (double)cell);\n  float d = __double2float_rn(dd);\n  if (a == 1 && SMEARED[l]) {{\n    double limited = (input >= 0.0 && input < dd) ? input : dd;\n    const double eps = __longlong_as_double(0x3e7ad7f2a0000000LL);\n    double f = __dmul_rn((double)floor_i32(__dadd_rn(__ddiv_rn(limited, FUD[l]), eps)), FUD[l]);\n    d = __double2float_rn(__dsub_rn(dd, f));\n  }}\n  float *B = MEM + AX_{r} + (long)l * {step} + (a == 0 ? 0 : (a == 1 ? 3 * {sx} : 3 * {sx} + 3 * {sy}));\n  B[j] = __int_as_float(cell);\n  B[n + j] = d;\n  B[2 * n + j] = smooth(__double2float_rn(dd));\n}}"
        ))
    }

    /// NoiseSampler (zeroed buffer then add_to_volume, amplitude 1) of a given stack, x, y, z axes.
    fn fn_volume_key(&mut self, name: &str, n: &dyn Noise, xz: f64, y: f64) -> R<String> {
        let mut c = Vec::new();
        self.layers(n, xz, y, 1.0, &mut c)?;
        let mut s = format!(
            "__device__ __noinline__ float {name}(int x, int y, int z) {{\n  float acc = 0.0f;\n"
        );
        for (t, sxz, sy, a, fudge) in c {
            let call = match fudge {
                None => format!(
                    "perlin_vol(PERM[{t}], OFF[{t}], x, {}, y, {}, z, {})",
                    f64lit(sxz),
                    f64lit(sy),
                    f64lit(sxz)
                ),
                Some(f) => format!(
                    "perlin_vol_smeared(PERM[{t}], OFF[{t}], x, {}, y, {}, z, {}, {})",
                    f64lit(sxz),
                    f64lit(sy),
                    f64lit(sxz),
                    f64lit(f)
                ),
            };
            writeln!(
                s,
                "  acc = __fadd_rn(acc, __fmul_rn({}, {call}));",
                f32lit(a)
            )?;
        }
        writeln!(s, "  return acc;\n}}")?;
        self.defs.push_str(&s);
        Ok(name.to_string())
    }
    fn point(&mut self, n: &dyn Noise) -> R<String> {
        if let Some(p) = n.as_perlin() {
            let t = self.table(p);
            let name = format!("pt_perlin{t}");
            if self
                .points
                .insert(p as *const PerlinNoise as usize, name.clone())
                .is_none()
            {
                writeln!(
                    self.defs,
                    "__device__ float {name}(double x, double y, double z) {{ return perlin_get3(PERM[{t}], OFF[{t}], x, y, z); }}"
                )?;
            }
            return Ok(name);
        }
        let s = n.as_stack().ok_or("unsupported noise (point mode)")?;
        let key = s as *const _ as *const u8 as usize;
        if let Some(name) = self.points.get(&key) {
            return Ok(name.clone());
        }
        let mut body = String::from("  float v = 0.0f;\n");
        for l in s.layers() {
            let f = self.point(&*l.noise)?;
            let fr = f64lit(l.frequency);
            writeln!(
                body,
                "  v = __fadd_rn(v, __fmul_rn({}, {f}(__dmul_rn(x, {fr}), __dmul_rn(y, {fr}), __dmul_rn(z, {fr}))));",
                f32lit(l.amplitude)
            )?;
        }
        let name = format!("pt_stack{}", self.points.len());
        writeln!(
            self.defs,
            "__device__ __noinline__ float {name}(double x, double y, double z) {{\n{body}  return v;\n}}"
        )?;
        self.points.insert(key, name.clone());
        Ok(name)
    }

    // ---------- splines: one CUDA function per multipoint node, coordinates passed as an array
    // ----------
    fn spline_root(
        &mut self,
        id: FunctionId,
        m: &MultipointSpline<FunctionId>,
    ) -> R<(String, Vec<FunctionId>)> {
        if let Some(r) = self.splines.get(&id) {
            return Ok(r.clone());
        }
        let mut coords = Vec::new();
        let mut local_splines = HashMap::new();
        let f = self.spline_fn(m, &mut coords, &mut local_splines)?;
        self.splines.insert(id, (f.clone(), coords.clone()));
        Ok((f, coords))
    }
    /// One function per multipoint node of the same root spline; `local_splines` remembers the
    /// sub-splines of this generation only (their addresses are only stable during it).
    fn spline_fn(
        &mut self,
        m: &MultipointSpline<FunctionId>,
        coords: &mut Vec<FunctionId>,
        local_splines: &mut HashMap<usize, String>,
    ) -> R<String> {
        let key = m as *const _ as usize;
        let ci = match coords.iter().position(|&c| c == m.coordinate()) {
            Some(i) => i,
            None => {
                coords.push(m.coordinate());
                coords.len() - 1
            }
        };
        if let Some(n) = local_splines.get(&key) {
            return Ok(n.clone());
        }
        let pts = m.points();
        let n = pts.len();
        let mut values = Vec::new();
        for p in pts {
            values.push(match &p.value {
                CubicSpline::Constant(v) => f32lit(*v),
                CubicSpline::Multipoint(sub) => {
                    format!("{}(C)", self.spline_fn(sub, coords, local_splines)?)
                }
            });
        }
        let locs = pts
            .iter()
            .map(|p| f32lit(p.location))
            .collect::<Vec<_>>()
            .join(", ");
        let name = format!("spline{}", self.nb_splines);
        self.nb_splines += 1;
        let mut s = format!(
            "__device__ __noinline__ float {name}(const float *C) {{\n  float input = C[{ci}];\n  const float LOC[{n}] = {{{locs}}};\n  int first = 0, remaining = {n};\n  while (remaining != 0) {{ int half = remaining / 2; int middle = first + half; if (input < LOC[middle]) remaining = half; else {{ first = middle + 1; remaining -= half + 1; }} }}\n"
        );
        for (e, cond) in [
            (0usize, "first == 0".to_string()),
            (n - 1, format!("first == {n}")),
        ] {
            let p = &pts[e];
            let body = if p.derivative == 0.0 {
                format!("return {};", values[e])
            } else {
                format!(
                    "float v = {}; return __fadd_rn(v, __fmul_rn({}, __fsub_rn(input, {})));",
                    values[e],
                    f32lit(p.derivative),
                    f32lit(p.location)
                )
            };
            writeln!(s, "  if ({cond}) {{ {body} }}")?;
        }
        writeln!(s, "  switch (first) {{")?;
        for i in 1..n {
            let (l, r) = (&pts[i - 1], &pts[i]);
            writeln!(
                s,
                "  case {i}: {{\n    float width = __fsub_rn({rl}, {ll});\n    float alpha = __fdiv_rn(__fsub_rn(input, {ll}), width);\n    float lv = {lv}; float rv = {rv};\n    float lc = __fsub_rn(__fmul_rn({ld}, width), __fsub_rn(rv, lv));\n    float rc = __fadd_rn(__fmul_rn({nrd}, width), __fsub_rn(rv, lv));\n    float linear = __fadd_rn(lv, __fmul_rn(alpha, __fsub_rn(rv, lv)));\n    float corr = __fadd_rn(lc, __fmul_rn(alpha, __fsub_rn(rc, lc)));\n    return __fadd_rn(linear, __fmul_rn(__fmul_rn(alpha, __fsub_rn(1.0f, alpha)), corr));\n  }}",
                rl = f32lit(r.location),
                ll = f32lit(l.location),
                lv = values[i - 1],
                rv = values[i],
                ld = f32lit(l.derivative),
                nrd = f32lit(-r.derivative)
            )?;
        }
        writeln!(s, "  }}\n  return qnan();\n}}")?;
        self.defs.push_str(&s);
        local_splines.insert(key, name.clone());
        Ok(name)
    }

    /// Expression of the value of `id` in the current kernel (literal, variable or memory read).
    fn value(&mut self, ssa: &mut Ssa, id: FunctionId, g: usize, ctx_3d: bool) -> R<String> {
        if let Some(v) = self.constant(id) {
            return Ok(f32lit(v));
        }
        if ctx_3d && self.point_mode && self.hoisted.is_some() && self.y_free(id)? {
            // Descent loop: computed once before the loop (surface_descent)
            let h = self.hoisted.as_mut().unwrap();
            let slot = match h.iter().position(|&b| b == id) {
                Some(s) => s,
                None => {
                    h.push(id);
                    h.len() - 1
                }
            };
            return Ok(format!("h{slot}"));
        }
        if ctx_3d && !self.point_mode && self.y_free(id)? {
            // Y-free value: computed once per column by the 2D pass (like Horizon's Slice)
            let slot = match self.edges[g].iter().position(|&b| b == id) {
                Some(s) => s,
                None => {
                    self.edges[g].push(id);
                    self.edges[g].len() - 1
                }
            };
            return Ok(format!("MEM[COL_{g}_{slot} + ci]"));
        }
        if let Some(v) = ssa.vars.get(&id) {
            return Ok(v.clone());
        }
        let node = self.node(id)?;
        let d3 = !self.y_free(id)?;
        let kind: &'static str = match &node {
            GraphNode::Reference { .. } => "Reference",
            GraphNode::Cache(_) => "Cache",
            GraphNode::Noise { .. } => "Noise",
            GraphNode::OffsetNoise { .. } => "OffsetNoise",
            GraphNode::Unary { .. } => "Unary",
            GraphNode::Binary { .. } => "Binary",
            GraphNode::Clamp { .. } => "Clamp",
            GraphNode::Choice { .. } => "Choice",
            GraphNode::Lerp { .. } => "Lerp",
            GraphNode::IntervalSelect { .. } => "IntervalSelect",
            GraphNode::Gradient { .. } => "Gradient",
            GraphNode::Spline(_) => "Spline",
            GraphNode::Interpolated { .. } => "Interpolated",
            GraphNode::WorldField(_) => "WorldField",
            GraphNode::BlendDensity(_) => "BlendDensity",
            GraphNode::BlendedNoise(_) => "BlendedNoise (CPU)",
            _ => "?",
        };
        *self.count.entry(kind).or_default() += 1;
        let expr: String = match node {
            GraphNode::Reference { input, .. }
            | GraphNode::Cache(input)
            | GraphNode::BlendDensity(input) => {
                // BlendDensity: without blending with old chunks, the value passes through
                // unchanged
                let v = self.value(ssa, input, g, d3)?;
                ssa.vars.insert(id, v.clone());
                return Ok(v);
            }
            GraphNode::WorldField(f) => f32lit(if matches!(f, WorldField::BlendAlpha) {
                1.0
            } else {
                0.0
            }),
            GraphNode::OffsetNoise { noise, kind } if self.point_mode => {
                // compile_shift_noise, point path: get3 then * 4 in float (ShiftB: get3(z/4, x/4,
                // 0))
                let n = self.c.get_noise_sampler(noise)?;
                self.kept.push(Box::new(n.clone()));
                let pt = self.point(&*n)?;
                let q = f64lit(0.25);
                let call = match kind {
                    ShiftNoiseKind::Shift => format!(
                        "{pt}(__dmul_rn((double)x, {q}), __dmul_rn((double)y, {q}), __dmul_rn((double)z, {q}))"
                    ),
                    ShiftNoiseKind::ShiftA => format!(
                        "{pt}(__dmul_rn((double)x, {q}), __dmul_rn((double)y, 0.0), __dmul_rn((double)z, {q}))"
                    ),
                    ShiftNoiseKind::ShiftB => {
                        format!("{pt}(__dmul_rn((double)z, {q}), __dmul_rn((double)x, {q}), 0.0)")
                    }
                };
                format!("__fmul_rn({call}, {})", f32lit(4.0))
            }
            GraphNode::OffsetNoise { noise, kind } => {
                let n = self.c.get_noise_sampler(noise)?;
                self.kept.push(Box::new(n.clone()));
                let f = match kind {
                    ShiftNoiseKind::Shift => {
                        self.fn_volume(id, &*n, 0.25, 0.25, 1.0, ["x", "y", "z"], true)?
                    }
                    ShiftNoiseKind::ShiftA => {
                        self.fn_volume(id, &*n, 0.25, 0.0, 1.0, ["x", "y", "z"], true)?
                    }
                    ShiftNoiseKind::ShiftB => {
                        self.fn_volume(id, &*n, 0.25, 0.25, 4.0, ["z", "x", "0"], false)?
                    }
                };
                format!("{f}(x, y, z)")
            }
            GraphNode::Noise {
                noise,
                xz_scale,
                y_scale,
                shift_x,
                shift_y,
                shift_z,
            } => {
                let n = self.c.get_noise_sampler(noise)?;
                self.kept.push(Box::new(n.clone()));
                let zero = |s: &Self, f: FunctionId| matches!(s.c.graph().node(f), Some(GraphNode::Constant(v)) if v.to_bits() == 0);
                if zero(self, shift_x) && zero(self, shift_y) && zero(self, shift_z) {
                    if self.point_mode {
                        // NoiseSampler::sample_value: get3(x * xz, y * y_scale, z * xz)
                        let pt = self.point(&*n)?;
                        let (xz, ys) = (f64lit(xz_scale), f64lit(y_scale));
                        format!(
                            "{pt}(__dmul_rn((double)x, {xz}), __dmul_rn((double)y, {ys}), __dmul_rn((double)z, {xz}))"
                        )
                    } else if d3 {
                        format!(
                            "{}(MEM, ix, iy, iz)",
                            self.fn_volume_axes(format!("{id:?}"), &*n, xz_scale, y_scale, 1.0, g)?
                        )
                    } else {
                        format!(
                            "{}(x, y, z)",
                            self.fn_volume(
                                id,
                                &*n,
                                xz_scale,
                                y_scale,
                                1.0,
                                ["x", "y", "z"],
                                false
                            )?
                        )
                    }
                } else {
                    let sx = self.value(ssa, shift_x, g, d3)?;
                    let sz = self.value(ssa, shift_z, g, d3)?;
                    let sy = if zero(self, shift_y) {
                        None
                    } else {
                        Some(self.value(ssa, shift_y, g, d3)?)
                    };
                    let pt = self.point(&*n)?;
                    let (xz, ys) = (f64lit(xz_scale), f64lit(y_scale));
                    let ye = match sy {
                        Some(sy) => {
                            format!("__dadd_rn(__dmul_rn((double)y, {ys}), (double)({sy}))")
                        }
                        None => format!("__dmul_rn((double)y, {ys})"),
                    };
                    format!(
                        "{pt}(__dadd_rn(__dmul_rn((double)x, {xz}), (double)({sx})), {ye}, __dadd_rn(__dmul_rn((double)z, {xz}), (double)({sz})))"
                    )
                }
            }
            GraphNode::Unary { kind, input } => {
                let v = self.value(ssa, input, g, d3)?;
                match kind {
                    GraphUnary::Abs => format!("fabsf({v})"),
                    GraphUnary::Square => format!("__fmul_rn({v}, {v})"),
                    GraphUnary::Cube => format!("__fmul_rn(__fmul_rn({v}, {v}), {v})"),
                    GraphUnary::HalfNegative => format!("leaky({v}, {})", f32lit(0.5)),
                    GraphUnary::QuarterNegative => format!("leaky({v}, {})", f32lit(0.25)),
                    GraphUnary::Reciprocal => format!("__fdiv_rn(1.0f, {v})"),
                    GraphUnary::Negate => format!("(-({v}))"),
                    GraphUnary::Squeeze => format!("squeeze({v})"),
                    GraphUnary::Sign => format!("signj({v})"),
                    GraphUnary::Sqrt => format!("__double2float_rn(__dsqrt_rn((double)({v})))"),
                    GraphUnary::Log => return Err("Unary::Log non pris en charge".into()),
                }
            }
            GraphNode::Clamp { input, min, max } => {
                format!(
                    "clampj({}, {}, {})",
                    self.value(ssa, input, g, d3)?,
                    f32lit(min),
                    f32lit(max)
                )
            }
            GraphNode::Binary { kind, left, right } => {
                self.binary(ssa, kind, left, right, g, d3)?
            }
            GraphNode::Choice {
                input,
                min,
                max,
                inside,
                outside,
            } => {
                let i = self.value(ssa, input, g, d3)?;
                let a = self.value(ssa, inside, g, d3)?;
                let b = self.value(ssa, outside, g, d3)?;
                format!(
                    "(({i}) >= {} && ({i}) < {}) ? ({a}) : ({b})",
                    f32lit(min),
                    f32lit(max)
                )
            }
            GraphNode::Lerp {
                alpha,
                first,
                second,
            } => {
                let a = self.value(ssa, alpha, g, d3)?;
                let f = self.value(ssa, first, g, d3)?;
                let s = self.value(ssa, second, g, d3)?;
                format!(
                    "(({a}) == 0.0f) ? ({f}) : ((({a}) == 1.0f) ? ({s}) : lerpi({a}, {f}, {s}))"
                )
            }
            GraphNode::IntervalSelect {
                input,
                thresholds,
                branches,
            } => {
                let v = self.value(ssa, input, g, d3)?;
                let mut bs = Vec::new();
                for b in &branches {
                    bs.push(self.value(ssa, *b, g, d3)?);
                }
                let mut e = bs.last().ok_or("no branch")?.clone();
                for (t, b) in thresholds.iter().zip(&bs).rev() {
                    e = format!("(({v}) < {}) ? ({b}) : ({e})", f32lit(*t));
                }
                e
            }
            GraphNode::Gradient {
                axis,
                tiling,
                from,
                to,
                low,
                high,
            } => {
                if !matches!(tiling, TilingMode::ClampToEdge) {
                    return Err("Gradient in Repeat mode is not supported".into());
                }
                let factor = (high - low) / to.wrapping_sub(from) as f32;
                let c = ["x", "y", "z"][axis as usize];
                format!(
                    "__fadd_rn({}, __fmul_rn((float)(min(max({c}, {}), {}) - ({from})), {}))",
                    f32lit(low),
                    from.min(to),
                    from.max(to),
                    f32lit(factor)
                )
            }
            GraphNode::Spline(s) => match &s {
                CubicSpline::Constant(v) => f32lit(*v),
                CubicSpline::Multipoint(m) => {
                    let (f, coords) = self.spline_root(id, m)?;
                    let mut vals = Vec::new();
                    for c in coords {
                        vals.push(self.value(ssa, c, g, d3)?);
                    }
                    let nv = self.var();
                    writeln!(
                        ssa.code,
                        "  float {nv}_c[{}] = {{{}}};",
                        vals.len(),
                        vals.join(", ")
                    )?;
                    format!("{f}({nv}_c)")
                }
            },
            GraphNode::Interpolated { .. } | GraphNode::BlendedNoise(_) if self.point_mode => {
                return Err(
                    "interpolation and BlendedNoise are not supported in point mode".into(),
                );
            }
            GraphNode::Slice {
                axis,
                coordinate,
                input,
            } if self.point_mode => {
                // SliceSampler: the input evaluated with one coordinate fixed
                let f = self.point_function(input)?;
                let mut c = ["x".to_string(), "y".to_string(), "z".to_string()];
                c[axis as usize] = coordinate.to_string();
                format!("{f}({}, {}, {})", c[0], c[1], c[2])
            }
            // SliceSampler in a volume: an input that does not depend on the sliced axis has the
            // same value on every plane
            GraphNode::Slice { axis, input, .. }
                if self.c.graph().domain_axes(input)? & (1 << axis as u8) == 0 =>
            {
                self.value(ssa, input, g, d3)?
            }
            // SliceSampler in a volume over an axis its input reads: the input on the plane, in
            // point mode, when nothing in it evaluates differently in a volume (coordinates,
            // constants and unary operations only, like the End's central island)
            GraphNode::Slice {
                axis,
                coordinate,
                input,
            } if self.pointwise(input)? => {
                let f = self.point_function(input)?;
                let mut c = ["x".to_string(), "y".to_string(), "z".to_string()];
                c[axis as usize] = coordinate.to_string();
                format!("{f}({}, {}, {})", c[0], c[1], c[2])
            }
            GraphNode::DistanceToPoint { point, metric } => {
                // DistanceToPointSampler: (point - position) as float, then the metric
                let d = |i: usize, c: &str| {
                    format!("(float)(int)((unsigned)({}) - (unsigned){c})", point[i])
                };
                let (dx, dy, dz) = (d(0, "x"), d(1, "y"), d(2, "z"));
                use horizon_core::density::DistanceMetric as M;
                match metric {
                    M::Euclidean => format!("dist_euclid({dx}, {dy}, {dz})"),
                    M::EuclideanSquared => format!("dist_squared({dx}, {dy}, {dz})"),
                    M::Manhattan => format!("dist_manhattan({dx}, {dy}, {dz})"),
                    M::Chebyshev => format!("dist_chebyshev({dx}, {dy}, {dz})"),
                }
            }
            GraphNode::Surface {
                density,
                upper,
                lower,
                height,
            } if self.point_mode => {
                // compile_surface_search: Slice Y = 0 around FindTopSurfaceSampler. The upper bound
                // is read at y = 0; the descent goes from cell to cell until a positive density.
                if height <= 0 {
                    return Err("find_top_surface: cell height is not positive".into());
                }
                let fd = self.point_function(density)?;
                let fu = self.point_function(upper)?;
                let name = format!("surface{}", self.point_functions.len());
                writeln!(
                    self.defs,
                    "__device__ __noinline__ float {name}(int x, int z) {{\n  float upper = {fu}(x, 0, z);\n  int y = (int)((unsigned)floor_sat((double)__fdiv_rn(upper, {h})) * {height}u);\n  if (y <= {lower}) return (float)({lower});\n  while (y >= {lower}) {{\n    if ({fd}(x, y, z) > 0.0f) return (float)y;\n    y -= {height};\n  }}\n  return (float)({lower});\n}}",
                    h = f32lit(height as f32)
                )?;
                format!("{name}(x, z)")
            }
            GraphNode::Surface {
                density,
                upper,
                lower,
                height,
            } => {
                // FindTopSurfaceSampler::sample_volume under the Slice Y = 0: the upper bound as a
                // volume on the y = 0 plane, then the density point by point, going down from cell
                // to cell.
                if height <= 0 {
                    return Err("find_top_surface: cell height is not positive".into());
                }
                let u = self.value(ssa, upper, g, d3)?;
                let fd = self.surface_descent(density, lower, height)?;
                let r = self.var();
                writeln!(ssa.code, "  float {r} = {fd}(x, z, {u});")?;
                r
            }
            GraphNode::Interpolated { input, xz, y } => {
                self.interpolated(ssa, id, input, [xz, y, xz], g)?
            }
            GraphNode::EndIslands => {
                // EndIslandSampler: the outer islands of a simplex noise drawn from the seed
                let sampler = Box::new(self.c.end_island_sampler()?);
                let t = self.table_of(sampler.noise().table());
                self.kept.push(sampler);
                format!("end_islands(PERM[{t}], OFF[{t}], x, z)")
            }
            GraphNode::BlendedNoise(def) => {
                // BlendedNoise::compile: Lerp(clamp(main + 0.5, 0, 1), min_limit, max_limit), three
                // stacks of smeared Perlin noises, drawn like DensityCompiler (hash of
                // "minecraft:terrain")
                let bbox = Box::new(self.c.blended_fbm_set(&def)?);
                let ptr: *const horizon_core::density::BlendedFbmSet = &*bbox;
                self.kept.push(bbox); // the content of a Box does not move while it lives
                let set: &horizon_core::density::BlendedFbmSet = unsafe { &*ptr };
                let (xz, ys) = (684.412 * def.xz_scale, 684.412 * def.y_scale);
                let main = self.fn_volume_axes(
                    format!("{id:?}main"),
                    &set.main,
                    xz / def.xz_factor,
                    ys / def.y_factor,
                    1.0,
                    g,
                )?;
                let minl =
                    self.fn_volume_axes(format!("{id:?}min"), &set.min_limit, xz, ys, 1.0, g)?;
                let maxl =
                    self.fn_volume_axes(format!("{id:?}max"), &set.max_limit, xz, ys, 1.0, g)?;
                let name = format!("mixed_axes{}", self.axes.len());
                writeln!(
                    self.defs,
                    "__device__ __noinline__ float {name}(const float *MEM, int ix, int iy, int iz) {{\n  float a = clampj(__fadd_rn({main}(MEM, ix, iy, iz), {}), 0.0f, 1.0f);\n  if (a == 0.0f) return {minl}(MEM, ix, iy, iz);\n  if (a == 1.0f) return {maxl}(MEM, ix, iy, iz);\n  return lerpi(a, {minl}(MEM, ix, iy, iz), {maxl}(MEM, ix, iy, iz));\n}}",
                    f32lit(0.5)
                )?;
                let v = self.var();
                writeln!(ssa.code, "  float {v} = {name}(MEM, ix, iy, iz);")?;
                ssa.vars.insert(id, v.clone());
                return Ok(v);
            }
            other => {
                let d = format!("{other:?}");
                return Err(format!("unsupported node: {}", &d[..d.len().min(80)]).into());
            }
        };
        let v = self.var();
        writeln!(ssa.code, "  float {v} = {expr};")?;
        ssa.vars.insert(id, v.clone());
        Ok(v)
    }

    /// DensityCompiler::compile_binary: min/max pruning by intervals, constants as unary
    /// operations.
    fn binary(
        &mut self,
        ssa: &mut Ssa,
        kind: GraphBinary,
        left: FunctionId,
        right: FunctionId,
        g: usize,
        d3: bool,
    ) -> R<String> {
        if matches!(kind, GraphBinary::Minimum | GraphBinary::Maximum) {
            let a = self.c.graph().range(left)?;
            let b = self.c.graph().range(right)?;
            let min = kind == GraphBinary::Minimum;
            if (min && a.max() < b.min()) || (!min && a.min() > b.max()) {
                return self.value(ssa, left, g, d3);
            }
            if (min && b.max() < a.min()) || (!min && b.min() > a.max()) {
                return self.value(ssa, right, g, d3);
            }
        }
        // Point mode: UnaryOp::Minimum/Maximum outside bulk go through java_min/java_max
        let point = self.point_mode;
        if let Some(c) = self.constant(left) {
            let v = self.value(ssa, right, g, d3)?;
            let c = f32lit(c);
            return Ok(match kind {
                GraphBinary::Add => format!("__fadd_rn({v}, {c})"),
                GraphBinary::Subtract => format!("__fsub_rn({c}, {v})"),
                GraphBinary::Multiply => format!("__fmul_rn({v}, {c})"),
                GraphBinary::Divide => format!("__fdiv_rn({c}, {v})"),
                GraphBinary::Minimum if point => format!("jmin({v}, {c})"),
                GraphBinary::Maximum if point => format!("jmax({v}, {c})"),
                GraphBinary::Minimum => format!("(({c}) < ({v}) ? ({c}) : ({v}))"),
                GraphBinary::Maximum => format!("(({c}) > ({v}) ? ({c}) : ({v}))"),
            });
        }
        if let Some(c) = self.constant(right) {
            let v = self.value(ssa, left, g, d3)?;
            return Ok(match kind {
                GraphBinary::Add => format!("__fadd_rn({v}, {})", f32lit(c)),
                GraphBinary::Subtract => format!("__fadd_rn({v}, {})", f32lit(-c)),
                GraphBinary::Multiply => format!("__fmul_rn({v}, {})", f32lit(c)),
                GraphBinary::Divide => format!("__fmul_rn({v}, {})", f32lit(1.0 / c)),
                GraphBinary::Minimum if point => format!("jmin({v}, {})", f32lit(c)),
                GraphBinary::Maximum if point => format!("jmax({v}, {})", f32lit(c)),
                GraphBinary::Minimum => format!("(({0}) < ({v}) ? ({0}) : ({v}))", f32lit(c)),
                GraphBinary::Maximum => format!("(({0}) > ({v}) ? ({0}) : ({v}))", f32lit(c)),
            });
        }
        let l = self.value(ssa, left, g, d3)?;
        let r = self.value(ssa, right, g, d3)?;
        if point {
            // BinarySampler::sample_value: shortcuts on the left value, then java_min/java_max
            return Ok(match kind {
                GraphBinary::Add => format!("__fadd_rn({l}, {r})"),
                GraphBinary::Subtract => format!("__fsub_rn({l}, {r})"),
                GraphBinary::Multiply => format!("(({l}) == 0.0f) ? 0.0f : __fmul_rn({l}, {r})"),
                GraphBinary::Divide => format!("(({l}) == 0.0f) ? 0.0f : __fdiv_rn({l}, {r})"),
                GraphBinary::Minimum => {
                    let rmin = f32lit(self.c.graph().range(right)?.min());
                    format!("(({l}) <= {rmin}) ? ({l}) : jmin({l}, {r})")
                }
                GraphBinary::Maximum => {
                    let rmax = f32lit(self.c.graph().range(right)?.max());
                    format!("(({l}) >= {rmax}) ? ({l}) : jmax({l}, {r})")
                }
            });
        }
        Ok(match kind {
            GraphBinary::Add => format!("__fadd_rn({l}, {r})"),
            GraphBinary::Subtract => format!("__fsub_rn({l}, {r})"),
            GraphBinary::Multiply => format!("__fmul_rn({l}, {r})"),
            GraphBinary::Divide => format!("__fdiv_rn({l}, {r})"),
            GraphBinary::Minimum => format!("(({r}) < ({l}) ? ({r}) : ({l}))"),
            GraphBinary::Maximum => format!("(({r}) > ({l}) ? ({r}) : ({l}))"),
        })
    }

    /// InterpolatedSampler (aligned block volume): cell corners on a coarse grid, lerp Z then X at
    /// each level, then successive increments in Y from the bottom of the cell.
    fn interpolated(
        &mut self,
        ssa: &mut Ssa,
        id: FunctionId,
        input: FunctionId,
        cell: [i32; 3],
        g: usize,
    ) -> R<String> {
        if g != 0 {
            return Err("nested interpolation is not supported".into());
        }
        let v = self.grids[0];
        if v.step != [1, 1, 1] || (0..3).any(|i| v.sizes[i] % cell[i] != 0) {
            return Err("the block grid must be aligned on the cells".into());
        }
        if !self.interps.contains_key(&id) {
            // Aligned volume: (size / cell) + 1 corners per axis
            // (InterpolatedSampler::sample_block_volume)
            let sizes = std::array::from_fn(|i| v.sizes[i] / cell[i] + 1);
            self.grids.push(Grid { sizes, step: cell });
            self.edges.push(Vec::new());
            self.roots.push(input);
            self.interps.insert(id, (self.grids.len() - 1, cell));
        }
        let (gl, c) = self.interps[&id];
        let [lsx, lsy, _] = self.grids[gl].sizes;
        let inv = c.map(|c| 1.0 / c as f32);
        let r = self.var();
        writeln!(
            ssa.code,
            "  float {r};\n  {{\n    int cx = fdiv(x, {cx}) - fdiv(MX, {cx}), cy = fdiv(y, {cy}) - fdiv(MY, {cy}), cz = fdiv(z, {cz}) - fdiv(MZ, {cz});\n    int ox = x - fdiv(x, {cx}) * {cx}, oy = y - fdiv(y, {cy}) * {cy}, oz = z - fdiv(z, {cz}) * {cz};\n    float ax = __fmul_rn((float)ox, {ix}), az = __fmul_rn((float)oz, {iz});\n    const float *S = MEM + OUTPUT_{gl};\n    #define C(a, b, c) S[((long)(c) * {lsx} + (a)) * {lsy} + (b)]\n    float bottom = lerpi(ax, lerpi(az, C(cx, cy, cz), C(cx, cy, cz + 1)), lerpi(az, C(cx + 1, cy, cz), C(cx + 1, cy, cz + 1)));\n    float top = lerpi(ax, lerpi(az, C(cx, cy + 1, cz), C(cx, cy + 1, cz + 1)), lerpi(az, C(cx + 1, cy + 1, cz), C(cx + 1, cy + 1, cz + 1)));\n    #undef C\n    float inc = __fmul_rn(__fsub_rn(top, bottom), {iy});\n    float val = __fadd_rn(bottom, __fmul_rn(inc, 0.0f));\n    for (int k = 0; k < oy; ++k) val = __fadd_rn(val, inc);\n    {r} = val;\n  }}",
            cx = c[0],
            cy = c[1],
            cz = c[2],
            ix = f32lit(inv[0]),
            iy = f32lit(inv[1]),
            iz = f32lit(inv[2]),
        )?;
        Ok(r)
    }

    /// The kernels of all the grids (the interpolation grids show up along the way).
    fn kernels(&mut self) -> R<String> {
        let mut src = String::new();
        let mut g = 0;
        let mut body = Vec::new();
        // The 2D passes in the loop: an interpolation read by a Y-free node (preliminary surface)
        // adds its corner grid during the 2D pass.
        while g < self.grids.len() {
            let mut ssa = Ssa::default();
            let r = self.value(&mut ssa, self.roots[g], g, true)?;
            // 2D pass: one value per column for each Y-free node read by a 3D node
            // One thread per (column, value): more parallelism than per column alone
            let mut case = String::new();
            for (s, b) in self.edges[g].clone().into_iter().enumerate() {
                let mut ssa = Ssa::default();
                let v = self.value(&mut ssa, b, g, false)?;
                writeln!(
                    case,
                    "  case {s}: {{\n{}  MEM[COL_{g}_{s} + ci] = {v};\n  break; }}",
                    ssa.code
                )?;
            }
            body.push((ssa.code, r, case));
            g += 1;
        }
        for (g, (code, r, case)) in body.into_iter().enumerate() {
            let gr = self.grids[g];
            let [sx, sy, sz] = gr.sizes;
            let [px, py, pz] = gr.step;
            let ns = self.edges[g].len().max(1);
            // y = MY: an interpolation read in the 2D pass falls on the first layer of corners
            writeln!(
                src,
                "extern \"C\" __global__ void k2d_{g}(float *MEM, int MX, int MY, int MZ) {{\n  long k = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (k >= (long){sx} * {sz} * {ns}) return;\n  int s = (int)(k / ((long){sx} * {sz}));\n  long ci = k % ((long){sx} * {sz});\n  int x = MX + (int)(ci % {sx}) * {px}, z = MZ + (int)(ci / {sx}) * {pz}, y = MY; long gi = 0;\n  switch (s) {{\n{case}  }}\n}}"
            )?;
            writeln!(
                src,
                "extern \"C\" __global__ void k3d_{g}(float *MEM, int MX, int MY, int MZ) {{\n  long gi = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (gi >= (long){sx} * {sy} * {sz}) return;\n  long ci = gi / {sy};\n  int ix = (int)(ci % {sx}), iy = (int)(gi % {sy}), iz = (int)(ci / {sx});\n  int x = MX + ix * {px}, y = MY + iy * {py}, z = MZ + iz * {pz};\n{code}  MEM[OUTPUT_{g} + gi] = {r};\n}}"
            )?;
        }
        Ok(src)
    }
}

/// The CUDA of a region and the layout of its memory (a single `MEM` buffer).
pub(crate) struct Program {
    pub src: String,
    /// Number of floats in `MEM`.
    pub size: usize,
    pub grids: Vec<Grid>,
    /// Number of Y-free values per column, for each grid.
    pub edges: Vec<usize>,
    /// Axis table kernels: (index, grid, number of threads).
    pub axes: Vec<(usize, usize, usize)>,
    pub lines: usize,
    /// Whether the root adds the beardifier last ([`beard_added_last`]): the GPU evaluates the
    /// beardifier as 0, then adds the structures' beard to the result.
    pub beard_last: bool,
}

/// Whether `root` is `add(x, beardifier)`, in either order and through references and caches,
/// with no beardifier in `x`: vanilla's final_density. The GPU evaluates the beardifier as 0 and
/// adds the structures' beard to the result, which is exact only for that shape; a datapack that
/// puts the beardifier elsewhere keeps its chunks with structures on the CPU.
pub(crate) fn beard_added_last(c: &DensityCompiler, root: FunctionId) -> bool {
    let g = c.graph();
    let peel = |mut id: FunctionId| loop {
        match g.node(id) {
            Some(GraphNode::Reference { input, .. } | GraphNode::Cache(input)) => id = *input,
            _ => return id,
        }
    };
    let is_beard = |id| {
        matches!(
            g.node(peel(id)),
            Some(GraphNode::WorldField(WorldField::Beardifier))
        )
    };
    // A bound sampler (no definition) may read anything: counted as a beardifier
    let reads_beard = |id: FunctionId| {
        let (mut todo, mut seen) = (vec![id], HashSet::new());
        while let Some(id) = todo.pop() {
            if !seen.insert(id) {
                continue;
            }
            match g.node(id) {
                None | Some(GraphNode::WorldField(WorldField::Beardifier)) => return true,
                Some(node) => todo.extend(node.children()),
            }
        }
        false
    };
    match g.node(peel(root)) {
        Some(GraphNode::Binary {
            kind: GraphBinary::Add,
            left,
            right,
        }) => {
            (is_beard(*right) && !reads_beard(*left)) || (is_beard(*left) && !reads_beard(*right))
        }
        _ => false,
    }
}

/// Generates the CUDA of `root` for a block grid of `sizes`.
pub(crate) fn generate(
    c: &DensityCompiler,
    root: FunctionId,
    seed: i64,
    sizes: [i32; 3],
) -> R<Program> {
    let mut g = Gen::new(
        c,
        Grid {
            sizes,
            step: [1, 1, 1],
        },
        Some(root),
        seed,
    );
    let kernels = g.kernels()?;
    if !g.external.is_empty() {
        return Err("unexpected CPU buffer".into());
    }
    let mut defs = String::new();
    let mut size = 0usize;
    for (k, gr) in g.grids.iter().enumerate() {
        writeln!(defs, "#define OUTPUT_{k} {size}L")?;
        size += gr.n();
    }
    for (k, b) in g.edges.iter().enumerate() {
        for s in 0..b.len() {
            writeln!(defs, "#define COL_{k}_{s} {size}L")?;
            size += g.grids[k].columns();
        }
    }
    let mut kernels_axes = String::new();
    let mut axes = Vec::new();
    for r in 0..g.axes.len() {
        let (gr, layers) = &g.axes[r];
        writeln!(defs, "#define AX_{r} {size}L")?;
        let n = layers.len() * layer_step(g.grids[*gr].sizes);
        size += n;
        kernels_axes.push_str(&g.kernel_axes(r)?);
        kernels_axes.push('\n');
        axes.push((r, *gr, n / 3));
    }
    writeln!(
        defs,
        "#define CGX {}\n#define CGY {}\n#define CGZ {}",
        sizes[0], sizes[1], sizes[2]
    )?;
    let mut src = format!(
        "{defs}__device__ const unsigned char PERM[{}][256] = {{\n",
        g.tables.len().max(1)
    );
    for (p, _) in &g.tables {
        writeln!(
            src,
            "  {{{}}},",
            p.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )?;
    }
    writeln!(
        src,
        "}};\n__device__ const double OFF[{}][3] = {{",
        g.tables.len().max(1)
    )?;
    for (_, o) in &g.tables {
        writeln!(src, "  {{{:e}, {:e}, {:e}}},", o[0], o[1], o[2])?;
    }
    writeln!(
        src,
        "}};\n{BASE}\n{OPS}\n{}\n{kernels}\n{kernels_axes}\n{EXCERPT}",
        g.defs
    )?;
    let lines = src.lines().count();
    Ok(Program {
        src,
        size,
        grids: g.grids.clone(),
        edges: g.edges.iter().map(|b| b.len()).collect(),
        axes,
        lines,
        beard_last: beard_added_last(c, root),
    })
}

/// Permutation tables and offsets of the Perlin noises `g` uses.
fn tables_cuda(g: &Gen) -> R<String> {
    let mut src = format!(
        "__device__ const unsigned char PERM[{}][256] = {{\n",
        g.tables.len().max(1)
    );
    for (p, _) in &g.tables {
        writeln!(
            src,
            "  {{{}}},",
            p.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )?;
    }
    writeln!(
        src,
        "}};\n__device__ const double OFF[{}][3] = {{",
        g.tables.len().max(1)
    )?;
    for (_, o) in &g.tables {
        writeln!(src, "  {{{:e}, {:e}, {:e}}},", o[0], o[1], o[2])?;
    }
    src.push_str("};\n");
    Ok(src)
}

/// CUDA that evaluates each root at a list of points, like Horizon's `sample_value`:
/// `kpt_K(const int *P, float *OUT, int n)` for root K, points (x, y, z) one after the other in P.
pub(crate) fn generate_points(c: &DensityCompiler, roots: &[FunctionId], seed: i64) -> R<String> {
    let first_root = *roots.first().ok_or("no root")?;
    let mut g = Gen::new(
        c,
        Grid {
            sizes: [1, 1, 1],
            step: [1, 1, 1],
        },
        Some(first_root),
        seed,
    );
    let mut kernels = String::new();
    for (k, &r) in roots.iter().enumerate() {
        let f = g.point_function(r)?;
        writeln!(
            kernels,
            "extern \"C\" __global__ void kpt_{k}(const int *P, float *OUT, int n) {{\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i >= n) return;\n  OUT[i] = {f}(P[3 * i], P[3 * i + 1], P[3 * i + 2]);\n}}"
        )?;
    }
    Ok(format!(
        "{}{BASE}\n{OPS}\n{}\n{kernels}",
        tables_cuda(&g)?,
        g.defs
    ))
}

const RANDOM: &str = include_str!("random.cu");
const AQUIFER: &str = include_str!("aquifer.cu");

/// CUDA of the aquifers: the six roots (order of AquiferField::ALL: barrier, flood, spread, lava,
/// exclusion, surface level) in point mode, then the kernels of `aquifer.cu`.
/// Without roots (no aquifers), the six fields are a stub the kernels never call.
pub(crate) fn generate_aquifer(
    c: &DensityCompiler,
    roots: Option<[FunctionId; 6]>,
    seed: i64,
) -> R<String> {
    let mut g = Gen::new(
        c,
        Grid {
            sizes: [1, 1, 1],
            step: [1, 1, 1],
        },
        None,
        seed,
    );
    let mut defs = String::new();
    let names = [
        "AQ_BARRIER",
        "AQ_FLOOD",
        "AQ_SPREAD",
        "AQ_LAVA",
        "AQ_EXCL",
        "AQ_SURFACE",
    ];
    match roots {
        Some(roots) => {
            for (name, r) in names.iter().zip(roots) {
                let f = g.point_function(r)?;
                writeln!(defs, "#define {name} {f}")?;
            }
        }
        None => {
            writeln!(
                defs,
                "__device__ __forceinline__ float aq_none(int, int, int) {{ return 0.0f; }}"
            )?;
            for name in names {
                writeln!(defs, "#define {name} aq_none")?;
            }
        }
    }
    Ok(format!(
        "{}{BASE}\n{OPS}\n{}\n{defs}\n{RANDOM}\n{AQUIFER}",
        tables_cuda(&g)?,
        g.defs
    ))
}

const BIOMES: &str = include_str!("biomes.cu");

/// CUDA of the biomes: the six climate roots (temperature, humidity, continentalness, erosion,
/// depth, weirdness) per column of quarts (`cl_column`, see `Gen::column_points`), like
/// `ClimateSampler::sample`, then the kernels of `biomes.cu`.
pub(crate) fn generate_biomes(c: &DensityCompiler, roots: [FunctionId; 6], seed: i64) -> R<String> {
    let mut g = Gen::new(
        c,
        Grid {
            sizes: [1, 1, 1],
            step: [1, 1, 1],
        },
        None,
        seed,
    );
    let column = g.column_points("cl_column", &roots)?;
    Ok(format!(
        "{}{BASE}\n{OPS}\n{}\n{column}\n{BIOMES}",
        tables_cuda(&g)?,
        g.defs
    ))
}

/// CUDA that evaluates noises (Horizon's `Noise::get3`) at points given in double:
/// `knz_K(const double *P, float *OUT, int n)` for noise K.
pub(crate) fn generate_noises(c: &DensityCompiler, noises: &[&dyn Noise]) -> R<String> {
    let mut g = Gen::new(
        c,
        Grid {
            sizes: [1, 1, 1],
            step: [1, 1, 1],
        },
        None,
        0,
    );
    let mut kernels = String::new();
    for (k, n) in noises.iter().enumerate() {
        let f = g.point(*n)?;
        writeln!(
            kernels,
            "extern \"C\" __global__ void knz_{k}(const double *P, float *OUT, int n) {{\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i >= n) return;\n  OUT[i] = {f}(P[3 * i], P[3 * i + 1], P[3 * i + 2]);\n}}"
        )?;
    }
    Ok(format!(
        "{}{BASE}\n{OPS}\n{}\n{kernels}\n{RANDOM}\n{DRAWS}",
        tables_cuda(&g)?,
        g.defs
    ))
}

/// Positional draws: first next_float, first next_double and three next_float in a row.
const DRAWS: &str = r#"
extern "C" __global__ void kdraws(const int *P, float *F, double *D, float *F3, int n, unsigned long long LO, unsigned long long HI) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long lo, hi;
  aq_at(LO, HI, P[3 * i], P[3 * i + 1], P[3 * i + 2], lo, hi);
  F[i] = aq_next_float(lo, hi);
  aq_at(LO, HI, P[3 * i], P[3 * i + 1], P[3 * i + 2], lo, hi);
  D[i] = aq_next_double(lo, hi);
  aq_at(LO, HI, P[3 * i], P[3 * i + 1], P[3 * i + 2], lo, hi);
  for (int k = 0; k < 3; ++k) F3[3 * i + k] = aq_next_float(lo, hi);
}
"#;

const SURFACE: &str = include_str!("surface.cu");

/// CUDA of the surface's S2 milestone: biome zoom and temperature (simplex noises).
pub(crate) fn generate_surface_s2(
    c: &DensityCompiler,
    height: &horizon_core::noise::SimplexNoise,
    freeze: &horizon_core::noise::NoiseStack,
    info: &horizon_core::noise::SimplexNoise,
) -> R<String> {
    let mut g = Gen::new(
        c,
        Grid {
            sizes: [1, 1, 1],
            step: [1, 1, 1],
        },
        None,
        0,
    );
    let mut defs = String::new();
    writeln!(defs, "#define T_HEIGHT {}", g.table_of(height.table()))?;
    writeln!(defs, "#define T_INFO {}", g.table_of(info.table()))?;
    let layers = freeze.layers();
    if layers.len() != 3 {
        return Err("freeze noise: 3 layers expected".into());
    }
    for (k, l) in layers.iter().enumerate() {
        let sx = l
            .noise
            .as_simplex()
            .ok_or("freeze noise: layer is not simplex")?;
        writeln!(defs, "#define T_FREEZE{k} {}", g.table_of(sx.table()))?;
        writeln!(defs, "#define T_FREEZE_FREQ_{k} {}", f64lit(l.frequency))?;
        writeln!(defs, "#define T_FREEZE_AMP_{k} {}", f32lit(l.amplitude))?;
    }
    Ok(format!(
        "{}{BASE}\n{OPS}\n{defs}\n{SURFACE}",
        tables_cuda(&g)?
    ))
}

/// A world's surface program: the CUDA of the `ksurface` kernel and the vein roots computed as
/// volumes (one chunk grid per root, in `volumes` order).
/// Side, in chunks, of the preliminary surface regions.
pub(crate) const LEVEL_R: i32 = 8;
pub(crate) struct SurfaceProgram {
    pub src: String,
    pub volumes: Vec<Program>,
    /// The preliminary surface on the y = 0 plane of a region of LEVEL_R x LEVEL_R chunks.
    pub level: Program,
    /// The vein grids of a region: starting y and height.
    pub veins_y: (i32, i32),
}

/// The surface rule translated to CUDA (functions R_k and C_k) and the descent of the columns.
pub(crate) fn generate_surface(
    c: &DensityCompiler,
    s: &crate::SurfaceSetup,
    seed: i64,
) -> R<SurfaceProgram> {
    use crate::{SurfaceCondition as SC, SurfaceRule as SR};
    let mut g = Gen::new(
        c,
        Grid {
            sizes: [1, 1, 1],
            step: [1, 1, 1],
        },
        None,
        seed,
    );
    let mut defs = String::new();
    // Noises of the column, of the conditions and of the bands
    let pt_surf = g.point(s.surface_noise)?;
    let pt_sec = g.point(s.secondary_noise)?;
    let pt_clay = g.point(s.clay_noise)?;
    let mut pts = Vec::new();
    for n in &s.noises {
        pts.push(g.point(*n)?);
    }
    // Temperature
    let (height, freeze, info) = s.temperature_noises;
    writeln!(defs, "#define T_HEIGHT {}", g.table_of(height.table()))?;
    writeln!(defs, "#define T_INFO {}", g.table_of(info.table()))?;
    let layers = freeze.layers();
    if layers.len() != 3 {
        return Err("freeze noise: 3 layers expected".into());
    }
    for (k, l) in layers.iter().enumerate() {
        let sx = l
            .noise
            .as_simplex()
            .ok_or("freeze noise: layer is not simplex")?;
        writeln!(defs, "#define T_FREEZE{k} {}", g.table_of(sx.table()))?;
        writeln!(defs, "#define T_FREEZE_FREQ_{k} {}", f64lit(l.frequency))?;
        writeln!(defs, "#define T_FREEZE_AMP_{k} {}", f32lit(l.amplitude))?;
    }
    writeln!(
        defs,
        "#define SF_MINY ({})\n#define SF_H {}\n#define SF_BMINY ({})\n#define SF_BH {}\n#define SF_ZOOM ({}ll)\n#define SF_SEA ({})",
        s.min_y, s.height, s.biome_min_y, s.biome_height, s.zoom_seed, s.sea_level
    )?;
    let [e_default, e_air, e_fluid, e_lava] = s.fill_states;
    writeln!(
        defs,
        "#define SF_E_DEFAULT {e_default}\n#define SF_E_AIR {e_air}\n#define SF_E_FLUID {e_fluid}\n#define SF_E_LAVA {e_lava}"
    )?;
    // Fluid states written by the rule: the CPU surface marks them for post-processing
    let fluids = s
        .fluid_states
        .iter()
        .map(|e| format!("e == {e}"))
        .collect::<Vec<_>>();
    writeln!(
        defs,
        "__device__ __forceinline__ bool sf_fluid(int e) {{ return {}; }}",
        if fluids.is_empty() {
            "false".to_string()
        } else {
            fluids.join(" || ")
        }
    )?;
    // Extensions (S4 milestone): biomes, blocks and noises of the pillars and icebergs
    if let Some(e) = &s.extensions {
        if e.frozen_oceans.len() > 2 {
            return Err("more than two frozen oceans".into());
        }
        let mut noises = Vec::new();
        for n in e.noises {
            noises.push(g.point(n)?);
        }
        let id = |b: Option<&u32>| b.map_or(-1, |&b| i64::from(b));
        writeln!(
            defs,
            "#define SF_EXT 1\n#define SF_ERODE ({})\n#define SF_FROZEN0 ({})\n#define SF_FROZEN1 ({})\n#define SF_DEFAULT {}\n#define SF_ICE {}\n#define SF_SNOW {}",
            id(e.eroded_badlands.as_ref()),
            id(e.frozen_oceans.first()),
            id(e.frozen_oceans.get(1)),
            e.default_block,
            e.packed_ice,
            e.snow
        )?;
        for (name, f) in ["BD_S", "BD_P", "BD_R", "IC_S", "IC_P", "IC_R"]
            .iter()
            .zip(&noises)
        {
            writeln!(defs, "#define {name} {f}")?;
        }
    }
    // Veins: density and richness as volumes (one grid per root and per region of LEVEL_R x LEVEL_R
    // chunks, over the y where a density can be positive), gap in point mode
    let mut volumes: Vec<FunctionId> = Vec::new();
    let mut mismatches = HashMap::new();
    let (world_top, mut low_v, mut high_v) = (s.min_y + s.height - 1, i32::MAX, i32::MIN);
    for r in &s.rules {
        if let SR::OreVein {
            density,
            richness,
            gap,
            positive_y: [low, high],
            ..
        } = r
        {
            for id in [*density, *richness] {
                if !volumes.contains(&id) {
                    volumes.push(id);
                }
            }
            if !mismatches.contains_key(gap) {
                let f = g.point_function(*gap)?;
                mismatches.insert(*gap, f);
            }
            let (low, high) = ((*low).max(s.min_y), (*high).min(world_top));
            if low <= high {
                low_v = low_v.min(low);
                high_v = high_v.max(high);
            }
        }
    }
    // Aligned on 16 (a multiple of the interpolation cells): the values do not depend on the extent
    let (vlo, vh) = if low_v <= high_v {
        let vlo = low_v.div_euclid(16) * 16;
        (vlo, (high_v + 1 - vlo + 15) / 16 * 16)
    } else {
        (0, 16)
    };
    let vr = 16 * LEVEL_R;
    writeln!(
        defs,
        "#define SF_VLO ({vlo})\n#define SF_VH {vh}\n#define SF_VR {vr}\n#define SF_VN ({}L)",
        i64::from(vr) * i64::from(vr) * i64::from(vh)
    )?;
    let mut tables = String::new();
    writeln!(
        tables,
        "__device__ const int BANDS[192] = {{{}}};",
        s.bands
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )?;
    let nt = s.temperatures.len().max(1);
    writeln!(tables, "#define NTB {}", s.temperatures.len())?;
    // Bits of the base temperatures (a __device__ initializer must be constant)
    writeln!(
        tables,
        "__device__ const unsigned TB[{nt}] = {{{}}};",
        s.temperatures
            .iter()
            .map(|t| format!("{:#010x}u", t.0.to_bits()))
            .collect::<Vec<_>>()
            .join(",")
    )?;
    writeln!(
        tables,
        "__device__ const int TG[{nt}] = {{{}}};",
        s.temperatures
            .iter()
            .map(|t| i32::from(t.1).to_string())
            .collect::<Vec<_>>()
            .join(",")
    )?;
    // Declarations, conditions then rules
    let mut code = String::new();
    for k in 0..s.conditions.len() {
        writeln!(code, "__device__ bool C_{k}(Ctx &c);")?;
    }
    for k in 0..s.rules.len() {
        writeln!(code, "__device__ int R_{k}(Ctx &c);")?;
    }
    for (k, cond) in s.conditions.iter().enumerate() {
        let body = match cond {
            SC::Hole => "return c.depth <= 0;".to_string(),
            SC::Steep => "return c.gx <= -4 || c.gz >= 4;".to_string(),
            SC::Biomes(ids) => {
                let words = ids.iter().map(|&i| i as usize / 64 + 1).max().unwrap_or(1);
                let mut m = vec![0u64; words];
                for &i in ids {
                    m[i as usize / 64] |= 1 << (i % 64);
                }
                writeln!(tables, "__device__ const unsigned long long BIO_{k}[{words}] = {{{}}};", m.iter().map(|v| format!("{v}ull")).collect::<Vec<_>>().join(","))?;
                format!("int b = sf_biome(c);\n  if (b < 0) {{ c.error = 1; return false; }}\n  return (b >> 6) < {words} && ((BIO_{k}[b >> 6] >> (b & 63)) & 1ull);")
            }
            SC::Temperature => "int b = sf_biome(c);\n  if (b < 0 || b >= NTB) { c.error = 1; return false; }\n  return !(sf_temperature(__int_as_float((int)TB[b]), TG[b], c.x, c.y, c.z, SF_SEA) >= 0.15f);".to_string(),
            SC::AbovePreliminary => "return c.y >= c.minimum;".to_string(),
            SC::Noise { noise, min, max, is_3d } => format!(
                "double v = (double){}((double)c.x, {}, (double)c.z);\n  return v >= {} && v <= {};",
                pts[*noise],
                if *is_3d { "(double)c.y" } else { "0.0" },
                f64lit(*min),
                f64lit(*max)
            ),
            SC::YAbove { anchor, multiplier, add_stone } => format!(
                "return (int)((unsigned)c.y + (unsigned)({})) >= (int)((unsigned)({anchor}) + (unsigned)c.depth * (unsigned)({multiplier}));",
                if *add_stone { "c.above" } else { "0" }
            ),
            SC::Water { offset, multiplier, add_stone } => format!(
                "return c.water == (-2147483647 - 1) || (int)((unsigned)c.y + (unsigned)({})) >= (int)((unsigned)c.water + (unsigned)({offset}) + (unsigned)c.depth * (unsigned)({multiplier}));",
                if *add_stone { "c.above" } else { "0" }
            ),
            SC::StoneDepth { offset, add_surface, secondary_range, ceiling } => format!(
                "int stone = {};\n  int surface = {};\n  int sec = {};\n  return stone <= (int)(1u + (unsigned)({offset}) + (unsigned)surface + (unsigned)sec);",
                if *ceiling { "c.below" } else { "c.above" },
                if *add_surface { "c.depth" } else { "0" },
                if *secondary_range == 0 {
                    "0".to_string()
                } else {
                    format!("(int)(0.0 + ((c.secondary - -1.0) / (1.0 - -1.0)) * ((double)({secondary_range}) - 0.0))")
                }
            ),
            SC::Gradient { low, high, positional } => format!(
                "if (c.y <= {low}) return true;\n  if (c.y >= {high}) return false;\n  double p = 1.0 + (((double)c.y - (double)({low})) / ((double)({high}) - (double)({low}))) * (0.0 - 1.0);\n  {}",
                match positional {
                    crate::Positional::Xoroshiro(lo, hi) => format!(
                        "unsigned long long lo, hi;\n  aq_at({lo}ull, {hi}ull, c.x, c.y, c.z, lo, hi);\n  return (double)aq_next_float(lo, hi) < p;"
                    ),
                    crate::Positional::Legacy(seed) => format!(
                        "long long st = lg_at((long long){}ull, c.x, c.y, c.z);\n  return (double)lg_next_float(st) < p;",
                        *seed as u64
                    ),
                }
            ),
            SC::Not(i) => format!("return !C_{i}(c);"),
        };
        writeln!(code, "__device__ bool C_{k}(Ctx &c) {{\n  {body}\n}}")?;
    }
    let (ore_lo, ore_hi) = s.ore_positional;
    for (k, r) in s.rules.iter().enumerate() {
        let body = match r {
            SR::Empty => "return -1;".to_string(),
            SR::Block(b) => format!("return {b};"),
            SR::Sequence(ch) => {
                let mut t = String::from("int r;");
                for i in ch {
                    write!(t, "\n  r = R_{i}(c);\n  if (r >= 0) return r;")?;
                }
                t.push_str("\n  return -1;");
                t
            }
            SR::Condition { condition, rule } => {
                format!("if (!C_{condition}(c)) return -1;\n  return R_{rule}(c);")
            }
            SR::Bandlands => format!(
                "float n = {pt_clay}((double)c.x, 0.0, (double)c.z);\n  int o = floor_sat((double)__fmul_rn(n, 4.0f) + 0.5);\n  int i = (int)((unsigned)c.y + (unsigned)o + 192u) % 192;\n  if (i < 0) {{ c.error = 1; return -1; }}\n  return BANDS[i];"
            ),
            SR::OreVein {
                blocks,
                raw_chance,
                density,
                richness,
                gap,
                positive_y: [low, high],
            } => {
                let vd = volumes.iter().position(|v| v == density).unwrap();
                let vr = volumes.iter().position(|v| v == richness).unwrap();
                // Outside positive_y the density is never positive: nothing, without a draw (like
                // the CPU)
                let (low, high) = ((*low).max(s.min_y), (*high).min(world_top));
                format!(
                    "if (c.y < {low} || c.y > {high}) return -1;\n  long iv = ((long)c.vz * SF_VR + c.vx) * SF_VH + (c.y - SF_VLO);\n  float d = c.VOL[{vd}L * SF_VN + iv];\n  if (d <= 0.0f) return -1;\n  unsigned long long lo, hi;\n  aq_at({ore_lo}ull, {ore_hi}ull, c.x, c.y, c.z, lo, hi);\n  if (aq_next_float(lo, hi) > d) return -1;\n  float ri = c.VOL[{vr}L * SF_VN + iv];\n  if (aq_next_float(lo, hi) < ri && {}(c.x, c.y, c.z) < 0.0f) return aq_next_float(lo, hi) < {} ? {} : {};\n  return {};",
                    mismatches[gap],
                    f32lit(*raw_chance),
                    blocks[1],
                    blocks[0],
                    blocks[2]
                )
            }
        };
        writeln!(
            code,
            "__device__ __noinline__ int R_{k}(Ctx &c) {{\n  {body}\n}}"
        )?;
    }
    // The root factory: Xoroshiro, or legacy (the Nether, the End) without ore veins nor
    // extensions, the only other users of its draws.
    let (rlo, rhi, depth_draw, depth_double) = match s.root_positional {
        crate::Positional::Xoroshiro(lo, hi) => (
            lo,
            hi,
            format!("unsigned long long lo, hi;\n  aq_at({lo}ull, {hi}ull, x, 0, z, lo, hi);"),
            "aq_next_double(lo, hi)",
        ),
        crate::Positional::Legacy(seed) => {
            if s.extensions.is_some()
                || s.rules
                    .iter()
                    .any(|r| matches!(r, crate::SurfaceRule::OreVein { .. }))
            {
                return Err("legacy random source with ore veins or extensions".into());
            }
            (
                0,
                0,
                format!(
                    "long long st = lg_at((long long){}ull, x, 0, z);",
                    seed as u64
                ),
                "lg_next_double(st)",
            )
        }
    };
    let root = s.root;
    write!(
        code,
        r#"
// The surface in three kernels. COLI, 10 integers per column: otop, ptop (first free y after the
// pillar), ftop (after the iceberg), depth, minimum, plo and phi (the pillar), gx, gz. META, three planes
// of blocks: above (0: the rule does not apply), below, water. ICE: the iceberg blocks (state + 1,
// 0: nothing).
#define SF_ALL 0xffffffffu
__device__ __forceinline__ unsigned sf_until(int l) {{ return l >= 31 ? SF_ALL : ((2u << l) - 1u); }}
__device__ __forceinline__ int sf_kind_p(const signed char *col, int y, int plo, int phi) {{
  return (y >= plo && y <= phi) ? SF_SOLID : sf_kind(col, y);
}}

// 1. One warp per column: first free y (warp vote), then lane 0 computes the column's scalars and its
// extensions (eroded badlands pillar, iceberg).
extern "C" __global__ void ksf_heads(const signed char *CODES, const unsigned short *B, const float *PL, unsigned *ICE, int *COLI, double *COLD, int *ERR, int cx, int cz) {{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int lc = t >> 5, l = t & 31;
  if (lc >= 256) return;
  const signed char *col = CODES + (long)lc * SF_H;
  int otop = SF_MINY;
  for (int base = SF_H - 1; base >= 0; base -= 32) {{
    int iy = base - l;
    unsigned m = __ballot_sync(SF_ALL, iy >= 0 && (col[iy] & 3) != 1);
    if (m) {{ otop = SF_MINY + (base - (__ffs(m) - 1)) + 1; break; }}
  }}
  if (l != 0) return;
  int lx = lc & 15, lz = lc >> 4;
  int x = cx * 16 + lx, z = cz * 16 + lz;
  int error = 0;
  {depth_draw}
  int depth = (int)(((double){pt_surf}((double)x, 0.0, (double)z) * 2.75 + 3.0) + {depth_double} * 0.25);
  double secondary = (double){pt_sec}((double)x, 0.0, (double)z);
  int minimum = (int)((unsigned)floor_sat((double)floorf(PL[lc])) + (unsigned)depth - 8u);
  // The pillar before the descent of its column, the iceberg after. They only depend on the codes (the
  // iceberg only writes into air and water, never over a block of the rule) and raise the first free y.
  int plo = 1, phi = 0, ptop = otop, ftop = otop;
#ifdef SF_EXT
  int bcol;
  {{
    int q[3];
    sf_quarter(SF_ZOOM, x, otop, z, q);
    bcol = sf_biome_quarter(B, cx, cz, SF_BMINY >> 2, SF_BH >> 2, q);
  }}
  if (bcol < 0) error = 1;
  if (bcol >= 0 && bcol == SF_ERODE) {{
    int high = sf_pillar(col, x, z, otop);
    if (high != SF_NOTHING) {{
      plo = otop;
      phi = min(high, SF_MINY + SF_H - 1);
      ptop = phi + 1;
      ftop = ptop;
    }}
  }}
  if (bcol >= 0 && (bcol == SF_FROZEN0 || bcol == SF_FROZEN1)) {{
    int plus = sf_iceberg(col, ICE + (long)lc * SF_H, x, z, otop, minimum, bcol, {rlo}ull, {rhi}ull, &error);
    if (plus != SF_NOTHING && plus + 1 > ftop) ftop = plus + 1;
  }}
#endif
  int *ci = COLI + 10 * lc;
  ci[0] = otop; ci[1] = ptop; ci[2] = ftop; ci[3] = depth; ci[4] = minimum; ci[5] = plo; ci[6] = phi;
  COLD[lc] = secondary;
  if (error) atomicOr(ERR, 1);
}}

// 2. One warp per column: the slopes (columns processed by x then z: x - 1 and z - 1 already have their
// extensions, x + 1 and z + 1 not yet, the column has its pillar but not its iceberg), then the descent
// of build_columns in packs of 32 blocks: below going up, above and water going down.
extern "C" __global__ void ksf_descent(const signed char *CODES, int *COLI, int *META) {{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int lc = t >> 5, l = t & 31;
  if (lc >= 256) return;
  const signed char *col = CODES + (long)lc * SF_H;
  int *ci = COLI + 10 * lc;
  int ptop = ci[1], plo = ci[5], phi = ci[6];
  if (l == 0) {{
    int lx = lc & 15, lz = lc >> 4;
    ci[7] = (lx < 15 ? COLI[10 * (lc + 1)] : ptop) - (lx > 0 ? COLI[10 * (lc - 1) + 2] : ptop);
    ci[8] = (lz < 15 ? COLI[10 * (lc + 16)] : ptop) - (lz > 0 ? COLI[10 * (lc - 16) + 2] : ptop);
  }}
  long n = 256L * SF_H, b0 = (long)lc * SF_H;
  int *A = META, *BW = META + n, *W = META + 2 * n;
  // below = y - (the highest non-solid below y); SF_MINY - 1 counts as air
  int dessous = SF_MINY - 1;
  for (int base = SF_MINY; base <= ptop; base += 32) {{
    int y = base + l;
    int k = sf_kind_p(col, y, plo, phi);
    unsigned ns = __ballot_sync(SF_ALL, k != SF_SOLID);
    unsigned before = l == 0 ? 0u : (ns & ((1u << l) - 1u));
    int prec = before ? base + (31 - __clz(before)) : dessous;
    if (k == SF_SOLID && y <= SF_MINY + SF_H - 1) BW[b0 + (y - SF_MINY)] = y - prec;
    if (ns) dessous = base + (31 - __clz(ns));
  }}
  // above = solids since the last air above; water = first fluid since that air, + 1. The descent
  // starts at ptop with above = 0 and water = MIN, as after an air block.
  int cC = 0, cA = 0, cW = -2147483647 - 1;
  for (int base = ptop; base >= SF_MINY; base -= 32) {{
    int y = base - l;
    int k = y < SF_MINY ? SF_AIR : sf_kind_p(col, y, plo, phi);
    unsigned ms = __ballot_sync(SF_ALL, k == SF_SOLID), ma = __ballot_sync(SF_ALL, k == SF_AIR), mf = __ballot_sync(SF_ALL, k == SF_FLUID);
    unsigned le = sf_until(l);
    int cs = cC + __popc(ms & le);
    unsigned a = ma & le;
    int la = a ? 31 - __clz(a) : -1;
    int ca = la >= 0 ? cC + __popc(ms & sf_until(la)) : cA;
    int w;
    if (la >= 0) {{
      unsigned f = mf & le & ~sf_until(la);
      w = f ? (base - (__ffs(f) - 1)) + 1 : (-2147483647 - 1);
    }} else if (cW != -2147483647 - 1) {{
      w = cW;
    }} else {{
      unsigned f = mf & le;
      w = f ? (base - (__ffs(f) - 1)) + 1 : (-2147483647 - 1);
    }}
    if (k == SF_SOLID && y <= SF_MINY + SF_H - 1) {{
      long i = b0 + (y - SF_MINY);
      A[i] = cs - ca;
      W[i] = w;
    }}
    cC = __shfl_sync(SF_ALL, cs, 31);
    cA = __shfl_sync(SF_ALL, ca, 31);
    cW = __shfl_sync(SF_ALL, w, 31);
  }}
}}

// 3. One thread per block: the rule where it applies, then the final state. Bit 30: the surface writes a
// fluid state; bit 31: fluid update scheduled by the fill (fill_noise_codes: code + 4 on a fluid
// state).
extern "C" __global__ void ksf_blocks(const signed char *CODES, const unsigned short *B, const float *VOL, const unsigned *ICE, const int *META, const int *COLI, const double *COLD, unsigned *END, int *ERR, int cx, int cz, int ox, int oz) {{
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long n = 256L * SF_H;
  if (i >= n) return;
  int lc = (int)(i / SF_H), iy = (int)(i % SF_H), y = SF_MINY + iy;
  const int *ci = COLI + 10 * lc;
  unsigned g = ICE[i];
  int o = g ? (int)(g - 1u) : ((y >= ci[5] && y <= ci[6]) ? SF_E_DEFAULT : -1);
  int above = META[i];
  if (above > 0) {{
    Ctx c;
    c.x = cx * 16 + (lc & 15); c.z = cz * 16 + (lc >> 4); c.y = y;
    c.depth = ci[3]; c.minimum = ci[4]; c.gx = ci[7]; c.gz = ci[8]; c.secondary = COLD[lc];
    c.above = above; c.below = META[n + i]; c.water = META[2 * n + i];
    c.biome = -2; c.block = (int)i; c.B = B; c.cx = cx; c.cz = cz; c.VOL = VOL; c.error = 0;
    c.vx = ox + (lc & 15); c.vz = oz + (lc >> 4);
    int r = R_{root}(c);
    if (r >= 0) o = sf_fluid(r) ? (r | 0x40000000) : r;
    if (c.error) atomicOr(ERR, 1);
  }}
  int k = CODES[i];
  int e = o >= 0 ? o : ((k & 3) == 0 ? SF_E_DEFAULT : ((k & 3) == 1 ? SF_E_AIR : ((k & 3) == 2 ? SF_E_FLUID : SF_E_LAVA)));
  END[i] = (unsigned)e | (((k & 4) != 0 && (k & 3) >= 2) ? 0x80000000u : 0u);
}}
"#
    )?;
    let src = format!(
        "{}{BASE}\n{OPS}\n{}\n{defs}\n{tables}\n{RANDOM}\n{SURFACE}\n{code}",
        tables_cuda(&g)?,
        g.defs
    );
    let mut progs = Vec::new();
    for id in volumes {
        progs.push(generate(c, id, seed, [vr, vh, vr])?);
    }
    // Preliminary surface: as a volume on the y = 0 plane (MaterialDensitySources), for a whole
    // region of LEVEL_R x LEVEL_R chunks: same corners (multiples of 16) and same interpolation as
    // a chunk's plane
    let level = generate(c, s.surface_level, seed, [vr, 1, vr])?;
    Ok(SurfaceProgram {
        src,
        volumes: progs,
        level,
        veins_y: (vlo, vh),
    })
}
