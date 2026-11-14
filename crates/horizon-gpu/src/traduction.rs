//! Traduction du graphe compile de Horizon (DensityCompiler) en CUDA : code lineaire, une variable par
//! noeud, en ordre topologique, pour chaque grille (passe 2D des valeurs sans Y, grilles de coins des
//! interpolations, grille de blocs). Toutes les operations float passent par des intrinsics `_rn`
//! (pas de fusion multiplication-addition) pour reproduire le CPU au bit pres.
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
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

type R<T> = Result<T, Box<dyn std::error::Error>>;
const BASE: &str = include_str!("base.cu");

/// Copie d'un chunk de la grille de blocs d'une region vers un tampon contigu, dans l'ordre de
/// `fill_noise` : colonne (x + z * 16), puis Y croissant.
const EXTRAIT: &str = r#"
extern "C" __global__ void kchunk(const float *MEM, float *OUT, int cx, int cz) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 256L * CGY) return;
  int lc = (int)(i / CGY), iy = (int)(i % CGY);
  int ix = cx * 16 + (lc & 15), iz = cz * 16 + (lc >> 4);
  OUT[i] = MEM[SORTIE_0 + ((long)iz * CGX + ix) * CGY + iy];
}
"#;

const OPS: &str = r#"
__device__ __forceinline__ float qnan() { return __int_as_float(0x7fc00000); }
__device__ __forceinline__ float jmin(float a, float b) {
    if (isnan(a) || isnan(b)) return qnan();
    if (a == 0.0f && b == 0.0f) return __int_as_float(__float_as_int(a) | __float_as_int(b));
    return a <= b ? a : b;
}
__device__ __forceinline__ float clampj(float v, float lo, float hi) { return v < lo ? lo : jmin(v, hi); }
__device__ __forceinline__ float squeeze(float v) {
    float c = clampj(v, -1.0f, 1.0f);
    return __fsub_rn(__fdiv_rn(c, 2.0f), __fdiv_rn(__fmul_rn(__fmul_rn(c, c), c), 24.0f));
}
__device__ __forceinline__ float leaky(float v, float f) { return v > 0.0f ? v : __fmul_rn(v, f); }
__device__ __forceinline__ float signj(float v) { return (v == 0.0f || isnan(v)) ? v : (v > 0.0f ? 1.0f : -1.0f); }
__device__ __forceinline__ float lerpi(float a, float f, float s) { return __fadd_rn(f, __fmul_rn(a, __fsub_rn(s, f))); }
__device__ __forceinline__ int fdiv(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
"#;

fn f32lit(v: f32) -> String {
    format!("__int_as_float((int)0x{:08x}u)", v.to_bits())
}
fn f64lit(v: f64) -> String {
    format!("__longlong_as_double((long long)0x{:016x}ull)", v.to_bits())
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Grille {
    pub tailles: [i32; 3],
    pub pas: [i32; 3],
}
impl Grille {
    pub fn n(&self) -> usize {
        self.tailles.iter().map(|&t| t as usize).product()
    }
    pub fn colonnes(&self) -> usize {
        (self.tailles[0] * self.tailles[2]) as usize
    }
}

/// Code lineaire d'un noyau : une variable par noeud deja calcule.
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
    bruits: HashMap<FunctionId, String>,
    splines: HashMap<usize, String>,
    grilles: Vec<Grille>,
    racines: Vec<FunctionId>,
    interps: HashMap<FunctionId, (usize, [i32; 3])>,
    bords: Vec<Vec<FunctionId>>,
    externes: Vec<(FunctionId, usize)>,
    nv: usize,
    compte: BTreeMap<&'static str, usize>,
    graine: i64,
    /// Bruits gardes en vie pendant toute la generation : les tables sont identifiees par leur adresse,
    /// qu'un objet libere pourrait laisser reutiliser par un autre bruit.
    gardes: Vec<Box<dyn std::any::Any>>,
    /// Bruits volumiques des noyaux 3D, avec leur partie double precalculee par axe :
    /// (grille, couches (table, echelle xz, echelle y, amplitude, fudge)).
    axes: Vec<(usize, Vec<(usize, f64, f64, f32, Option<f64>)>)>,
    index_axes: HashMap<(String, usize), String>,
}

/// Une couche precalculee : pour chaque axe, cellule (bits d'entier), position et lissage.
fn pas_couche(t: [i32; 3]) -> usize {
    3 * (t[0] + t[1] + t[2]) as usize
}

impl<'a> Gen<'a> {
    fn new(c: &'a DensityCompiler, principale: Grille, racine: FunctionId, graine: i64) -> Self {
        Self {
            axes: Vec::new(),
            index_axes: HashMap::new(),
            gardes: Vec::new(),
            graine,
            c,
            defs: String::new(),
            tables: Vec::new(),
            index_tables: HashMap::new(),
            points: HashMap::new(),
            bruits: HashMap::new(),
            splines: HashMap::new(),
            grilles: vec![principale],
            racines: vec![racine],
            interps: HashMap::new(),
            bords: vec![Vec::new()],
            externes: Vec::new(),
            nv: 0,
            compte: BTreeMap::new(),
        }
    }
    fn node(&self, id: FunctionId) -> R<GraphNode> {
        Ok(self
            .c
            .graph()
            .node(id)
            .ok_or("sampler lie : non pris en charge")?
            .clone())
    }
    fn sans_y(&self, id: FunctionId) -> R<bool> {
        Ok(self.c.graph().domain_axes(id)? & 2 == 0)
    }
    /// DensityCompiler::constant apres optimisation : les references sont deroulees.
    fn constante(&self, mut id: FunctionId) -> Option<f32> {
        loop {
            match self.c.graph().node(id)? {
                GraphNode::Constant(v) => return Some(*v),
                GraphNode::Reference { input, .. } => id = *input,
                _ => return None,
            }
        }
    }
    fn var(&mut self) -> String {
        self.nv += 1;
        format!("v{}", self.nv)
    }

    // ---------- bruits ----------
    fn table(&mut self, p: &PerlinNoise) -> usize {
        self.table_de(p.table())
    }
    fn table_de(&mut self, t: &horizon_core::noise::GradientTable) -> usize {
        let cle = t as *const _ as usize;
        if let Some(&i) = self.index_tables.get(&cle) {
            return i;
        }
        self.tables.push((*t.permutations(), *t.offsets()));
        self.index_tables.insert(cle, self.tables.len() - 1);
        self.tables.len() - 1
    }
    fn couches(
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
                self.table_de(s.table()),
                xz,
                y,
                amp,
                Some(s.fudge_y_scale()),
            ));
        } else if let Some(s) = n.as_stack() {
            for l in s.layers() {
                self.couches(
                    &*l.noise,
                    xz * l.frequency,
                    y * l.frequency,
                    amp * l.amplitude,
                    out,
                )?;
            }
        } else {
            return Err("bruit non pris en charge".into());
        }
        Ok(())
    }
    /// Fonction CUDA `float nom(int x, int y, int z)` : somme volumique d'un bruit (chemin add_to_volume).
    fn fn_volume(
        &mut self,
        id: FunctionId,
        n: &dyn Noise,
        xz: f64,
        y: f64,
        amp: f32,
        axes: [&str; 3],
        fois4: bool,
    ) -> R<String> {
        if let Some(nom) = self.bruits.get(&id) {
            return Ok(nom.clone());
        }
        let mut c = Vec::new();
        self.couches(n, xz, y, amp, &mut c)?;
        let nom = format!("bruit{}", self.bruits.len());
        let mut s = format!(
            "__device__ __noinline__ float {nom}(int x, int y, int z) {{\n  float acc = 0.0f;\n"
        );
        for (t, sxz, sy, a, fudge) in c {
            let appel = match fudge {
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
                "  acc = __fadd_rn(acc, __fmul_rn({}, {appel}));",
                f32lit(a)
            )?;
        }
        writeln!(
            s,
            "  return {};\n}}",
            if fois4 {
                format!("__fmul_rn(acc, {})", f32lit(4.0))
            } else {
                "acc".into()
            }
        )?;
        self.defs.push_str(&s);
        self.bruits.insert(id, nom.clone());
        Ok(nom)
    }
    /// Bruit volumique d'un noyau 3D (axes x, y, z de la grille g) : la partie double vient des tables
    /// par axe, calculees une fois par zone par le noyau `kax_N` ; il ne reste que du float par point.
    fn fn_volume_axes(
        &mut self,
        cle: String,
        n: &dyn Noise,
        xz: f64,
        y: f64,
        amp: f32,
        g: usize,
    ) -> R<String> {
        if let Some(nom) = self.index_axes.get(&(cle.clone(), g)) {
            return Ok(nom.clone());
        }
        let mut c = Vec::new();
        self.couches(n, xz, y, amp, &mut c)?;
        let r = self.axes.len();
        let t = self.grilles[g].tailles;
        let [sx, sy, sz] = t;
        let pas = pas_couche(t);
        let nom = format!("baxes{r}");
        let tables = c
            .iter()
            .map(|l| l.0.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let amps = c.iter().map(|l| f32lit(l.3)).collect::<Vec<_>>().join(", ");
        let nl = c.len();
        writeln!(
            self.defs,
            "__device__ __noinline__ float {nom}(const float *MEM, int ix, int iy, int iz) {{\n  const int T[{nl}] = {{{tables}}};\n  const float A[{nl}] = {{{amps}}};\n  float acc = 0.0f;\n  for (int l = 0; l < {nl}; ++l) {{\n    const float *B = MEM + AX_{r} + (long)l * {pas};\n    const float *X = B, *Y = B + 3 * {sx}, *Z = B + 3 * {sx} + 3 * {sy};\n    float v = perlin_core(PERM[T[l]], __float_as_int(X[ix]), X[{sx} + ix], X[2 * {sx} + ix], __float_as_int(Y[iy]), Y[{sy} + iy], Y[2 * {sy} + iy], __float_as_int(Z[iz]), Z[{sz} + iz], Z[2 * {sz} + iz]);\n    acc = __fadd_rn(acc, __fmul_rn(A[l], v));\n  }}\n  return acc;\n}}"
        )?;
        self.axes.push((g, c));
        self.index_axes.insert((cle, g), nom.clone());
        Ok(nom)
    }

    /// Noyau de precalcul d'une table d'axes : un thread par (couche, axe, indice), en double, exactement
    /// comme PerlinNoise::add_volume (wrap, decalage, floor_i32, position, lissage, fudge en Y).
    fn noyau_axes(&self, r: usize) -> R<String> {
        let (g, c) = &self.axes[r];
        let gr = self.grilles[*g];
        let [sx, sy, sz] = gr.tailles;
        let [px, py, pz] = gr.pas;
        let nl = c.len();
        let pas = pas_couche(gr.tailles);
        let liste = |f: &dyn Fn(&(usize, f64, f64, f32, Option<f64>)) -> String| {
            c.iter().map(f).collect::<Vec<_>>().join(", ")
        };
        let t = liste(&|l| l.0.to_string());
        let sxz = liste(&|l| f64lit(l.1));
        let syy = liste(&|l| f64lit(l.2));
        let fud = liste(&|l| f64lit(l.4.unwrap_or(0.0)));
        let etale = liste(&|l| {
            if l.4.is_some() {
                "1".into()
            } else {
                "0".into()
            }
        });
        Ok(format!(
            "extern \"C\" __global__ void kax_{r}(float *MEM, int MX, int MY, int MZ) {{\n  const int T[{nl}] = {{{t}}};\n  const double SXZ[{nl}] = {{{sxz}}};\n  const double SY[{nl}] = {{{syy}}};\n  const double FUD[{nl}] = {{{fud}}};\n  const int ETALE[{nl}] = {{{etale}}};\n  long k = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (k >= (long){nl} * ({sx} + {sy} + {sz})) return;\n  int l = (int)(k / ({sx} + {sy} + {sz})), i = (int)(k % ({sx} + {sy} + {sz}));\n  int a = i < {sx} ? 0 : (i < {sx} + {sy} ? 1 : 2);\n  int j = a == 0 ? i : (a == 1 ? i - {sx} : i - {sx} - {sy});\n  int n = a == 0 ? {sx} : (a == 1 ? {sy} : {sz});\n  int c = a == 0 ? MX + j * {px} : (a == 1 ? MY + j * {py} : MZ + j * {pz});\n  double e = a == 1 ? SY[l] : SXZ[l];\n  double entree = __dmul_rn((double)c, e);\n  double h = __dadd_rn(wrapd(entree), OFF[T[l]][a]);\n  int cell = floor_i32(h);\n  double dd = __dsub_rn(h, (double)cell);\n  float d = __double2float_rn(dd);\n  if (a == 1 && ETALE[l]) {{\n    double limited = (entree >= 0.0 && entree < dd) ? entree : dd;\n    const double eps = __longlong_as_double(0x3e7ad7f2a0000000LL);\n    double f = __dmul_rn((double)floor_i32(__dadd_rn(__ddiv_rn(limited, FUD[l]), eps)), FUD[l]);\n    d = __double2float_rn(__dsub_rn(dd, f));\n  }}\n  float *B = MEM + AX_{r} + (long)l * {pas} + (a == 0 ? 0 : (a == 1 ? 3 * {sx} : 3 * {sx} + 3 * {sy}));\n  B[j] = __int_as_float(cell);\n  B[n + j] = d;\n  B[2 * n + j] = smooth(__double2float_rn(dd));\n}}"
        ))
    }

    /// NoiseSampler (tampon a zero puis add_to_volume, amplitude 1) d'une pile donnee, axes x, y, z.
    fn fn_volume_cle(&mut self, nom: &str, n: &dyn Noise, xz: f64, y: f64) -> R<String> {
        let mut c = Vec::new();
        self.couches(n, xz, y, 1.0, &mut c)?;
        let mut s = format!(
            "__device__ __noinline__ float {nom}(int x, int y, int z) {{\n  float acc = 0.0f;\n"
        );
        for (t, sxz, sy, a, fudge) in c {
            let appel = match fudge {
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
                "  acc = __fadd_rn(acc, __fmul_rn({}, {appel}));",
                f32lit(a)
            )?;
        }
        writeln!(s, "  return acc;\n}}")?;
        self.defs.push_str(&s);
        Ok(nom.to_string())
    }
    fn point(&mut self, n: &dyn Noise) -> R<String> {
        if let Some(p) = n.as_perlin() {
            let t = self.table(p);
            let nom = format!("pt_perlin{t}");
            if self
                .points
                .insert(p as *const PerlinNoise as usize, nom.clone())
                .is_none()
            {
                writeln!(
                    self.defs,
                    "__device__ float {nom}(double x, double y, double z) {{ return perlin_get3(PERM[{t}], OFF[{t}], x, y, z); }}"
                )?;
            }
            return Ok(nom);
        }
        let s = n.as_stack().ok_or("bruit non pris en charge (point)")?;
        let cle = s as *const _ as *const u8 as usize;
        if let Some(nom) = self.points.get(&cle) {
            return Ok(nom.clone());
        }
        let mut corps = String::from("  float v = 0.0f;\n");
        for l in s.layers() {
            let f = self.point(&*l.noise)?;
            let fr = f64lit(l.frequency);
            writeln!(
                corps,
                "  v = __fadd_rn(v, __fmul_rn({}, {f}(__dmul_rn(x, {fr}), __dmul_rn(y, {fr}), __dmul_rn(z, {fr}))));",
                f32lit(l.amplitude)
            )?;
        }
        let nom = format!("pt_pile{}", self.points.len());
        writeln!(
            self.defs,
            "__device__ __noinline__ float {nom}(double x, double y, double z) {{\n{corps}  return v;\n}}"
        )?;
        self.points.insert(cle, nom.clone());
        Ok(nom)
    }

    // ---------- splines : une fonction CUDA par noeud multipoint, coordonnees passees en tableau ----------
    fn spline_fn(
        &mut self,
        m: &MultipointSpline<FunctionId>,
        coords: &mut Vec<FunctionId>,
    ) -> R<String> {
        let cle = m as *const _ as usize;
        let ci = match coords.iter().position(|&c| c == m.coordinate()) {
            Some(i) => i,
            None => {
                coords.push(m.coordinate());
                coords.len() - 1
            }
        };
        if let Some(n) = self.splines.get(&cle) {
            return Ok(n.clone());
        }
        let pts = m.points();
        let n = pts.len();
        let mut valeurs = Vec::new();
        for p in pts {
            valeurs.push(match &p.value {
                CubicSpline::Constant(v) => f32lit(*v),
                CubicSpline::Multipoint(sub) => format!("{}(C)", self.spline_fn(sub, coords)?),
            });
        }
        let locs = pts
            .iter()
            .map(|p| f32lit(p.location))
            .collect::<Vec<_>>()
            .join(", ");
        let nom = format!("spline{}", self.splines.len());
        let mut s = format!(
            "__device__ __noinline__ float {nom}(const float *C) {{\n  float input = C[{ci}];\n  const float LOC[{n}] = {{{locs}}};\n  int first = 0, remaining = {n};\n  while (remaining != 0) {{ int half = remaining / 2; int middle = first + half; if (input < LOC[middle]) remaining = half; else {{ first = middle + 1; remaining -= half + 1; }} }}\n"
        );
        for (e, cond) in [
            (0usize, "first == 0".to_string()),
            (n - 1, format!("first == {n}")),
        ] {
            let p = &pts[e];
            let corps = if p.derivative == 0.0 {
                format!("return {};", valeurs[e])
            } else {
                format!(
                    "float v = {}; return __fadd_rn(v, __fmul_rn({}, __fsub_rn(input, {})));",
                    valeurs[e],
                    f32lit(p.derivative),
                    f32lit(p.location)
                )
            };
            writeln!(s, "  if ({cond}) {{ {corps} }}")?;
        }
        writeln!(s, "  switch (first) {{")?;
        for i in 1..n {
            let (l, r) = (&pts[i - 1], &pts[i]);
            writeln!(
                s,
                "  case {i}: {{\n    float width = __fsub_rn({rl}, {ll});\n    float alpha = __fdiv_rn(__fsub_rn(input, {ll}), width);\n    float lv = {lv}; float rv = {rv};\n    float lc = __fsub_rn(__fmul_rn({ld}, width), __fsub_rn(rv, lv));\n    float rc = __fadd_rn(__fmul_rn({nrd}, width), __fsub_rn(rv, lv));\n    float linear = __fadd_rn(lv, __fmul_rn(alpha, __fsub_rn(rv, lv)));\n    float corr = __fadd_rn(lc, __fmul_rn(alpha, __fsub_rn(rc, lc)));\n    return __fadd_rn(linear, __fmul_rn(__fmul_rn(alpha, __fsub_rn(1.0f, alpha)), corr));\n  }}",
                rl = f32lit(r.location),
                ll = f32lit(l.location),
                lv = valeurs[i - 1],
                rv = valeurs[i],
                ld = f32lit(l.derivative),
                nrd = f32lit(-r.derivative)
            )?;
        }
        writeln!(s, "  }}\n  return qnan();\n}}")?;
        self.defs.push_str(&s);
        self.splines.insert(cle, nom.clone());
        Ok(nom)
    }

    /// Expression de la valeur de `id` dans le noyau courant (litteral, variable ou lecture memoire).
    fn valeur(&mut self, ssa: &mut Ssa, id: FunctionId, g: usize, ctx_3d: bool) -> R<String> {
        if let Some(v) = self.constante(id) {
            return Ok(f32lit(v));
        }
        if ctx_3d && self.sans_y(id)? {
            // Valeur sans Y : calculee une fois par colonne par la passe 2D (comme le Slice de Horizon)
            let slot = match self.bords[g].iter().position(|&b| b == id) {
                Some(s) => s,
                None => {
                    self.bords[g].push(id);
                    self.bords[g].len() - 1
                }
            };
            return Ok(format!("MEM[COL_{g}_{slot} + ci]"));
        }
        if let Some(v) = ssa.vars.get(&id) {
            return Ok(v.clone());
        }
        let node = self.node(id)?;
        let d3 = !self.sans_y(id)?;
        let genre: &'static str = match &node {
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
        *self.compte.entry(genre).or_default() += 1;
        let expr: String = match node {
            GraphNode::Reference { input, .. }
            | GraphNode::Cache(input)
            | GraphNode::BlendDensity(input) => {
                // BlendDensity : sans melange avec d'anciens chunks, la valeur passe telle quelle
                let v = self.valeur(ssa, input, g, d3)?;
                ssa.vars.insert(id, v.clone());
                return Ok(v);
            }
            GraphNode::WorldField(f) => f32lit(if matches!(f, WorldField::BlendAlpha) {
                1.0
            } else {
                0.0
            }),
            GraphNode::OffsetNoise { noise, kind } => {
                let n = self.c.get_noise_sampler(noise)?;
                self.gardes.push(Box::new(n.clone()));
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
                self.gardes.push(Box::new(n.clone()));
                let zero = |s: &Self, f: FunctionId| matches!(s.c.graph().node(f), Some(GraphNode::Constant(v)) if v.to_bits() == 0);
                if zero(self, shift_x) && zero(self, shift_y) && zero(self, shift_z) {
                    if d3 {
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
                    let sx = self.valeur(ssa, shift_x, g, d3)?;
                    let sz = self.valeur(ssa, shift_z, g, d3)?;
                    let sy = if zero(self, shift_y) {
                        None
                    } else {
                        Some(self.valeur(ssa, shift_y, g, d3)?)
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
                let v = self.valeur(ssa, input, g, d3)?;
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
                    self.valeur(ssa, input, g, d3)?,
                    f32lit(min),
                    f32lit(max)
                )
            }
            GraphNode::Binary { kind, left, right } => {
                self.binaire(ssa, kind, left, right, g, d3)?
            }
            GraphNode::Choice {
                input,
                min,
                max,
                inside,
                outside,
            } => {
                let i = self.valeur(ssa, input, g, d3)?;
                let a = self.valeur(ssa, inside, g, d3)?;
                let b = self.valeur(ssa, outside, g, d3)?;
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
                let a = self.valeur(ssa, alpha, g, d3)?;
                let f = self.valeur(ssa, first, g, d3)?;
                let s = self.valeur(ssa, second, g, d3)?;
                format!(
                    "(({a}) == 0.0f) ? ({f}) : ((({a}) == 1.0f) ? ({s}) : lerpi({a}, {f}, {s}))"
                )
            }
            GraphNode::IntervalSelect {
                input,
                thresholds,
                branches,
            } => {
                let v = self.valeur(ssa, input, g, d3)?;
                let mut bs = Vec::new();
                for b in &branches {
                    bs.push(self.valeur(ssa, *b, g, d3)?);
                }
                let mut e = bs.last().ok_or("aucune branche")?.clone();
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
                    return Err("Gradient en mode Repeat non pris en charge".into());
                }
                let facteur = (high - low) / to.wrapping_sub(from) as f32;
                let c = ["x", "y", "z"][axis as usize];
                format!(
                    "__fadd_rn({}, __fmul_rn((float)(min(max({c}, {}), {}) - ({from})), {}))",
                    f32lit(low),
                    from.min(to),
                    from.max(to),
                    f32lit(facteur)
                )
            }
            GraphNode::Spline(s) => match &s {
                CubicSpline::Constant(v) => f32lit(*v),
                CubicSpline::Multipoint(m) => {
                    let mut coords = Vec::new();
                    let f = self.spline_fn(m, &mut coords)?;
                    let mut vals = Vec::new();
                    for c in coords {
                        vals.push(self.valeur(ssa, c, g, d3)?);
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
            GraphNode::Interpolated { input, xz, y } => {
                self.interpole(ssa, id, input, [xz, y, xz], g)?
            }
            GraphNode::BlendedNoise(def) => {
                // BlendedNoise::compile : Lerp(clamp(main + 0.5, 0, 1), min_limit, max_limit), trois piles
                // de bruits de Perlin etales, tirees comme DensityCompiler (hachage de "minecraft:terrain")
                let mut hasard = horizon_core::random::XoroshiroRandom::new(self.graine)
                    .fork_positional()
                    .from_hash_of("minecraft:terrain");
                let boite = Box::new(def.create_fbm_set(&mut hasard)?);
                let ptr: *const horizon_core::density::BlendedFbmSet = &*boite;
                self.gardes.push(boite); // le contenu d'une Box ne bouge plus tant qu'elle vit
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
                let nom = format!("melange_axes{}", self.axes.len());
                writeln!(
                    self.defs,
                    "__device__ __noinline__ float {nom}(const float *MEM, int ix, int iy, int iz) {{\n  float a = clampj(__fadd_rn({main}(MEM, ix, iy, iz), {}), 0.0f, 1.0f);\n  if (a == 0.0f) return {minl}(MEM, ix, iy, iz);\n  if (a == 1.0f) return {maxl}(MEM, ix, iy, iz);\n  return lerpi(a, {minl}(MEM, ix, iy, iz), {maxl}(MEM, ix, iy, iz));\n}}",
                    f32lit(0.5)
                )?;
                let v = self.var();
                writeln!(ssa.code, "  float {v} = {nom}(MEM, ix, iy, iz);")?;
                ssa.vars.insert(id, v.clone());
                return Ok(v);
            }
            autre => {
                let d = format!("{autre:?}");
                return Err(format!("noeud non pris en charge : {}", &d[..d.len().min(80)]).into());
            }
        };
        let v = self.var();
        writeln!(ssa.code, "  float {v} = {expr};")?;
        ssa.vars.insert(id, v.clone());
        Ok(v)
    }

    /// DensityCompiler::compile_binary : elagage min/max par intervalles, constantes en operations unaires.
    fn binaire(
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
                return self.valeur(ssa, left, g, d3);
            }
            if (min && b.max() < a.min()) || (!min && b.min() > a.max()) {
                return self.valeur(ssa, right, g, d3);
            }
        }
        if let Some(c) = self.constante(left) {
            let v = self.valeur(ssa, right, g, d3)?;
            let c = f32lit(c);
            return Ok(match kind {
                GraphBinary::Add => format!("__fadd_rn({v}, {c})"),
                GraphBinary::Subtract => format!("__fsub_rn({c}, {v})"),
                GraphBinary::Multiply => format!("__fmul_rn({v}, {c})"),
                GraphBinary::Divide => format!("__fdiv_rn({c}, {v})"),
                GraphBinary::Minimum => format!("(({c}) < ({v}) ? ({c}) : ({v}))"),
                GraphBinary::Maximum => format!("(({c}) > ({v}) ? ({c}) : ({v}))"),
            });
        }
        if let Some(c) = self.constante(right) {
            let v = self.valeur(ssa, left, g, d3)?;
            return Ok(match kind {
                GraphBinary::Add => format!("__fadd_rn({v}, {})", f32lit(c)),
                GraphBinary::Subtract => format!("__fadd_rn({v}, {})", f32lit(-c)),
                GraphBinary::Multiply => format!("__fmul_rn({v}, {})", f32lit(c)),
                GraphBinary::Divide => format!("__fmul_rn({v}, {})", f32lit(1.0 / c)),
                GraphBinary::Minimum => format!("(({0}) < ({v}) ? ({0}) : ({v}))", f32lit(c)),
                GraphBinary::Maximum => format!("(({0}) > ({v}) ? ({0}) : ({v}))", f32lit(c)),
            });
        }
        let l = self.valeur(ssa, left, g, d3)?;
        let r = self.valeur(ssa, right, g, d3)?;
        Ok(match kind {
            GraphBinary::Add => format!("__fadd_rn({l}, {r})"),
            GraphBinary::Subtract => format!("__fsub_rn({l}, {r})"),
            GraphBinary::Multiply => format!("__fmul_rn({l}, {r})"),
            GraphBinary::Divide => format!("__fdiv_rn({l}, {r})"),
            GraphBinary::Minimum => format!("(({r}) < ({l}) ? ({r}) : ({l}))"),
            GraphBinary::Maximum => format!("(({r}) > ({l}) ? ({r}) : ({l}))"),
        })
    }

    /// InterpolatedSampler (volume de blocs aligne) : coins de cellules sur une grille grossiere,
    /// lerp Z puis X a chaque niveau, puis increments successifs en Y depuis le bas de la cellule.
    fn interpole(
        &mut self,
        ssa: &mut Ssa,
        id: FunctionId,
        input: FunctionId,
        cellule: [i32; 3],
        g: usize,
    ) -> R<String> {
        if g != 0 {
            return Err("interpolation imbriquee non prise en charge".into());
        }
        let v = self.grilles[0];
        if v.pas != [1, 1, 1] || (0..3).any(|i| v.tailles[i] % cellule[i] != 0) {
            return Err("la grille de blocs doit etre alignee sur les cellules".into());
        }
        if !self.interps.contains_key(&id) {
            // Volume aligne : (taille / cellule) + 1 coins par axe (InterpolatedSampler::sample_block_volume)
            let tailles = std::array::from_fn(|i| v.tailles[i] / cellule[i] + 1);
            self.grilles.push(Grille {
                tailles,
                pas: cellule,
            });
            self.bords.push(Vec::new());
            self.racines.push(input);
            self.interps.insert(id, (self.grilles.len() - 1, cellule));
        }
        let (gl, c) = self.interps[&id];
        let [lsx, lsy, _] = self.grilles[gl].tailles;
        let inv = c.map(|c| 1.0 / c as f32);
        let r = self.var();
        writeln!(
            ssa.code,
            "  float {r};\n  {{\n    int cx = fdiv(x, {cx}) - fdiv(MX, {cx}), cy = fdiv(y, {cy}) - fdiv(MY, {cy}), cz = fdiv(z, {cz}) - fdiv(MZ, {cz});\n    int ox = x - fdiv(x, {cx}) * {cx}, oy = y - fdiv(y, {cy}) * {cy}, oz = z - fdiv(z, {cz}) * {cz};\n    float ax = __fmul_rn((float)ox, {ix}), az = __fmul_rn((float)oz, {iz});\n    const float *S = MEM + SORTIE_{gl};\n    #define C(a, b, c) S[((long)(c) * {lsx} + (a)) * {lsy} + (b)]\n    float bottom = lerpi(ax, lerpi(az, C(cx, cy, cz), C(cx, cy, cz + 1)), lerpi(az, C(cx + 1, cy, cz), C(cx + 1, cy, cz + 1)));\n    float top = lerpi(ax, lerpi(az, C(cx, cy + 1, cz), C(cx, cy + 1, cz + 1)), lerpi(az, C(cx + 1, cy + 1, cz), C(cx + 1, cy + 1, cz + 1)));\n    #undef C\n    float inc = __fmul_rn(__fsub_rn(top, bottom), {iy});\n    float val = __fadd_rn(bottom, __fmul_rn(inc, 0.0f));\n    for (int k = 0; k < oy; ++k) val = __fadd_rn(val, inc);\n    {r} = val;\n  }}",
            cx = c[0],
            cy = c[1],
            cz = c[2],
            ix = f32lit(inv[0]),
            iy = f32lit(inv[1]),
            iz = f32lit(inv[2]),
        )?;
        Ok(r)
    }

    /// Les noyaux de toutes les grilles (les grilles d'interpolation apparaissent en cours de route).
    fn noyaux(&mut self) -> R<String> {
        let mut src = String::new();
        let mut g = 0;
        let mut corps3d = Vec::new();
        while g < self.grilles.len() {
            let mut ssa = Ssa::default();
            let r = self.valeur(&mut ssa, self.racines[g], g, true)?;
            corps3d.push((ssa.code, r));
            g += 1;
        }
        for (g, (code, r)) in corps3d.into_iter().enumerate() {
            let gr = self.grilles[g];
            let [sx, sy, sz] = gr.tailles;
            let [px, py, pz] = gr.pas;
            // Passe 2D : une valeur par colonne pour chaque noeud sans Y lu par un noeud 3D
            // Un thread par (colonne, valeur) : plus de parallelisme que par colonne seule
            let mut cas = String::new();
            for (s, b) in self.bords[g].clone().into_iter().enumerate() {
                let mut ssa = Ssa::default();
                let v = self.valeur(&mut ssa, b, g, false)?;
                writeln!(
                    cas,
                    "  case {s}: {{\n{}  MEM[COL_{g}_{s} + ci] = {v};\n  break; }}",
                    ssa.code
                )?;
            }
            let ns = self.bords[g].len().max(1);
            writeln!(
                src,
                "extern \"C\" __global__ void k2d_{g}(float *MEM, int MX, int MY, int MZ) {{\n  long k = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (k >= (long){sx} * {sz} * {ns}) return;\n  int s = (int)(k / ((long){sx} * {sz}));\n  long ci = k % ((long){sx} * {sz});\n  int x = MX + (int)(ci % {sx}) * {px}, z = MZ + (int)(ci / {sx}) * {pz}, y = 0; long gi = 0;\n  switch (s) {{\n{cas}  }}\n}}"
            )?;
            writeln!(
                src,
                "extern \"C\" __global__ void k3d_{g}(float *MEM, int MX, int MY, int MZ) {{\n  long gi = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (gi >= (long){sx} * {sy} * {sz}) return;\n  long ci = gi / {sy};\n  int ix = (int)(ci % {sx}), iy = (int)(gi % {sy}), iz = (int)(ci / {sx});\n  int x = MX + ix * {px}, y = MY + iy * {py}, z = MZ + iz * {pz};\n{code}  MEM[SORTIE_{g} + gi] = {r};\n}}"
            )?;
        }
        Ok(src)
    }
}

/// Le CUDA d'une region et la disposition de sa memoire (un seul tampon `MEM`).
pub(crate) struct Programme {
    pub src: String,
    /// Nombre de floats de `MEM`.
    pub taille: usize,
    pub grilles: Vec<Grille>,
    /// Nombre de valeurs sans Y par colonne, pour chaque grille.
    pub bords: Vec<usize>,
    /// Noyaux de tables d'axes : (indice, grille, nombre de threads).
    pub axes: Vec<(usize, usize, usize)>,
    pub lignes: usize,
}

/// Genere le CUDA de `racine` pour une grille de blocs de `tailles`.
pub(crate) fn generer(
    c: &DensityCompiler,
    racine: FunctionId,
    graine: i64,
    tailles: [i32; 3],
) -> R<Programme> {
    let mut g = Gen::new(
        c,
        Grille {
            tailles,
            pas: [1, 1, 1],
        },
        racine,
        graine,
    );
    let noyaux = g.noyaux()?;
    if !g.externes.is_empty() {
        return Err("tampon CPU inattendu".into());
    }
    let mut defs = String::new();
    let mut taille = 0usize;
    for (k, gr) in g.grilles.iter().enumerate() {
        writeln!(defs, "#define SORTIE_{k} {taille}L")?;
        taille += gr.n();
    }
    for (k, b) in g.bords.iter().enumerate() {
        for s in 0..b.len() {
            writeln!(defs, "#define COL_{k}_{s} {taille}L")?;
            taille += g.grilles[k].colonnes();
        }
    }
    let mut noyaux_axes = String::new();
    let mut axes = Vec::new();
    for r in 0..g.axes.len() {
        let (gr, couches) = &g.axes[r];
        writeln!(defs, "#define AX_{r} {taille}L")?;
        let n = couches.len() * pas_couche(g.grilles[*gr].tailles);
        taille += n;
        noyaux_axes.push_str(&g.noyau_axes(r)?);
        noyaux_axes.push('\n');
        axes.push((r, *gr, n / 3));
    }
    writeln!(
        defs,
        "#define CGX {}\n#define CGY {}\n#define CGZ {}",
        tailles[0], tailles[1], tailles[2]
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
        "}};\n{BASE}\n{OPS}\n{}\n{noyaux}\n{noyaux_axes}\n{EXTRAIT}",
        g.defs
    )?;
    let lignes = src.lines().count();
    Ok(Programme {
        src,
        taille,
        grilles: g.grilles.clone(),
        bords: g.bords.iter().map(|b| b.len()).collect(),
        axes,
        lignes,
    })
}
