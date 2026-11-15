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
__device__ __forceinline__ float jmax(float a, float b) {
    if (isnan(a) || isnan(b)) return qnan();
    if (a == 0.0f && b == 0.0f) return __int_as_float(__float_as_int(a) & __float_as_int(b));
    return a >= b ? a : b;
}
__device__ __forceinline__ float clampj(float v, float lo, float hi) { return v < lo ? lo : jmin(v, hi); }
// Java (int)Math.floor(double) : sature, NaN donne 0
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
    /// Spline racine (noeud du graphe) : fonction CUDA et ordre complet de ses coordonnees. Une spline
    /// reutilisee doit recevoir exactement le meme tableau de coordonnees qu'a sa generation.
    splines: HashMap<FunctionId, (String, Vec<FunctionId>)>,
    nb_splines: usize,
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
    /// Mode point : `sample_value` de Horizon (un point a la fois) au lieu de `sample_volume`.
    /// Bruits par `get3`, operations binaires avec leurs raccourcis et java_min/java_max.
    mode_point: bool,
    /// Fonctions `float pfN(int x, int y, int z)` deja generees en mode point.
    fonctions_point: HashMap<FunctionId, String>,
    /// En mode point, pendant la boucle d'une descente de `find_top_surface` : les sous-graphes sans Y lus
    /// par un noeud a Y, calcules une fois avant la boucle (variables `hK`).
    hisses: Option<Vec<FunctionId>>,
    nb_descentes: usize,
}

/// Une couche precalculee : pour chaque axe, cellule (bits d'entier), position et lissage.
fn pas_couche(t: [i32; 3]) -> usize {
    3 * (t[0] + t[1] + t[2]) as usize
}

impl<'a> Gen<'a> {
    fn new(c: &'a DensityCompiler, principale: Grille, racine: Option<FunctionId>, graine: i64) -> Self {
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
            nb_splines: 0,
            grilles: vec![principale],
            racines: racine.into_iter().collect(),
            interps: HashMap::new(),
            bords: vec![Vec::new()],
            externes: Vec::new(),
            nv: 0,
            compte: BTreeMap::new(),
            mode_point: false,
            fonctions_point: HashMap::new(),
            hisses: None,
            nb_descentes: 0,
        }
    }
    /// `__device__ float sdN(int x, int z, float upper)` : la descente de FindTopSurfaceSampler depuis la
    /// borne haute, la densite en mode point. Ses sous-graphes sans Y sont calcules une fois avant la boucle :
    /// ils ne changent pas d'une cellule a l'autre de la colonne.
    fn descente_surface(&mut self, density: FunctionId, lower: i32, height: i32) -> R<String> {
        let ancien_mode = std::mem::replace(&mut self.mode_point, true);
        let ancien_hisses = self.hisses.replace(Vec::new());
        let mut boucle = Ssa::default();
        let v = self.valeur(&mut boucle, density, 0, true);
        let hisses = std::mem::replace(&mut self.hisses, ancien_hisses).unwrap_or_default();
        let r = v.and_then(|v| {
            let mut avant = Ssa::default();
            for (k, id) in hisses.iter().enumerate() {
                let hv = self.valeur(&mut avant, *id, 0, false)?;
                writeln!(avant.code, "  float h{k} = {hv};")?;
            }
            Ok((v, avant.code))
        });
        self.mode_point = ancien_mode;
        let (v, prelude) = r?;
        let nom = format!("sd{}", self.nb_descentes);
        self.nb_descentes += 1;
        writeln!(
            self.defs,
            "__device__ __noinline__ float {nom}(int x, int z, float upper) {{\n  int y = 0;\n{prelude}  int sy = (int)((unsigned)floor_sat((double)__fdiv_rn(upper, {h})) * {height}u);\n  if (sy <= {lower}) return (float)({lower});\n  while (sy >= {lower}) {{\n    int y = sy;\n{boucle}    if ({v} > 0.0f) return (float)sy;\n    sy -= {height};\n  }}\n  return (float)({lower});\n}}",
            h = f32lit(height as f32),
            boucle = boucle.code
        )?;
        Ok(nom)
    }
    /// `__device__ void NOM(int x, int z, int qy0, int nqy, float *V, long pas)` : les racines en mode
    /// point (comme `sample_value`) en chaque quart Y j d'une colonne, a y = (qy0 + j) << 2, dans
    /// V[j * pas + k] pour la racine k. Comme dans `descente_surface`, les sous-graphes sans Y sont
    /// calcules une fois avant la boucle, et partages entre les racines (le meme noeud, une seule fois).
    fn colonne_points(&mut self, nom: &str, racines: &[FunctionId]) -> R<String> {
        let ancien_mode = std::mem::replace(&mut self.mode_point, true);
        let ancien_hisses = self.hisses.replace(Vec::new());
        let mut boucle = Ssa::default();
        let mut vs = Vec::new();
        let mut erreur = None;
        for &r in racines {
            match self.valeur(&mut boucle, r, 0, true) {
                Ok(v) => vs.push(v),
                Err(e) => {
                    erreur = Some(e);
                    break;
                }
            }
        }
        let hisses = std::mem::replace(&mut self.hisses, ancien_hisses).unwrap_or_default();
        let r = match erreur {
            Some(e) => Err(e),
            None => (|| -> R<String> {
                let mut avant = Ssa::default();
                for (k, id) in hisses.iter().enumerate() {
                    let hv = self.valeur(&mut avant, *id, 0, false)?;
                    writeln!(avant.code, "  float h{k} = {hv};")?;
                }
                Ok(avant.code)
            })(),
        };
        self.mode_point = ancien_mode;
        let prelude = r?;
        let mut ecritures = String::new();
        for (k, v) in vs.iter().enumerate() {
            writeln!(ecritures, "    V[j * pas + {k}] = {v};")?;
        }
        Ok(format!(
            "__device__ __noinline__ void {nom}(int x, int z, int qy0, int nqy, float *V, long pas) {{\n  int y = 0;\n{prelude}  for (int j = 0; j < nqy; ++j) {{\n    int y = (int)((unsigned)(qy0 + j) << 2);\n{boucle}{ecritures}  }}\n}}",
            boucle = boucle.code
        ))
    }
    /// `__device__ float pfN(int x, int y, int z)` : la valeur de `id` en un point, exactement comme
    /// `DensitySampler::sample_value` de Horizon.
    fn fonction_point(&mut self, id: FunctionId) -> R<String> {
        if let Some(nom) = self.fonctions_point.get(&id) {
            return Ok(nom.clone());
        }
        let ancien = std::mem::replace(&mut self.mode_point, true);
        // Une fonction a part ne voit pas les variables hissees de sa descente appelante
        let ancien_hisses = self.hisses.take();
        let mut ssa = Ssa::default();
        let v = self.valeur(&mut ssa, id, 0, false);
        self.hisses = ancien_hisses;
        self.mode_point = ancien;
        let v = v?;
        let nom = format!("pf{}", self.fonctions_point.len());
        writeln!(self.defs, "__device__ __noinline__ float {nom}(int x, int y, int z) {{\n{}  return {v};\n}}", ssa.code)?;
        self.fonctions_point.insert(id, nom.clone());
        Ok(nom)
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
    fn spline_racine(&mut self, id: FunctionId, m: &MultipointSpline<FunctionId>) -> R<(String, Vec<FunctionId>)> {
        if let Some(r) = self.splines.get(&id) {
            return Ok(r.clone());
        }
        let mut coords = Vec::new();
        let mut locales = HashMap::new();
        let f = self.spline_fn(m, &mut coords, &mut locales)?;
        self.splines.insert(id, (f.clone(), coords.clone()));
        Ok((f, coords))
    }
    /// Une fonction par noeud multipoint d'une meme spline racine ; `locales` memorise les sous-splines
    /// de cette generation seulement (leurs adresses ne sont stables que pendant celle-ci).
    fn spline_fn(
        &mut self,
        m: &MultipointSpline<FunctionId>,
        coords: &mut Vec<FunctionId>,
        locales: &mut HashMap<usize, String>,
    ) -> R<String> {
        let cle = m as *const _ as usize;
        let ci = match coords.iter().position(|&c| c == m.coordinate()) {
            Some(i) => i,
            None => {
                coords.push(m.coordinate());
                coords.len() - 1
            }
        };
        if let Some(n) = locales.get(&cle) {
            return Ok(n.clone());
        }
        let pts = m.points();
        let n = pts.len();
        let mut valeurs = Vec::new();
        for p in pts {
            valeurs.push(match &p.value {
                CubicSpline::Constant(v) => f32lit(*v),
                CubicSpline::Multipoint(sub) => format!("{}(C)", self.spline_fn(sub, coords, locales)?),
            });
        }
        let locs = pts
            .iter()
            .map(|p| f32lit(p.location))
            .collect::<Vec<_>>()
            .join(", ");
        let nom = format!("spline{}", self.nb_splines);
        self.nb_splines += 1;
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
        locales.insert(cle, nom.clone());
        Ok(nom)
    }

    /// Expression de la valeur de `id` dans le noyau courant (litteral, variable ou lecture memoire).
    fn valeur(&mut self, ssa: &mut Ssa, id: FunctionId, g: usize, ctx_3d: bool) -> R<String> {
        if let Some(v) = self.constante(id) {
            return Ok(f32lit(v));
        }
        if ctx_3d && self.mode_point && self.hisses.is_some() && self.sans_y(id)? {
            // Boucle d'une descente : calcule une fois avant la boucle (descente_surface)
            let h = self.hisses.as_mut().unwrap();
            let slot = match h.iter().position(|&b| b == id) {
                Some(s) => s,
                None => {
                    h.push(id);
                    h.len() - 1
                }
            };
            return Ok(format!("h{slot}"));
        }
        if ctx_3d && !self.mode_point && self.sans_y(id)? {
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
            GraphNode::OffsetNoise { noise, kind } if self.mode_point => {
                // compile_shift_noise, chemin point : get3 puis * 4 en float (ShiftB : get3(z/4, x/4, 0))
                let n = self.c.get_noise_sampler(noise)?;
                self.gardes.push(Box::new(n.clone()));
                let pt = self.point(&*n)?;
                let q = f64lit(0.25);
                let appel = match kind {
                    ShiftNoiseKind::Shift => format!("{pt}(__dmul_rn((double)x, {q}), __dmul_rn((double)y, {q}), __dmul_rn((double)z, {q}))"),
                    ShiftNoiseKind::ShiftA => format!("{pt}(__dmul_rn((double)x, {q}), __dmul_rn((double)y, 0.0), __dmul_rn((double)z, {q}))"),
                    ShiftNoiseKind::ShiftB => format!("{pt}(__dmul_rn((double)z, {q}), __dmul_rn((double)x, {q}), 0.0)"),
                };
                format!("__fmul_rn({appel}, {})", f32lit(4.0))
            }
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
                    if self.mode_point {
                        // NoiseSampler::sample_value : get3(x * xz, y * y_scale, z * xz)
                        let pt = self.point(&*n)?;
                        let (xz, ys) = (f64lit(xz_scale), f64lit(y_scale));
                        format!("{pt}(__dmul_rn((double)x, {xz}), __dmul_rn((double)y, {ys}), __dmul_rn((double)z, {xz}))")
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
                    let (f, coords) = self.spline_racine(id, m)?;
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
            GraphNode::Interpolated { .. } | GraphNode::BlendedNoise(_) if self.mode_point => {
                return Err("interpolation et BlendedNoise non pris en charge en mode point".into());
            }
            GraphNode::Slice { axis, coordinate, input } if self.mode_point => {
                // SliceSampler : l'entree evaluee avec une coordonnee fixee
                let f = self.fonction_point(input)?;
                let mut c = ["x".to_string(), "y".to_string(), "z".to_string()];
                c[axis as usize] = coordinate.to_string();
                format!("{f}({}, {}, {})", c[0], c[1], c[2])
            }
            GraphNode::Surface { density, upper, lower, height } if self.mode_point => {
                // compile_surface_search : Slice Y = 0 autour de FindTopSurfaceSampler. La borne haute est
                // lue en y = 0 ; on descend de cellule en cellule jusqu'a une densite positive.
                if height <= 0 {
                    return Err("find_top_surface : hauteur de cellule non positive".into());
                }
                let fd = self.fonction_point(density)?;
                let fu = self.fonction_point(upper)?;
                let nom = format!("surface{}", self.fonctions_point.len());
                writeln!(
                    self.defs,
                    "__device__ __noinline__ float {nom}(int x, int z) {{\n  float upper = {fu}(x, 0, z);\n  int y = (int)((unsigned)floor_sat((double)__fdiv_rn(upper, {h})) * {height}u);\n  if (y <= {lower}) return (float)({lower});\n  while (y >= {lower}) {{\n    if ({fd}(x, y, z) > 0.0f) return (float)y;\n    y -= {height};\n  }}\n  return (float)({lower});\n}}",
                    h = f32lit(height as f32)
                )?;
                format!("{nom}(x, z)")
            }
            GraphNode::Surface { density, upper, lower, height } => {
                // FindTopSurfaceSampler::sample_volume sous le Slice Y = 0 : la borne haute en volume sur le
                // plan y = 0, puis la densite point par point en descendant de cellule en cellule.
                if height <= 0 {
                    return Err("find_top_surface : hauteur de cellule non positive".into());
                }
                let u = self.valeur(ssa, upper, g, d3)?;
                let fd = self.descente_surface(density, lower, height)?;
                let r = self.var();
                writeln!(ssa.code, "  float {r} = {fd}(x, z, {u});")?;
                r
            }
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
        // Mode point : UnaryOp::Minimum/Maximum hors bulk passent par java_min/java_max
        let point = self.mode_point;
        if let Some(c) = self.constante(left) {
            let v = self.valeur(ssa, right, g, d3)?;
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
        if let Some(c) = self.constante(right) {
            let v = self.valeur(ssa, left, g, d3)?;
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
        let l = self.valeur(ssa, left, g, d3)?;
        let r = self.valeur(ssa, right, g, d3)?;
        if point {
            // BinarySampler::sample_value : raccourcis sur la valeur de gauche, puis java_min/java_max
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
        let mut corps = Vec::new();
        // Les passes 2D dans la boucle : une interpolation lue par un noeud sans Y (surface
        // preliminaire) ajoute sa grille de coins pendant la passe 2D.
        while g < self.grilles.len() {
            let mut ssa = Ssa::default();
            let r = self.valeur(&mut ssa, self.racines[g], g, true)?;
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
            corps.push((ssa.code, r, cas));
            g += 1;
        }
        for (g, (code, r, cas)) in corps.into_iter().enumerate() {
            let gr = self.grilles[g];
            let [sx, sy, sz] = gr.tailles;
            let [px, py, pz] = gr.pas;
            let ns = self.bords[g].len().max(1);
            // y = MY : une interpolation lue en passe 2D tombe sur la premiere couche de coins
            writeln!(
                src,
                "extern \"C\" __global__ void k2d_{g}(float *MEM, int MX, int MY, int MZ) {{\n  long k = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (k >= (long){sx} * {sz} * {ns}) return;\n  int s = (int)(k / ((long){sx} * {sz}));\n  long ci = k % ((long){sx} * {sz});\n  int x = MX + (int)(ci % {sx}) * {px}, z = MZ + (int)(ci / {sx}) * {pz}, y = MY; long gi = 0;\n  switch (s) {{\n{cas}  }}\n}}"
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
        Some(racine),
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

/// Tables de permutation et decalages des bruits de Perlin utilises par `g`.
fn tables_cuda(g: &Gen) -> R<String> {
    let mut src = format!("__device__ const unsigned char PERM[{}][256] = {{\n", g.tables.len().max(1));
    for (p, _) in &g.tables {
        writeln!(src, "  {{{}}},", p.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))?;
    }
    writeln!(src, "}};\n__device__ const double OFF[{}][3] = {{", g.tables.len().max(1))?;
    for (_, o) in &g.tables {
        writeln!(src, "  {{{:e}, {:e}, {:e}}},", o[0], o[1], o[2])?;
    }
    src.push_str("};\n");
    Ok(src)
}

/// CUDA qui evalue chaque racine en une liste de points, comme `sample_value` de Horizon :
/// `kpt_K(const int *P, float *OUT, int n)` pour la racine K, points (x, y, z) a la suite dans P.
pub(crate) fn generer_points(c: &DensityCompiler, racines: &[FunctionId], graine: i64) -> R<String> {
    let premiere = *racines.first().ok_or("aucune racine")?;
    let mut g = Gen::new(c, Grille { tailles: [1, 1, 1], pas: [1, 1, 1] }, Some(premiere), graine);
    let mut noyaux = String::new();
    for (k, &r) in racines.iter().enumerate() {
        let f = g.fonction_point(r)?;
        writeln!(
            noyaux,
            "extern \"C\" __global__ void kpt_{k}(const int *P, float *OUT, int n) {{\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i >= n) return;\n  OUT[i] = {f}(P[3 * i], P[3 * i + 1], P[3 * i + 2]);\n}}"
        )?;
    }
    Ok(format!("{}{BASE}\n{OPS}\n{}\n{noyaux}", tables_cuda(&g)?, g.defs))
}

const ALEATOIRE: &str = include_str!("aleatoire.cu");
const AQUIFERE: &str = include_str!("aquifere.cu");

/// CUDA des aquiferes : les six racines (ordre de AquiferField::ALL : barriere, inondation, etalement,
/// lave, exclusion, niveau de surface) en mode point, puis les noyaux de `aquifere.cu`.
pub(crate) fn generer_aquifere(c: &DensityCompiler, racines: [FunctionId; 6], graine: i64) -> R<String> {
    let mut g = Gen::new(c, Grille { tailles: [1, 1, 1], pas: [1, 1, 1] }, None, graine);
    let mut defs = String::new();
    for (nom, r) in ["AQ_BARRIER", "AQ_FLOOD", "AQ_SPREAD", "AQ_LAVA", "AQ_EXCL", "AQ_SURFACE"].iter().zip(racines) {
        let f = g.fonction_point(r)?;
        writeln!(defs, "#define {nom} {f}")?;
    }
    Ok(format!("{}{BASE}\n{OPS}\n{}\n{defs}\n{ALEATOIRE}\n{AQUIFERE}", tables_cuda(&g)?, g.defs))
}

const BIOMES: &str = include_str!("biomes.cu");

/// CUDA des biomes : les six racines du climat (temperature, humidite, continentalite, erosion,
/// profondeur, bizarrerie) par colonne de quarts (`cl_colonne`, voir `Gen::colonne_points`), comme
/// `ClimateSampler::sample`, puis les noyaux de `biomes.cu`.
pub(crate) fn generer_biomes(c: &DensityCompiler, racines: [FunctionId; 6], graine: i64) -> R<String> {
    let mut g = Gen::new(c, Grille { tailles: [1, 1, 1], pas: [1, 1, 1] }, None, graine);
    let colonne = g.colonne_points("cl_colonne", &racines)?;
    Ok(format!("{}{BASE}\n{OPS}\n{}\n{colonne}\n{BIOMES}", tables_cuda(&g)?, g.defs))
}

/// CUDA qui evalue des bruits (`Noise::get3` de Horizon) en des points donnes en double :
/// `knz_K(const double *P, float *OUT, int n)` pour le bruit K.
pub(crate) fn generer_bruits(c: &DensityCompiler, bruits: &[&dyn Noise]) -> R<String> {
    let mut g = Gen::new(c, Grille { tailles: [1, 1, 1], pas: [1, 1, 1] }, None, 0);
    let mut noyaux = String::new();
    for (k, n) in bruits.iter().enumerate() {
        let f = g.point(*n)?;
        writeln!(
            noyaux,
            "extern \"C\" __global__ void knz_{k}(const double *P, float *OUT, int n) {{\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i >= n) return;\n  OUT[i] = {f}(P[3 * i], P[3 * i + 1], P[3 * i + 2]);\n}}"
        )?;
    }
    Ok(format!("{}{BASE}\n{OPS}\n{}\n{noyaux}\n{ALEATOIRE}\n{TIRAGES}", tables_cuda(&g)?, g.defs))
}

/// Tirages positionnels : premier next_float, premier next_double et trois next_float a la suite.
const TIRAGES: &str = r#"
extern "C" __global__ void ktirages(const int *P, float *F, double *D, float *F3, int n, unsigned long long LO, unsigned long long HI) {
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

/// CUDA du jalon S2 de la surface : zoom des biomes et temperature (bruits simplex).
pub(crate) fn generer_surface_s2(
    c: &DensityCompiler,
    hauteur: &horizon_core::noise::SimplexNoise,
    gel: &horizon_core::noise::NoiseStack,
    info: &horizon_core::noise::SimplexNoise,
) -> R<String> {
    let mut g = Gen::new(c, Grille { tailles: [1, 1, 1], pas: [1, 1, 1] }, None, 0);
    let mut defs = String::new();
    writeln!(defs, "#define T_HAUTEUR {}", g.table_de(hauteur.table()))?;
    writeln!(defs, "#define T_INFO {}", g.table_de(info.table()))?;
    let couches = gel.layers();
    if couches.len() != 3 {
        return Err("bruit de gel : 3 couches attendues".into());
    }
    for (k, l) in couches.iter().enumerate() {
        let sx = l.noise.as_simplex().ok_or("bruit de gel : couche non simplex")?;
        writeln!(defs, "#define T_GEL{k} {}", g.table_de(sx.table()))?;
        writeln!(defs, "#define T_GEL_FREQ_{k} {}", f64lit(l.frequency))?;
        writeln!(defs, "#define T_GEL_AMP_{k} {}", f32lit(l.amplitude))?;
    }
    Ok(format!("{}{BASE}\n{OPS}\n{defs}\n{SURFACE}", tables_cuda(&g)?))
}

/// Le programme de surface d'un monde : le CUDA du noyau `ksurface` et les racines des filons
/// calculees en volume (une grille d'un chunk par racine, dans l'ordre de `volumes`).
/// Cote, en chunks, des regions de la surface preliminaire.
pub(crate) const NIVEAU_R: i32 = 8;
pub(crate) struct ProgrammeSurface {
    pub src: String,
    pub volumes: Vec<Programme>,
    /// La surface preliminaire sur le plan y = 0 d'une region de NIVEAU_R x NIVEAU_R chunks.
    pub niveau: Programme,
    /// Les grilles des filons d'une region : y de depart et hauteur.
    pub filons_y: (i32, i32),
}

/// La regle de surface traduite en CUDA (fonctions R_k et C_k) et la descente des colonnes.
pub(crate) fn generer_surface(c: &DensityCompiler, s: &crate::SurfaceSetup, graine: i64) -> R<ProgrammeSurface> {
    use crate::{SurfaceCondition as SC, SurfaceRule as SR};
    let mut g = Gen::new(c, Grille { tailles: [1, 1, 1], pas: [1, 1, 1] }, None, graine);
    let mut defs = String::new();
    // Bruits de la colonne, des conditions et des bandes
    let pt_surf = g.point(s.surface_noise)?;
    let pt_sec = g.point(s.secondary_noise)?;
    let pt_clay = g.point(s.clay_noise)?;
    let mut pts = Vec::new();
    for n in &s.noises {
        pts.push(g.point(*n)?);
    }
    // Temperature
    let (hauteur, gel, info) = s.temperature_noises;
    writeln!(defs, "#define T_HAUTEUR {}", g.table_de(hauteur.table()))?;
    writeln!(defs, "#define T_INFO {}", g.table_de(info.table()))?;
    let couches = gel.layers();
    if couches.len() != 3 {
        return Err("bruit de gel : 3 couches attendues".into());
    }
    for (k, l) in couches.iter().enumerate() {
        let sx = l.noise.as_simplex().ok_or("bruit de gel : couche non simplex")?;
        writeln!(defs, "#define T_GEL{k} {}", g.table_de(sx.table()))?;
        writeln!(defs, "#define T_GEL_FREQ_{k} {}", f64lit(l.frequency))?;
        writeln!(defs, "#define T_GEL_AMP_{k} {}", f32lit(l.amplitude))?;
    }
    writeln!(defs, "#define SF_MINY ({})\n#define SF_H {}\n#define SF_ZOOM ({}ll)\n#define SF_MER ({})", s.min_y, s.height, s.zoom_seed, s.sea_level)?;
    let [e_defaut, e_air, e_fluide, e_lave] = s.fill_states;
    writeln!(defs, "#define SF_E_DEFAUT {e_defaut}\n#define SF_E_AIR {e_air}\n#define SF_E_FLUIDE {e_fluide}\n#define SF_E_LAVE {e_lave}")?;
    // Etats a fluide ecrits par la regle : la surface CPU les marque pour le post-traitement
    let fluides = s.fluid_states.iter().map(|e| format!("e == {e}")).collect::<Vec<_>>();
    writeln!(
        defs,
        "__device__ __forceinline__ bool sf_fluide(int e) {{ return {}; }}",
        if fluides.is_empty() { "false".to_string() } else { fluides.join(" || ") }
    )?;
    // Extensions (jalon S4) : biomes, blocs et bruits des piliers et des icebergs
    if let Some(e) = &s.extensions {
        if e.frozen_oceans.len() > 2 {
            return Err("plus de deux oceans geles".into());
        }
        let mut bruits = Vec::new();
        for n in e.noises {
            bruits.push(g.point(n)?);
        }
        let id = |b: Option<&u32>| b.map_or(-1, |&b| i64::from(b));
        writeln!(
            defs,
            "#define SF_EXT 1\n#define SF_ERODE ({})\n#define SF_GELE0 ({})\n#define SF_GELE1 ({})\n#define SF_DEFAUT {}\n#define SF_GLACE {}\n#define SF_NEIGE {}",
            id(e.eroded_badlands.as_ref()),
            id(e.frozen_oceans.first()),
            id(e.frozen_oceans.get(1)),
            e.default_block,
            e.packed_ice,
            e.snow
        )?;
        for (nom, f) in ["BD_S", "BD_P", "BD_R", "IC_S", "IC_P", "IC_R"].iter().zip(&bruits) {
            writeln!(defs, "#define {nom} {f}")?;
        }
    }
    // Filons : densite et richesse en volume (une grille par racine et par region de NIVEAU_R x NIVEAU_R
    // chunks, sur les y ou une densite peut etre positive), espacement en mode point
    let mut volumes: Vec<FunctionId> = Vec::new();
    let mut ecarts = HashMap::new();
    let (haut_monde, mut bas_v, mut haut_v) = (s.min_y + s.height - 1, i32::MAX, i32::MIN);
    for r in &s.rules {
        if let SR::OreVein { density, richness, gap, positive_y: [bas, haut], .. } = r {
            for id in [*density, *richness] {
                if !volumes.contains(&id) {
                    volumes.push(id);
                }
            }
            if !ecarts.contains_key(gap) {
                let f = g.fonction_point(*gap)?;
                ecarts.insert(*gap, f);
            }
            let (bas, haut) = ((*bas).max(s.min_y), (*haut).min(haut_monde));
            if bas <= haut {
                bas_v = bas_v.min(bas);
                haut_v = haut_v.max(haut);
            }
        }
    }
    // Aligne sur 16 (multiple des cellules d'interpolation) : les valeurs ne dependent pas de l'etendue
    let (vlo, vh) = if bas_v <= haut_v {
        let vlo = bas_v.div_euclid(16) * 16;
        (vlo, (haut_v + 1 - vlo + 15) / 16 * 16)
    } else {
        (0, 16)
    };
    let vr = 16 * NIVEAU_R;
    writeln!(defs, "#define SF_VLO ({vlo})\n#define SF_VH {vh}\n#define SF_VR {vr}\n#define SF_VN ({}L)", i64::from(vr) * i64::from(vr) * i64::from(vh))?;
    let mut tables = String::new();
    writeln!(tables, "__device__ const int BANDES[192] = {{{}}};", s.bands.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))?;
    let nt = s.temperatures.len().max(1);
    writeln!(tables, "#define NTB {}", s.temperatures.len())?;
    // Bits des temperatures de base (un initialiseur __device__ doit etre constant)
    writeln!(tables, "__device__ const unsigned TB[{nt}] = {{{}}};", s.temperatures.iter().map(|t| format!("{:#010x}u", t.0.to_bits())).collect::<Vec<_>>().join(","))?;
    writeln!(tables, "__device__ const int TG[{nt}] = {{{}}};", s.temperatures.iter().map(|t| i32::from(t.1).to_string()).collect::<Vec<_>>().join(","))?;
    // Declarations, conditions puis regles
    let mut code = String::new();
    for k in 0..s.conditions.len() {
        writeln!(code, "__device__ bool C_{k}(Ctx &c);")?;
    }
    for k in 0..s.rules.len() {
        writeln!(code, "__device__ int R_{k}(Ctx &c);")?;
    }
    for (k, cond) in s.conditions.iter().enumerate() {
        let corps = match cond {
            SC::Hole => "return c.depth <= 0;".to_string(),
            SC::Steep => "return c.gx <= -4 || c.gz >= 4;".to_string(),
            SC::Biomes(ids) => {
                let mots = ids.iter().map(|&i| i as usize / 64 + 1).max().unwrap_or(1);
                let mut m = vec![0u64; mots];
                for &i in ids {
                    m[i as usize / 64] |= 1 << (i % 64);
                }
                writeln!(tables, "__device__ const unsigned long long BIO_{k}[{mots}] = {{{}}};", m.iter().map(|v| format!("{v}ull")).collect::<Vec<_>>().join(","))?;
                format!("int b = sf_biome(c);\n  if (b < 0) {{ c.erreur = 1; return false; }}\n  return (b >> 6) < {mots} && ((BIO_{k}[b >> 6] >> (b & 63)) & 1ull);")
            }
            SC::Temperature => "int b = sf_biome(c);\n  if (b < 0 || b >= NTB) { c.erreur = 1; return false; }\n  return !(sf_temperature(__int_as_float((int)TB[b]), TG[b], c.x, c.y, c.z, SF_MER) >= 0.15f);".to_string(),
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
                "int pierre = {};\n  int surface = {};\n  int sec = {};\n  return pierre <= (int)(1u + (unsigned)({offset}) + (unsigned)surface + (unsigned)sec);",
                if *ceiling { "c.below" } else { "c.above" },
                if *add_surface { "c.depth" } else { "0" },
                if *secondary_range == 0 {
                    "0".to_string()
                } else {
                    format!("(int)(0.0 + ((c.secondary - -1.0) / (1.0 - -1.0)) * ((double)({secondary_range}) - 0.0))")
                }
            ),
            SC::Gradient { low, high, positional } => format!(
                "if (c.y <= {low}) return true;\n  if (c.y >= {high}) return false;\n  double p = 1.0 + (((double)c.y - (double)({low})) / ((double)({high}) - (double)({low}))) * (0.0 - 1.0);\n  unsigned long long lo, hi;\n  aq_at({}ull, {}ull, c.x, c.y, c.z, lo, hi);\n  return (double)aq_next_float(lo, hi) < p;",
                positional.0, positional.1
            ),
            SC::Not(i) => format!("return !C_{i}(c);"),
        };
        writeln!(code, "__device__ bool C_{k}(Ctx &c) {{\n  {corps}\n}}")?;
    }
    let (ore_lo, ore_hi) = s.ore_positional;
    for (k, r) in s.rules.iter().enumerate() {
        let corps = match r {
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
            SR::Condition { condition, rule } => format!("if (!C_{condition}(c)) return -1;\n  return R_{rule}(c);"),
            SR::Bandlands => format!(
                "float n = {pt_clay}((double)c.x, 0.0, (double)c.z);\n  int o = floor_sat((double)__fmul_rn(n, 4.0f) + 0.5);\n  int i = (int)((unsigned)c.y + (unsigned)o + 192u) % 192;\n  if (i < 0) {{ c.erreur = 1; return -1; }}\n  return BANDES[i];"
            ),
            SR::OreVein { blocks, raw_chance, density, richness, gap, positive_y: [bas, haut] } => {
                let vd = volumes.iter().position(|v| v == density).unwrap();
                let vr = volumes.iter().position(|v| v == richness).unwrap();
                // Hors de positive_y la densite n'est jamais positive : rien, sans tirage (comme le CPU)
                let (bas, haut) = ((*bas).max(s.min_y), (*haut).min(haut_monde));
                format!(
                    "if (c.y < {bas} || c.y > {haut}) return -1;\n  long iv = ((long)c.vz * SF_VR + c.vx) * SF_VH + (c.y - SF_VLO);\n  float d = c.VOL[{vd}L * SF_VN + iv];\n  if (d <= 0.0f) return -1;\n  unsigned long long lo, hi;\n  aq_at({ore_lo}ull, {ore_hi}ull, c.x, c.y, c.z, lo, hi);\n  if (aq_next_float(lo, hi) > d) return -1;\n  float ri = c.VOL[{vr}L * SF_VN + iv];\n  if (aq_next_float(lo, hi) < ri && {}(c.x, c.y, c.z) < 0.0f) return aq_next_float(lo, hi) < {} ? {} : {};\n  return {};",
                    ecarts[gap],
                    f32lit(*raw_chance),
                    blocks[1],
                    blocks[0],
                    blocks[2]
                )
            }
        };
        writeln!(code, "__device__ __noinline__ int R_{k}(Ctx &c) {{\n  {corps}\n}}")?;
    }
    let (rlo, rhi) = s.root_positional;
    let racine = s.root;
    write!(
        code,
        r#"
// La surface en trois noyaux. COLI, 10 entiers par colonne : otop, ptop (premier y libre apres le pilier),
// ftop (apres l'iceberg), depth, minimum, plo et phi (le pilier), gx, gz. META, trois plans de blocs :
// above (0 : la regle ne s'applique pas), below, water. GLACE : les blocs d'iceberg (etat + 1, 0 : rien).
#define SF_TOUS 0xffffffffu
__device__ __forceinline__ unsigned sf_jusqua(int l) {{ return l >= 31 ? SF_TOUS : ((2u << l) - 1u); }}
__device__ __forceinline__ int sf_genre_p(const signed char *col, int y, int plo, int phi) {{
  return (y >= plo && y <= phi) ? SF_SOLIDE : sf_genre(col, y);
}}

// 1. Une warp par colonne : premier y libre (vote de la warp), puis la ligne 0 fait les scalaires de la
// colonne et ses extensions (pilier des badlands erodes, iceberg).
extern "C" __global__ void ksf_tetes(const signed char *CODES, const unsigned short *B, const float *PL, unsigned *GLACE, int *COLI, double *COLD, int *ERR, int cx, int cz) {{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int lc = t >> 5, l = t & 31;
  if (lc >= 256) return;
  const signed char *col = CODES + (long)lc * SF_H;
  int otop = SF_MINY;
  for (int base = SF_H - 1; base >= 0; base -= 32) {{
    int iy = base - l;
    unsigned m = __ballot_sync(SF_TOUS, iy >= 0 && (col[iy] & 3) != 1);
    if (m) {{ otop = SF_MINY + (base - (__ffs(m) - 1)) + 1; break; }}
  }}
  if (l != 0) return;
  int lx = lc & 15, lz = lc >> 4;
  int x = cx * 16 + lx, z = cz * 16 + lz;
  int erreur = 0;
  unsigned long long lo, hi;
  aq_at({rlo}ull, {rhi}ull, x, 0, z, lo, hi);
  int depth = (int)(((double){pt_surf}((double)x, 0.0, (double)z) * 2.75 + 3.0) + aq_next_double(lo, hi) * 0.25);
  double secondary = (double){pt_sec}((double)x, 0.0, (double)z);
  int minimum = (int)((unsigned)floor_sat((double)floorf(PL[lc])) + (unsigned)depth - 8u);
  // Le pilier avant la descente de sa colonne, l'iceberg apres. Ils ne dependent que des codes (l'iceberg
  // n'ecrit que dans l'air et l'eau, jamais sur un bloc de la regle) et montent le premier y libre.
  int plo = 1, phi = 0, ptop = otop, ftop = otop;
#ifdef SF_EXT
  int bcol;
  {{
    int q[3];
    sf_quart(SF_ZOOM, x, otop, z, q);
    bcol = sf_biome_quart(B, cx, cz, SF_MINY >> 2, SF_H >> 2, q);
  }}
  if (bcol < 0) erreur = 1;
  if (bcol >= 0 && bcol == SF_ERODE) {{
    int haut = sf_pilier(col, x, z, otop);
    if (haut != SF_RIEN) {{
      plo = otop;
      phi = min(haut, SF_MINY + SF_H - 1);
      ptop = phi + 1;
      ftop = ptop;
    }}
  }}
  if (bcol >= 0 && (bcol == SF_GELE0 || bcol == SF_GELE1)) {{
    int plus = sf_iceberg(col, GLACE + (long)lc * SF_H, x, z, otop, minimum, bcol, {rlo}ull, {rhi}ull, &erreur);
    if (plus != SF_RIEN && plus + 1 > ftop) ftop = plus + 1;
  }}
#endif
  int *ci = COLI + 10 * lc;
  ci[0] = otop; ci[1] = ptop; ci[2] = ftop; ci[3] = depth; ci[4] = minimum; ci[5] = plo; ci[6] = phi;
  COLD[lc] = secondary;
  if (erreur) atomicOr(ERR, 1);
}}

// 2. Une warp par colonne : les pentes (colonnes traitees en x puis z : x - 1 et z - 1 ont deja leurs
// extensions, x + 1 et z + 1 pas encore, la colonne a son pilier mais pas son iceberg), puis la descente
// de build_columns par paquets de 32 blocs : below en montant, above et water en descendant.
extern "C" __global__ void ksf_descente(const signed char *CODES, int *COLI, int *META) {{
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
  // below = y - (le plus haut non-solide sous y) ; SF_MINY - 1 compte comme de l'air
  int dessous = SF_MINY - 1;
  for (int base = SF_MINY; base <= ptop; base += 32) {{
    int y = base + l;
    int k = sf_genre_p(col, y, plo, phi);
    unsigned ns = __ballot_sync(SF_TOUS, k != SF_SOLIDE);
    unsigned avant = l == 0 ? 0u : (ns & ((1u << l) - 1u));
    int prec = avant ? base + (31 - __clz(avant)) : dessous;
    if (k == SF_SOLIDE && y <= SF_MINY + SF_H - 1) BW[b0 + (y - SF_MINY)] = y - prec;
    if (ns) dessous = base + (31 - __clz(ns));
  }}
  // above = solides depuis le dernier air au-dessus ; water = premier fluide depuis cet air, + 1. La
  // descente commence a ptop avec above = 0 et water = MIN, comme apres un bloc d'air.
  int cC = 0, cA = 0, cW = -2147483647 - 1;
  for (int base = ptop; base >= SF_MINY; base -= 32) {{
    int y = base - l;
    int k = y < SF_MINY ? SF_AIR : sf_genre_p(col, y, plo, phi);
    unsigned ms = __ballot_sync(SF_TOUS, k == SF_SOLIDE), ma = __ballot_sync(SF_TOUS, k == SF_AIR), mf = __ballot_sync(SF_TOUS, k == SF_FLUIDE);
    unsigned le = sf_jusqua(l);
    int cs = cC + __popc(ms & le);
    unsigned a = ma & le;
    int la = a ? 31 - __clz(a) : -1;
    int ca = la >= 0 ? cC + __popc(ms & sf_jusqua(la)) : cA;
    int w;
    if (la >= 0) {{
      unsigned f = mf & le & ~sf_jusqua(la);
      w = f ? (base - (__ffs(f) - 1)) + 1 : (-2147483647 - 1);
    }} else if (cW != -2147483647 - 1) {{
      w = cW;
    }} else {{
      unsigned f = mf & le;
      w = f ? (base - (__ffs(f) - 1)) + 1 : (-2147483647 - 1);
    }}
    if (k == SF_SOLIDE && y <= SF_MINY + SF_H - 1) {{
      long i = b0 + (y - SF_MINY);
      A[i] = cs - ca;
      W[i] = w;
    }}
    cC = __shfl_sync(SF_TOUS, cs, 31);
    cA = __shfl_sync(SF_TOUS, ca, 31);
    cW = __shfl_sync(SF_TOUS, w, 31);
  }}
}}

// 3. Un thread par bloc : la regle la ou elle s'applique, puis l'etat final. Bit 30 : la surface ecrit un
// etat a fluide ; bit 31 : mise a jour de fluide planifiee par le remplissage (fill_noise_codes : code + 4
// sur un etat a fluide).
extern "C" __global__ void ksf_blocs(const signed char *CODES, const unsigned short *B, const float *VOL, const unsigned *GLACE, const int *META, const int *COLI, const double *COLD, unsigned *FIN, int *ERR, int cx, int cz, int ox, int oz) {{
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long n = 256L * SF_H;
  if (i >= n) return;
  int lc = (int)(i / SF_H), iy = (int)(i % SF_H), y = SF_MINY + iy;
  const int *ci = COLI + 10 * lc;
  unsigned g = GLACE[i];
  int o = g ? (int)(g - 1u) : ((y >= ci[5] && y <= ci[6]) ? SF_E_DEFAUT : -1);
  int above = META[i];
  if (above > 0) {{
    Ctx c;
    c.x = cx * 16 + (lc & 15); c.z = cz * 16 + (lc >> 4); c.y = y;
    c.depth = ci[3]; c.minimum = ci[4]; c.gx = ci[7]; c.gz = ci[8]; c.secondary = COLD[lc];
    c.above = above; c.below = META[n + i]; c.water = META[2 * n + i];
    c.biome = -2; c.bloc = (int)i; c.B = B; c.cx = cx; c.cz = cz; c.VOL = VOL; c.erreur = 0;
    c.vx = ox + (lc & 15); c.vz = oz + (lc >> 4);
    int r = R_{racine}(c);
    if (r >= 0) o = sf_fluide(r) ? (r | 0x40000000) : r;
    if (c.erreur) atomicOr(ERR, 1);
  }}
  int k = CODES[i];
  int e = o >= 0 ? o : ((k & 3) == 0 ? SF_E_DEFAUT : ((k & 3) == 1 ? SF_E_AIR : ((k & 3) == 2 ? SF_E_FLUIDE : SF_E_LAVE)));
  FIN[i] = (unsigned)e | (((k & 4) != 0 && (k & 3) >= 2) ? 0x80000000u : 0u);
}}
"#
    )?;
    let src = format!("{}{BASE}\n{OPS}\n{}\n{defs}\n{tables}\n{ALEATOIRE}\n{SURFACE}\n{code}", tables_cuda(&g)?, g.defs);
    let mut progs = Vec::new();
    for id in volumes {
        progs.push(generer(c, id, graine, [vr, vh, vr])?);
    }
    // Surface preliminaire : en volume sur le plan y = 0 (MaterialDensitySources), pour toute une region de
    // NIVEAU_R x NIVEAU_R chunks : memes coins (multiples de 16) et meme interpolation que le plan d'un chunk
    let niveau = generer(c, s.surface_level, graine, [vr, 1, vr])?;
    Ok(ProgrammeSurface { src, volumes: progs, niveau, filons_y: (vlo, vh) })
}
