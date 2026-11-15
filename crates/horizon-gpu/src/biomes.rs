//! Biomes sur GPU (`HORIZON_GPU_BIOMES`) : voir BIOMES.md.
//!
//! Le GPU calcule les biomes d'une region de R x R chunks d'un coup : `kbi_climat` echantillonne le
//! climat colonne par colonne, `kbi_recherche` fait la recherche de chaque quart dans l'arbre du climat
//! sans point de depart, puis `kbi_ordre` applique, chunk par chunk et dans l'ordre du remplissage, le
//! resultat precedent de la recherche (`ClimateSearch::last`). Le CPU construit chaque `BiomeChunk`
//! depuis la region.
use super::{Contexte, contexte, flux_prioritaire, ptx, traduction};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use horizon_core::density::graph::{DensityCompiler, FunctionId};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// L'arbre de recherche du climat (`ClimateIndex::tree`) et l'identifiant du biome de chaque entree.
pub struct BiomeTree<'a> {
    /// Par noeud : les six minimums, les six maximums, la distance du decalage au carre.
    pub spaces: &'a [[i64; 13]],
    /// Par noeud : premier enfant et nombre d'enfants, ou entree et 0 pour une feuille.
    pub links: &'a [(u32, u32)],
    /// L'identifiant de chaque entree ; `u32::MAX` si son nom n'en a pas (le chunk qui la choisit
    /// repasse par le CPU, qui leve l'erreur).
    pub entry_ids: &'a [u32],
}

/// Drapeaux de `biomes.cu` : pile du parcours trop courte, recherche sans resultat (la region repasse
/// par le CPU) ; entree sans identifiant (seuls les chunks concernes repassent par le CPU).
const E_PILE: i32 = 1;
const E_VIDE: i32 = 2;

/// Threads d'un bloc de `kbi_ordre` (BI_TUILE de biomes.cu).
const TUILE: u32 = 128;

struct Ressources {
    st: Arc<CudaStream>,
    /// Les six champs de chaque quart, par quart Y puis colonne
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
    erreurs: AtomicU64,
    chunks: AtomicU64,
    replis: AtomicU64,
    compares: AtomicU64,
    ecarts: AtomicU64,
}

pub struct BiomeGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    k_climat: CudaFunction,
    k_recherche: CudaFunction,
    k_ordre: CudaFunction,
    e: CudaSlice<i64>,
    l: CudaSlice<u32>,
    ids: CudaSlice<u32>,
    /// Cote d'une region en chunks
    r: i32,
    /// Premier quart Y et nombre de quarts Y d'un chunk
    qy0: i32,
    nqy: i32,
    pool: Mutex<Vec<Ressources>>,
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
        let n = s.regions.load(Relaxed);
        if n > 0 {
            eprintln!(
                "GPU biomes : {n} regions de {r} x {r} chunks, {} us par region (max {}) ; {} chunks du GPU, {} par le CPU ; {} regions en erreur ; mode test : {} chunks compares, {} ecarts",
                s.ns.load(Relaxed) / n / 1000,
                s.ns_max.load(Relaxed) / 1000,
                s.chunks.load(Relaxed),
                s.replis.load(Relaxed),
                s.erreurs.load(Relaxed),
                s.compares.load(Relaxed),
                s.ecarts.load(Relaxed),
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
    /// `roots` : les six racines du climat du routeur (temperature, vegetation, continents, erosion,
    /// depth, ridges) ; `min_y` et `height` : ceux des `BiomeChunk` ; `region` : cote d'une region.
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
            return Err("arbre du climat".into());
        }
        for &(a, k) in tree.links {
            let ok = if k == 0 {
                (a as usize) < ne
            } else {
                a as usize + k as usize <= n
            };
            if !ok {
                return Err("liens de l'arbre du climat".into());
            }
        }
        if min_y % 16 != 0 || height <= 0 || height % 16 != 0 || !(1..=32).contains(&region) {
            return Err("hauteur des chunks ou cote des regions".into());
        }
        let src = traduction::generer_biomes(compiler, roots, seed).map_err(|e| e.to_string())?;
        if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP_BIOMES") {
            let _ = std::fs::write(f, &src);
        }
        let ctx = contexte()?;
        let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
        let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
        let module = ctx.load_module(ptx(&src, arch)?).ctx("module des biomes")?;
        let k_climat = module.load_function("kbi_climat").ctx("kbi_climat")?;
        let k_recherche = module.load_function("kbi_recherche").ctx("kbi_recherche")?;
        let k_ordre = module.load_function("kbi_ordre").ctx("kbi_ordre")?;
        let st = ctx.new_stream().ctx("flux CUDA")?;
        let plat: Vec<i64> = tree.spaces.iter().flatten().copied().collect();
        let e = st.clone_htod(&plat).ctx("arbre du climat")?;
        let liens: Vec<u32> = tree.links.iter().flat_map(|&(a, k)| [a, k]).collect();
        let l = st.clone_htod(&liens).ctx("liens de l'arbre")?;
        let ids = st.clone_htod(tree.entry_ids).ctx("identifiants des biomes")?;
        st.synchronize().ctx("arbre du climat")?;
        Ok(Self {
            ctx,
            _module: module,
            k_climat,
            k_recherche,
            k_ordre,
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
    /// Cote d'une region en chunks.
    pub fn region_side(&self) -> i32 {
        self.r
    }
    /// Quarts d'un chunk : 16 par quart Y.
    pub fn quarts_per_chunk(&self) -> usize {
        16 * self.nqy as usize
    }
    /// Un chunk construit depuis le GPU, ou repasse par le CPU (fusion avec un ancien monde, entree
    /// sans identifiant, region en erreur).
    pub fn note_chunk(&self, gpu: bool) {
        if gpu {
            self.stats.chunks.fetch_add(1, Relaxed);
        } else {
            self.stats.replis.fetch_add(1, Relaxed);
        }
    }
    /// Mode test : un chunk compare au CPU ; rend le nombre de chunks compares.
    pub fn note_test(&self, ecart: bool) -> u64 {
        if ecart {
            self.stats.ecarts.fetch_add(1, Relaxed);
        }
        self.stats.compares.fetch_add(1, Relaxed) + 1
    }
    /// Ecarts au CPU vus en mode test.
    pub fn mismatches(&self) -> u64 {
        self.stats.ecarts.load(Relaxed)
    }
    fn ressource(&self, n: usize) -> Result<Ressources, String> {
        if let Some(r) = self.pool.lock().unwrap().pop() {
            return Ok(r);
        }
        let st = flux_prioritaire(&self.ctx)?;
        Ok(Ressources {
            qf: st.alloc_zeros::<f32>(6 * n).ctx("champs du climat")?,
            q: st.alloc_zeros::<i64>(6 * n).ctx("valeurs du climat")?,
            best: st.alloc_zeros::<i32>(n).ctx("feuilles")?,
            fmin: st.alloc_zeros::<i64>(n).ctx("distances")?,
            out: st.alloc_zeros::<u32>(n).ctx("biomes")?,
            err: st.alloc_zeros::<i32>(1).ctx("drapeaux")?,
            h_out: unsafe { self.ctx.alloc_pinned_with_flags::<u32>(n, 0) }.ctx("tampon hote")?,
            h_err: unsafe { self.ctx.alloc_pinned_with_flags::<i32>(1, 0) }.ctx("tampon hote")?,
            st,
        })
    }
    /// Les biomes des R x R chunks a partir du chunk (`cx0`, `cz0`) : pour chaque chunk (z * R + x),
    /// l'identifiant de ses quarts dans l'ordre de `BiomeChunk::fill` (`u32::MAX` : entree sans
    /// identifiant). Bloquant.
    pub fn region(&self, cx0: i32, cz0: i32) -> Result<Vec<u32>, String> {
        let t = Instant::now();
        let n = (self.r * self.r) as usize * self.quarts_per_chunk();
        let mut res = self.ressource(n)?;
        let resultat = (|| -> Result<Vec<u32>, String> {
            let st = res.st.clone();
            st.memset_zeros(&mut res.err).ctx("drapeaux")?;
            let (r, qy0, nqy) = (self.r, self.qy0, self.nqy);
            let mut b = st.launch_builder(&self.k_climat);
            b.arg(&cx0).arg(&cz0).arg(&r).arg(&qy0).arg(&nqy).arg(&mut res.qf);
            unsafe { b.launch(config((r * r * 16) as usize, 32)) }.ctx("climat")?;
            let mut b = st.launch_builder(&self.k_recherche);
            b.arg(&r).arg(&nqy).arg(&res.qf).arg(&mut res.q).arg(&mut res.best).arg(&mut res.fmin);
            b.arg(&self.e).arg(&self.l).arg(&mut res.err);
            unsafe { b.launch(config(n, 128)) }.ctx("recherche")?;
            let mut b = st.launch_builder(&self.k_ordre);
            b.arg(&r).arg(&nqy).arg(&res.q).arg(&res.best).arg(&res.fmin).arg(&self.e).arg(&self.l);
            b.arg(&self.ids).arg(&mut res.out).arg(&mut res.err);
            let blocs = LaunchConfig {
                grid_dim: ((r * r) as u32, 1, 1),
                block_dim: (TUILE, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(blocs) }.ctx("ordre des recherches")?;
            st.memcpy_dtoh(&res.out, &mut res.h_out.as_mut_slice().ctx("tampon hote")?[..n])
                .ctx("retour des biomes")?;
            st.memcpy_dtoh(&res.err, res.h_err.as_mut_slice().ctx("tampon hote")?)
                .ctx("retour des drapeaux")?;
            st.synchronize().ctx("attente des biomes")?;
            let err = res.h_err.as_slice().ctx("tampon hote")?[0];
            if err & (E_PILE | E_VIDE) != 0 {
                return Err(format!("recherche du climat (drapeaux {err})"));
            }
            Ok(res.h_out.as_slice().ctx("tampon hote")?[..n].to_vec())
        })();
        let ns = t.elapsed().as_nanos() as u64;
        self.stats.regions.fetch_add(1, Relaxed);
        self.stats.ns.fetch_add(ns, Relaxed);
        self.stats.ns_max.fetch_max(ns, Relaxed);
        match resultat {
            Ok(v) => {
                self.pool.lock().unwrap().push(res);
                Ok(v)
            }
            Err(e) => {
                self.stats.erreurs.fetch_add(1, Relaxed);
                let _ = res.st.synchronize();
                Err(e)
            }
        }
    }
}
