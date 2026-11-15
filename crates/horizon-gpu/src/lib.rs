//! final_density sur GPU (CUDA), identique au bit pres au sampler CPU de Horizon.
//!
//! Le GPU calcule d'un coup une region de R x R chunks (8 x 8 par defaut) et la garde en memoire
//! GPU ; chaque chunk demande ensuite par un worker est copie (393 Ko en 16 x 384 x 16) dans le
//! tampon de `fill_noise`, qui fait le reste sur le CPU comme avant (aquiferes, blocs, heightmaps).
//!
//! Active par la variable d'environnement `HORIZON_GPU_DENSITY` :
//! - `1` : compilation en arriere-plan, le CPU sert les chunks en attendant ;
//! - `wait` : compilation avant le premier chunk (tests de parite).
//!
//! Reglages : `HORIZON_GPU_SYNC` (attente : block par defaut, spin, yield, auto),
//! `HORIZON_GPU_PRIORITY=0` (sans priorite pour les flux du chemin critique),
//! `HORIZON_GPU_REGION` (cote d'une region en chunks, 8), `HORIZON_GPU_REGIONS_MAX`
//! (regions gardees en memoire GPU, 128), `HORIZON_GPU_CACHE` (dossier du PTX compile).
//! Au demarrage, un auto-test compare trois chunks au CPU ; a la moindre difference ou erreur CUDA,
//! le GPU est desactive et tout repasse par le CPU.
mod biomes;
mod grottes;
mod lumiere;
mod traduction;

pub use biomes::{BiomeGpu, BiomeTree};
pub use lumiere::{LightGpu, LightInput, LightJob, LightSection};

pub use grottes::{CAVE_CANYON, CAVE_MASK_WORDS, CAVE_ROOM, CAVE_TUNNEL, CaveGpu, CaveItem, CaveJob, CaveTarget};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use horizon_core::density::graph::{DensityCompiler, FunctionId};
use horizon_core::density::{Beardifier, DensitySampler, SamplerContext, TerrainAdjustment};
use horizon_core::noise::DensityVolume;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

trait Contexte<T> {
    fn ctx(self, quoi: &str) -> Result<T, String>;
}
impl<T, E: std::fmt::Debug> Contexte<T> for Result<T, E> {
    fn ctx(self, quoi: &str) -> Result<T, String> {
        self.map_err(|e| format!("{quoi} : {e:?}"))
    }
}

fn env_ou(nom: &str, defaut: i32) -> i32 {
    std::env::var(nom)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(defaut)
}

/// Le contexte CUDA du processus, partage par toutes les dimensions.
fn contexte() -> Result<Arc<CudaContext>, String> {
    static CTX: OnceLock<Result<Arc<CudaContext>, String>> = OnceLock::new();
    CTX.get_or_init(|| {
        let ctx = CudaContext::new(0).ctx("contexte CUDA")?;
        // Attente de la fin d'un calcul : HORIZON_GPU_SYNC = block (par defaut : le thread dort), spin,
        // yield ou auto (heuristique CUDA, souvent spin : le coeur tourne a vide pendant l'attente).
        // Avec la lumiere sur GPU, le CPU n'est plus le goulot de /pregen : dormir rend 4 a 6 % du CPU
        // par chunk sans changer le debit (GPU_ATTENTE.md).
        use cudarc::driver::sys::CUctx_flags::*;
        let attente = match std::env::var("HORIZON_GPU_SYNC").as_deref() {
            Ok("spin") => Some(CU_CTX_SCHED_SPIN),
            Ok("yield") => Some(CU_CTX_SCHED_YIELD),
            Ok("auto") => None,
            _ => Some(CU_CTX_SCHED_BLOCKING_SYNC),
        };
        if let Some(f) = attente {
            ctx.set_flags(f).ctx("mode d'attente")?;
        }
        // Synchronisation geree ici : chaque calcul et chaque copie attend la fin de son flux.
        unsafe { ctx.disable_event_tracking() };
        Ok(ctx)
    })
    .clone()
}

/// Mots u32 d'une section dans la sortie de `SurfaceGpu::enable_sections` (sections.cu, SC_PAS) :
/// en-tete de 5 mots, 128 etats, 128 compteurs, 1024 mots de donnees.
pub const SECTION_WORDS: usize = 5 + 2 * 128 + 1024;
/// Marques de post-traitement par liste au plus dans cette sortie (au-dela : etats bruts, CPU).
pub const SECTION_MARKS: usize = 2048;

/// Un flux du chemin critique (un worker attend son resultat : remplissage, surface, region) : la plus
/// haute priorite du GPU, pour passer devant les lots asynchrones (lumiere, grottes, minerais), crees a
/// la priorite par defaut. `HORIZON_GPU_PRIORITY=0` les met tous a la meme.
fn flux_prioritaire(ctx: &Arc<CudaContext>) -> Result<Arc<CudaStream>, String> {
    static PRIORITE: OnceLock<i32> = OnceLock::new();
    let p = *PRIORITE.get_or_init(|| {
        if std::env::var("HORIZON_GPU_PRIORITY").as_deref() == Ok("0") || ctx.bind_to_thread().is_err() {
            return 0;
        }
        cudarc::driver::result::stream::get_priority_range().map_or(0, |(_, haute)| haute)
    });
    ctx.new_stream_with_priority(p).ctx("flux CUDA")
}

/// Les aquiferes d'une dimension, pour decider aussi chaque bloc sur GPU.
#[derive(Clone, Copy, Debug)]
pub struct AquiferSetup {
    /// Racines dans l'ordre de AquiferField::ALL (barriere, inondation, etalement, lave, exclusion,
    /// niveau de surface).
    pub roots: [FunctionId; 6],
    /// Etat (lo, hi) de la fabrique positionnelle des aquiferes (Xoroshiro).
    pub positional: (u64, u64),
    pub sea_level: i32,
}

/// final_density d'une dimension sur GPU.
pub struct GpuFinalDensity {
    partage: Arc<Partage>,
}

struct Partage {
    nom: String,
    moteur: OnceLock<Option<Moteur>>,
}

impl std::fmt::Debug for GpuFinalDensity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let etat = match self.partage.moteur.get() {
            None => "en preparation",
            Some(None) => "desactive",
            Some(Some(m)) if m.hors_service.load(Relaxed) => "hors service",
            Some(Some(_)) => "pret",
        };
        write!(f, "GpuFinalDensity({}, {etat})", self.partage.nom)
    }
}

impl GpuFinalDensity {
    /// `None` si `HORIZON_GPU_DENSITY` est absent ou vaut 0, si la source aleatoire est legacy, ou si
    /// un noeud du graphe n'a pas de traduction CUDA. `cpu` est le sampler de reference de l'auto-test.
    pub fn new(
        compiler: &DensityCompiler,
        root: FunctionId,
        seed: i64,
        legacy: bool,
        cpu: Arc<dyn DensitySampler>,
        min_y: i32,
        height: i32,
        nom: &str,
        aquifer: Option<AquiferSetup>,
    ) -> Option<Self> {
        let mode = std::env::var("HORIZON_GPU_DENSITY").ok()?;
        if mode.is_empty() || mode == "0" {
            return None;
        }
        if legacy {
            log::info!("GPU final_density ({nom}) : source aleatoire legacy, reste sur CPU");
            return None;
        }
        let r = env_ou("HORIZON_GPU_REGION", 8).clamp(1, 32);
        let capacite = env_ou("HORIZON_GPU_REGIONS_MAX", 128).max(1) as usize;
        let t = Instant::now();
        let prog = match traduction::generer(compiler, root, seed, [16 * r, height, 16 * r]) {
            Ok(p) => p,
            Err(e) => {
                log::info!(
                    "GPU final_density ({nom}) : pas de traduction CUDA ({e}), reste sur CPU"
                );
                return None;
            }
        };
        log::info!(
            "GPU final_density ({nom}) : {} lignes de CUDA generees en {:.0} ms, regions de {r}x{r} chunks ({:.0} Mo GPU chacune)",
            prog.lignes,
            t.elapsed().as_secs_f64() * 1e3,
            prog.taille as f64 * 4e-6
        );
        if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP") {
            let _ = std::fs::write(f, &prog.src);
        }
        // Aquiferes : HORIZON_GPU_AQUIFERS=0 les laisse sur CPU
        let aquifere = aquifer
            .filter(|_| std::env::var("HORIZON_GPU_AQUIFERS").map_or(true, |v| v != "0"))
            .and_then(|a| match traduction::generer_aquifere(compiler, a.roots, seed) {
                Ok(src) => Some((a, src)),
                Err(e) => {
                    log::info!("GPU final_density ({nom}) : aquiferes sans traduction CUDA ({e}), restent sur CPU");
                    None
                }
            });
        let partage = Arc::new(Partage {
            nom: nom.to_owned(),
            moteur: OnceLock::new(),
        });
        let p = partage.clone();
        let travail = move || {
            let t = Instant::now();
            let m = Moteur::demarrer(prog, aquifere, r, capacite, min_y, height)
                .and_then(|m| m.auto_test(&*cpu).map(|()| m));
            match m {
                Ok(m) => {
                    let msg = format!(
                        "GPU final_density ({}) : pret en {:.1} s, auto-test identique au CPU{}",
                        p.nom,
                        t.elapsed().as_secs_f64(),
                        if m.aq.is_some() { ", aquiferes sur GPU" } else { "" }
                    );
                    log::info!("{msg}");
                    eprintln!("{msg}");
                    let _ = p.moteur.set(Some(m));
                }
                Err(e) => {
                    log::error!(
                        "GPU final_density ({}) : desactive, tout reste sur CPU : {e}",
                        p.nom
                    );
                    eprintln!(
                        "GPU final_density ({}) : desactive, tout reste sur CPU : {e}",
                        p.nom
                    );
                    let _ = p.moteur.set(None);
                }
            }
        };
        if mode == "wait" {
            travail();
        } else if let Err(e) = std::thread::Builder::new()
            .name("horizon-gpu".into())
            .spawn(travail)
        {
            log::error!("GPU final_density ({nom}) : thread de compilation : {e}");
            return None;
        }
        Some(Self { partage })
    }

    /// Ecrit la densite finale du chunk dans `out` (16 x hauteur x 16, ordre de `fill_noise`).
    /// `false` : GPU pas encore pret, desactive ou en erreur ; le CPU doit calculer ce chunk.
    pub fn fill_chunk(&self, chunk_x: i32, chunk_z: i32, out: &mut [f32]) -> bool {
        match self.partage.moteur.get() {
            Some(Some(m)) => m.chunk(chunk_x, chunk_z, out),
            _ => false,
        }
    }

    /// Codes de chaque bloc du chunk (ordre de `fill_noise`) : 0 densite > 0, 1 air, 2 fluide par defaut,
    /// 3 lave, 8 bloc par defaut decide par la pression de l'aquifere, +4 mise a jour de fluide planifiee.
    /// Renvoie le skip_sampling_above_y du chunk, ou `None` (GPU absent, sans aquiferes ou en erreur).
    /// `grottes` : le lot des carvers du chunk, dont l'aquifere GPU calcule alors aussi les cellules.
    pub fn chunk_blocks(&self, chunk_x: i32, chunk_z: i32, out: &mut [i8], grottes: Option<&mut CaveJob>) -> Option<i32> {
        match self.partage.moteur.get() {
            Some(Some(m)) if m.aq.is_some() => m.chunk_codes(chunk_x, chunk_z, out, None, grottes),
            _ => None,
        }
    }

    /// Comme `chunk_blocks`, pour un chunk dont le terrain s'adapte a des structures : le beardifier est
    /// ajoute a la densite finale sur GPU avant la decision d'aquifere. `None` pour les noyaux de terrain
    /// des bibliotheques de structures (mods), qui restent sur CPU.
    pub fn chunk_blocks_beard(&self, chunk_x: i32, chunk_z: i32, beard: &Beardifier, out: &mut [i8], grottes: Option<&mut CaveJob>) -> Option<i32> {
        match self.partage.moteur.get() {
            Some(Some(m)) if m.aq.is_some() => m.chunk_codes_beard(chunk_x, chunk_z, beard, out, None, grottes),
            _ => None,
        }
    }

    /// `chunk_blocks` (`chunk_blocks_beard` avec `beard`), puis la surface du chunk sur le meme flux : ses
    /// codes ne repassent pas par l'hote. Renvoie le saut du chunk et l'issue de la surface : `Some(true)`
    /// etats finaux dans `etats` (voir `SurfaceGpu::chunk_states`), `Some(false)` une condition non
    /// evaluable, `None` surface non calculee (le chunk passe alors par la surface CPU).
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_blocks_surface(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        beard: Option<&Beardifier>,
        out: &mut [i8],
        sg: &SurfaceGpu,
        biomes: &[u16],
        etats: &mut [u32],
        mut grottes: Option<&mut CaveJob>,
        mut sections: Option<&mut Vec<u32>>,
    ) -> (Option<i32>, Option<bool>) {
        let avec_sections = sections.is_some();
        let Some(Some(m)) = self.partage.moteur.get() else {
            return (None, None);
        };
        if m.aq.is_none() {
            return (None, None);
        }
        let simple = |out: &mut [i8], grottes: Option<&mut CaveJob>| match beard {
            Some(b) => m.chunk_codes_beard(chunk_x, chunk_z, b, out, None, grottes),
            None => m.chunk_codes(chunk_x, chunk_z, out, None, grottes),
        };
        if etats.len() != out.len() || out.len() != 256 * sg.height as usize || biomes.len() != sg.biome_count() {
            return (simple(out, grottes), None);
        }
        let Ok(mut res) = sg.ressource() else {
            return (simple(out, grottes), None);
        };
        let t = Instant::now();
        let mut issue: Option<Result<Arc<RegionSurface>, String>> = None;
        let mut suite = |st: &Arc<CudaStream>, cd: &CudaSlice<i8>| {
            issue = Some(sg.enfiler(&mut res, st, Some(cd), chunk_x, chunk_z, biomes, avec_sections));
        };
        let saut = match beard {
            Some(b) => m.chunk_codes_beard(chunk_x, chunk_z, b, out, Some(&mut suite), grottes.take()),
            None => m.chunk_codes(chunk_x, chunk_z, out, Some(&mut suite), grottes.take()),
        };
        // Le flux des codes a ete attendu par chunk_codes : les etats sont arrives
        let surface = match (saut, issue) {
            (Some(_), Some(Ok(_region))) => {
                let lu = sg.lire(&mut res, etats, sections.take());
                sg.rendre(res, lu.is_ok(), t);
                lu.ok()
            }
            (_, Some(Err(e))) => {
                static SIGNALEE: OnceLock<()> = OnceLock::new();
                if SIGNALEE.set(()).is_ok() {
                    log::error!("GPU surface : chunk {chunk_x} {chunk_z} : {e}");
                }
                None
            }
            _ => None,
        };
        (saut, surface)
    }

    /// Un chunk dont le CPU a recalcule les blocs : la limite d'air de son aquifere differait du GPU.
    pub fn note_aquifer_mismatch(&self) {
        if let Some(Some(m)) = self.partage.moteur.get() {
            m.stats.ecarts.fetch_add(1, Relaxed);
        }
    }

    /// Un chunk calcule sur CPU parce que son terrain depend de structures ou de vieux chunks.
    pub fn note_cpu_fallback(&self) {
        if let Some(Some(m)) = self.partage.moteur.get() {
            m.stats.replis.fetch_add(1, Relaxed);
        }
    }
}

impl Drop for Partage {
    fn drop(&mut self) {
        if let Some(Some(m)) = self.moteur.get()
            && m.stats.chunks.load(Relaxed) > 0
        {
            eprintln!("{}", m.bilan());
        }
    }
}

/// Ce que le GPU garde d'une region : densite finale, et codes des blocs si les aquiferes y sont.
struct DonneesRegion {
    mem: CudaSlice<f32>,
    codes: Option<CudaSlice<i8>>,
    /// Positions et statuts des cellules d'aquifere, pour refaire les chunks a structures
    cellules: Option<Cellules>,
    /// skip_sampling_above_y de chaque chunk de la region (cz * r + cx)
    sauts: Vec<i32>,
}

struct Cellules {
    p: CudaSlice<i32>,
    st: CudaSlice<i32>,
    /// cx0, cy0, cz0, ncx, ncy, ncz
    grille: [i32; 6],
}

#[derive(Default)]
struct Region {
    donnees: OnceLock<Option<DonneesRegion>>,
    servis: AtomicU32,
}

struct Cache {
    carte: HashMap<(i32, i32), (Arc<Region>, u64)>,
    horloge: u64,
    capacite: usize,
}

/// Flux et tampons d'un worker pour la copie d'un chunk, rendus au pool apres usage.
struct Ressources {
    stream: Arc<CudaStream>,
    buf: CudaSlice<f32>,
    hote: PinnedHostSlice<f32>,
    buf_codes: CudaSlice<i8>,
    hote_codes: PinnedHostSlice<i8>,
}

#[derive(Default)]
struct Stats {
    chunks: AtomicU64,
    ns_copie: AtomicU64,
    regions: AtomicU64,
    ns_regions: AtomicU64,
    evictions: AtomicU64,
    replis: AtomicU64,
    /// Chunks remplis a partir des codes d'aquiferes du GPU
    chunks_aq: AtomicU64,
    /// Chunks dont la limite d'air de l'aquifere CPU differait du GPU
    ecarts: AtomicU64,
    /// Chunks a structures (beardifier) faits sur GPU
    chunks_beard: AtomicU64,
}

/// Les noyaux des aquiferes et leurs constantes.
struct Aquiferes {
    _module: Arc<cudarc::driver::CudaModule>,
    surface: CudaFunction,
    cellules: CudaFunction,
    saut: CudaFunction,
    blocs: CudaFunction,
    chunk: CudaFunction,
    beard: CudaFunction,
    /// Substance des cellules creusees (grottes)
    grottes: CudaFunction,
    /// Noyau du beard de vanilla (24 x 24 x 24)
    noyau: CudaSlice<f32>,
    positionnel: (u64, u64),
    mer: i32,
}

type Lancement = (CudaFunction, usize, usize);

struct Moteur {
    ctx: Arc<CudaContext>,
    /// Tables d'axes : (noyau, grille, threads)
    f_ax: Vec<Lancement>,
    /// Grilles dans l'ordre de lancement : (grille, passe 2D, noyau 3D, points)
    f_gr: Vec<(usize, Option<(CudaFunction, usize)>, CudaFunction, usize)>,
    f_chunk: CudaFunction,
    aq: Option<Aquiferes>,
    pas_grilles: Vec<[i32; 3]>,
    taille: usize,
    r: i32,
    min_y: i32,
    n_chunk: usize,
    cache: Mutex<Cache>,
    pool: Mutex<Vec<Ressources>>,
    stats: Stats,
    hors_service: AtomicBool,
}

fn config(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// FNV-1a 64 bits : nom stable du PTX en cache.
fn empreinte(octets: &[u8]) -> u64 {
    octets.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// PTX du programme, depuis le cache disque si possible (NVRTC prend environ 12 s).
fn ptx(src: &str, arch: &'static str) -> Result<Ptx, String> {
    let dossier = std::env::var_os("HORIZON_GPU_CACHE")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME").map(|d| PathBuf::from(d).join("horizon-gpu"))
        })
        .or_else(|| {
            std::env::var_os("HOME").map(|d| PathBuf::from(d).join(".cache").join("horizon-gpu"))
        });
    let cle = empreinte(format!("{arch}\n{src}").as_bytes());
    let fichier = dossier
        .as_ref()
        .map(|d| d.join(format!("final_density-{cle:016x}.ptx")));
    if let Some(f) = &fichier
        && let Ok(texte) = std::fs::read_to_string(f)
    {
        return Ok(Ptx::from_src(texte));
    }
    let options = CompileOptions {
        arch: Some(arch),
        options: vec![
            "--fmad=false".into(),
            "--ftz=false".into(),
            "--prec-div=true".into(),
            "--prec-sqrt=true".into(),
        ],
        ..Default::default()
    };
    let p = compile_ptx_with_opts(src, options).ctx("compilation NVRTC")?;
    if let (Some(d), Some(f)) = (&dossier, &fichier) {
        let _ = std::fs::create_dir_all(d).and_then(|()| std::fs::write(f, p.to_src()));
    }
    Ok(p)
}

impl Moteur {
    fn demarrer(
        prog: traduction::Programme,
        aquifere: Option<(AquiferSetup, String)>,
        r: i32,
        capacite: usize,
        min_y: i32,
        hauteur: i32,
    ) -> Result<Self, String> {
        let ctx = contexte()?;
        let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
        let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
        let module = ctx
            .load_module(ptx(&prog.src, arch)?)
            .ctx("chargement du module")?;
        let f = |nom: &str| module.load_function(nom).ctx(nom);
        let mut f_ax = Vec::new();
        for &(k, gr, nb) in &prog.axes {
            f_ax.push((f(&format!("kax_{k}"))?, gr, nb));
        }
        let mut f_gr = Vec::new();
        for k in (1..prog.grilles.len()).chain(std::iter::once(0)) {
            let gr = prog.grilles[k];
            let k2d = if prog.bords[k] > 0 {
                Some((f(&format!("k2d_{k}"))?, gr.colonnes() * prog.bords[k]))
            } else {
                None
            };
            f_gr.push((k, k2d, f(&format!("k3d_{k}"))?, gr.n()));
        }
        let aq = match aquifere {
            Some((a, src)) => {
                let m = ctx.load_module(ptx(&src, arch)?).ctx("module des aquiferes")?;
                let g = |nom: &str| m.load_function(nom).ctx(nom);
                let st = ctx.new_stream().ctx("flux CUDA")?;
                let noyau = st
                    .clone_htod(&horizon_core::density::beard_kernel()[..])
                    .ctx("noyau du beard")?;
                st.synchronize().ctx("noyau du beard")?;
                Some(Aquiferes {
                    beard: g("kbeard_chunk")?,
                    grottes: g("kgr_substance")?,
                    noyau,
                    surface: g("kaq_surface")?,
                    cellules: g("kaq_cellules")?,
                    saut: g("kaq_saut")?,
                    blocs: g("kaq_blocs_region")?,
                    chunk: g("kaq_chunk")?,
                    _module: m,
                    positionnel: a.positional,
                    mer: a.sea_level,
                })
            }
            None => None,
        };
        Ok(Self {
            f_chunk: f("kchunk")?,
            aq,
            ctx,
            f_ax,
            f_gr,
            pas_grilles: prog.grilles.iter().map(|g| g.pas).collect(),
            taille: prog.taille,
            r,
            min_y,
            n_chunk: 256 * hauteur as usize,
            cache: Mutex::new(Cache {
                carte: HashMap::new(),
                horloge: 0,
                capacite,
            }),
            pool: Mutex::new(Vec::new()),
            stats: Stats::default(),
            hors_service: AtomicBool::new(false),
        })
    }

    /// Trois chunks (dont des coordonnees negatives) compares bit a bit au sampler CPU.
    fn auto_test(&self, cpu: &dyn DensitySampler) -> Result<(), String> {
        let h = (self.n_chunk / 256) as i32;
        for (cx, cz) in [(0, 0), (self.r - 1, 1), (-1, -1)] {
            let mut gpu = vec![0.0f32; self.n_chunk];
            let mut reference = vec![0.0f32; self.n_chunk];
            self.chunk_interne(cx, cz, &mut gpu)?;
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
                    "auto-test : {diff} densites differentes du CPU dans le chunk {cx} {cz} ({})",
                    ex.join(", ")
                ));
            }
        }
        Ok(())
    }

    fn chunk(&self, cx: i32, cz: i32, out: &mut [f32]) -> bool {
        // Hors de la bordure du monde (30 M de blocs), le CPU garde la main
        if self.hors_service.load(Relaxed)
            || out.len() != self.n_chunk
            || cx.unsigned_abs() > 1_875_000
            || cz.unsigned_abs() > 1_875_000
        {
            return false;
        }
        match self.chunk_interne(cx, cz, out) {
            Ok(()) => {
                let n = self.stats.chunks.load(Relaxed);
                if n.is_power_of_two() && n >= 1024 || n % 65536 == 0 {
                    self.journal();
                }
                true
            }
            Err(e) => {
                if !self.hors_service.swap(true, Relaxed) {
                    log::error!("GPU final_density desactive, retour au CPU : {e}");
                }
                false
            }
        }
    }

    fn journal(&self) {
        log::info!("{}", self.bilan());
    }

    fn bilan(&self) -> String {
        let s = &self.stats;
        let (n, reg) = (s.chunks.load(Relaxed), s.regions.load(Relaxed));
        format!(
            "GPU final_density : {n} chunks dont {} avec aquiferes (copie {:.0} us en moyenne), {reg} regions calculees ({:.2} ms en moyenne), {} evictions, {} chunks sur CPU (structures, melange), {} ecarts d'aquifere, {} chunks a structures sur GPU",
            s.chunks_aq.load(Relaxed),
            s.ns_copie.load(Relaxed) as f64 / 1e3 / n.max(1) as f64,
            s.ns_regions.load(Relaxed) as f64 / 1e6 / reg.max(1) as f64,
            s.evictions.load(Relaxed),
            s.replis.load(Relaxed),
            s.ecarts.load(Relaxed),
            s.chunks_beard.load(Relaxed)
        )
    }

    fn chunk_interne(&self, cx: i32, cz: i32, out: &mut [f32]) -> Result<(), String> {
        let r = self.r;
        let (rx, rz) = (cx.div_euclid(r), cz.div_euclid(r));
        let (lx, lz) = (cx - rx * r, cz - rz * r);
        let region = self.region((rx, rz));
        let mut res = self.ressource()?;
        let resultat = self.copier(&region, rx, rz, lx, lz, &mut res, out);
        self.pool.lock().unwrap().push(res);
        if resultat.is_ok() && region.servis.fetch_add(1, Relaxed) + 1 == (r * r) as u32 {
            // Tous les chunks de la region sont servis : sa memoire GPU est rendue
            let mut c = self.cache.lock().unwrap();
            if c.carte
                .get(&(rx, rz))
                .is_some_and(|(e, _)| Arc::ptr_eq(e, &region))
            {
                c.carte.remove(&(rx, rz));
            }
        }
        resultat
    }

    fn copier(
        &self,
        region: &Region,
        rx: i32,
        rz: i32,
        lx: i32,
        lz: i32,
        res: &mut Ressources,
        out: &mut [f32],
    ) -> Result<(), String> {
        // Le premier worker qui demande la region la calcule ; les autres attendent ici
        let donnees = self.donnees(region, rx, rz, res)?;
        let t = Instant::now();
        let mut b = res.stream.launch_builder(&self.f_chunk);
        b.arg(&donnees.mem).arg(&mut res.buf).arg(&lx).arg(&lz);
        unsafe { b.launch(config(self.n_chunk)) }.ctx("copie du chunk")?;
        res.stream
            .memcpy_dtoh(&res.buf, &mut res.hote)
            .ctx("retour du chunk")?;
        // Sans suivi d'evenements, cudarc n'enregistre pas l'evenement du tampon verrouille et
        // `as_slice` n'attendrait pas la fin de la copie : on attend le flux lui-meme.
        res.stream.synchronize().ctx("attente du retour")?;
        out.copy_from_slice(res.hote.as_slice().ctx("lecture du retour")?);
        self.stats.chunks.fetch_add(1, Relaxed);
        self.stats
            .ns_copie
            .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        Ok(())
    }

    /// Les donnees de la region ; le premier worker qui la demande la calcule, les autres attendent.
    fn donnees<'r>(&self, region: &'r Region, rx: i32, rz: i32, res: &Ressources) -> Result<&'r DonneesRegion, String> {
        let d = region.donnees.get_or_init(|| {
            let t = Instant::now();
            let m = self.calculer(rx, rz, &res.stream);
            self.stats.regions.fetch_add(1, Relaxed);
            self.stats
                .ns_regions
                .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            m.map_err(|e| log::error!("GPU final_density : region {rx} {rz} : {e}"))
                .ok()
        });
        d.as_ref().ok_or_else(|| "calcul de region en echec".to_string())
    }

    /// Codes des blocs d'un chunk et son skip_sampling_above_y.
    fn chunk_codes(&self, cx: i32, cz: i32, out: &mut [i8], mut suite: Option<&mut dyn FnMut(&Arc<CudaStream>, &CudaSlice<i8>)>, grottes: Option<&mut CaveJob>) -> Option<i32> {
        if self.hors_service.load(Relaxed)
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
        let resultat = self.ressource().and_then(|mut res| {
            let rr = self.copier_codes(&region, rx, rz, lx, lz, &mut res, out, suite.take());
            self.pool.lock().unwrap().push(res);
            rr
        });
        match resultat {
            Ok(saut) => {
                if let Some(j) = grottes {
                    self.grottes_substance(&region, lx, lz, j);
                }
                if region.servis.fetch_add(1, Relaxed) + 1 == (r * r) as u32 {
                    let mut c = self.cache.lock().unwrap();
                    if c.carte.get(&(rx, rz)).is_some_and(|(e, _)| Arc::ptr_eq(e, &region)) {
                        c.carte.remove(&(rx, rz));
                    }
                }
                let n = self.stats.chunks.load(Relaxed);
                if n.is_power_of_two() && n >= 1024 || n % 65536 == 0 {
                    self.journal();
                }
                Some(saut)
            }
            Err(e) => {
                if !self.hors_service.swap(true, Relaxed) {
                    log::error!("GPU final_density desactive, retour au CPU : {e}");
                }
                None
            }
        }
    }

    fn chunk_codes_beard(&self, cx: i32, cz: i32, beard: &Beardifier, out: &mut [i8], mut suite: Option<&mut dyn FnMut(&Arc<CudaStream>, &CudaSlice<i8>)>, grottes: Option<&mut CaveJob>) -> Option<i32> {
        static SANS_BEARD: OnceLock<bool> = OnceLock::new();
        if *SANS_BEARD.get_or_init(|| std::env::var("HORIZON_GPU_BEARD").as_deref() == Ok("0")) {
            return None;
        }
        let (pieces, jonctions, kp, kj, zone) = beard.parts();
        if !kp.is_empty() || !kj.is_empty() || self.hors_service.load(Relaxed) || out.len() != self.n_chunk || cx.unsigned_abs() > 1_875_000 || cz.unsigned_abs() > 1_875_000 {
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
                [a[0], a[1], a[2], b[0], b[1], b[2], adj, p.ground_level_delta]
            })
            .collect();
        let jc: Vec<i32> = jonctions.iter().flat_map(|j| [j.source_x, j.source_ground_y, j.source_z]).collect();
        let z: Vec<i32> = zone.map_or(vec![0; 6], |b| {
            let (a, c) = (b.minimum(), b.maximum());
            vec![a[0], a[1], a[2], c[0], c[1], c[2]]
        });
        let r = self.r;
        let (rx, rz) = (cx.div_euclid(r), cz.div_euclid(r));
        let (lx, lz) = (cx - rx * r, cz - rz * r);
        let region = self.region((rx, rz));
        let resultat = self.ressource().and_then(|mut res| {
            let rr = (|| -> Result<i32, String> {
                let donnees = self.donnees(&region, rx, rz, &res)?;
                let (aq, cel) = match (&self.aq, &donnees.cellules) {
                    (Some(a), Some(c)) => (a, c),
                    _ => return Err("region sans cellules d'aquiferes".into()),
                };
                let t = Instant::now();
                let st = res.stream.clone();
                // Des tableaux vides ne s'allouent pas : au moins un element
                let pd = st.clone_htod(if pc.is_empty() { &[0i32][..] } else { &pc[..] }).ctx("pieces")?;
                let jd = st.clone_htod(if jc.is_empty() { &[0i32][..] } else { &jc[..] }).ctx("jonctions")?;
                let zd = st.clone_htod(&z[..]).ctx("zone")?;
                let saut = donnees.sauts[(lz * self.r + lx) as usize];
                let (np, nj, avec) = ((pc.len() / 8) as i32, (jc.len() / 3) as i32, i32::from(zone.is_some()));
                let (gx, h) = (16 * self.r, (self.n_chunk / 256) as i32);
                let (x0, z0) = (rx * self.r * 16, rz * self.r * 16);
                let gr = cel.grille;
                let mut b = st.launch_builder(&aq.beard);
                b.arg(&donnees.mem).arg(&mut res.buf_codes).arg(&pd).arg(&np).arg(&jd).arg(&nj).arg(&aq.noyau).arg(&zd).arg(&avec);
                b.arg(&cel.p).arg(&cel.st).arg(&saut).arg(&lx).arg(&lz).arg(&x0).arg(&z0).arg(&gx).arg(&self.min_y).arg(&h);
                b.arg(&gr[0]).arg(&gr[1]).arg(&gr[2]).arg(&gr[3]).arg(&gr[4]).arg(&gr[5]).arg(&aq.mer);
                unsafe { b.launch(config(self.n_chunk)) }.ctx("chunk a structures")?;
                // La surface du chunk a la suite, sur le meme flux
                if let Some(s) = suite.take() {
                    s(&st, &res.buf_codes);
                }
                st.memcpy_dtoh(&res.buf_codes, &mut res.hote_codes).ctx("retour des codes")?;
                st.synchronize().ctx("attente des codes")?;
                drop((pd, jd, zd));
                // Copie puis controle : les codes que le GPU vient d'ecrire sont lus une fois en memoire,
                // le controle les relit dans le cache. En erreur, `out` revient a zero.
                out.copy_from_slice(res.hote_codes.as_slice().ctx("lecture des codes")?);
                if out.contains(&-1) {
                    out.fill(0);
                    return Err("cellule d'aquifere hors grille".into());
                }
                self.stats.chunks.fetch_add(1, Relaxed);
                self.stats.chunks_beard.fetch_add(1, Relaxed);
                self.stats.ns_copie.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                Ok(saut)
            })();
            self.pool.lock().unwrap().push(res);
            rr
        });
        match resultat {
            Ok(saut) => {
                if let Some(j) = grottes {
                    self.grottes_substance(&region, lx, lz, j);
                }
                if region.servis.fetch_add(1, Relaxed) + 1 == (r * r) as u32 {
                    let mut c = self.cache.lock().unwrap();
                    if c.carte.get(&(rx, rz)).is_some_and(|(e, _)| Arc::ptr_eq(e, &region)) {
                        c.carte.remove(&(rx, rz));
                    }
                }
                Some(saut)
            }
            Err(e) => {
                if !self.hors_service.swap(true, Relaxed) {
                    log::error!("GPU final_density desactive, retour au CPU : {e}");
                }
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn copier_codes(
        &self,
        region: &Region,
        rx: i32,
        rz: i32,
        lx: i32,
        lz: i32,
        res: &mut Ressources,
        out: &mut [i8],
        suite: Option<&mut dyn FnMut(&Arc<CudaStream>, &CudaSlice<i8>)>,
    ) -> Result<i32, String> {
        let donnees = self.donnees(region, rx, rz, res)?;
        let (aq, codes) = match (&self.aq, &donnees.codes) {
            (Some(a), Some(c)) => (a, c),
            _ => return Err("region sans codes d'aquiferes".into()),
        };
        let t = Instant::now();
        let (gx, h) = (16 * self.r, (self.n_chunk / 256) as i32);
        let mut b = res.stream.launch_builder(&aq.chunk);
        b.arg(codes).arg(&mut res.buf_codes).arg(&lx).arg(&lz).arg(&gx).arg(&h);
        unsafe { b.launch(config(self.n_chunk)) }.ctx("copie des codes")?;
        // La surface du chunk a la suite, sur le meme flux
        if let Some(s) = suite {
            s(&res.stream, &res.buf_codes);
        }
        res.stream
            .memcpy_dtoh(&res.buf_codes, &mut res.hote_codes)
            .ctx("retour des codes")?;
        res.stream.synchronize().ctx("attente des codes")?;
        // Copie puis controle (chunk_codes_beard) : une lecture en memoire, le controle dans le cache.
        out.copy_from_slice(res.hote_codes.as_slice().ctx("lecture des codes")?);
        if out.contains(&-1) {
            out.fill(0);
            return Err("cellule d'aquifere hors grille".into());
        }
        self.stats.chunks.fetch_add(1, Relaxed);
        self.stats.chunks_aq.fetch_add(1, Relaxed);
        self.stats
            .ns_copie
            .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        Ok(donnees.sauts[(lz * self.r + lx) as usize])
    }

    /// Enfile sur le flux d'un lot de grottes la substance de ses cellules, avec l'aquifere de la region.
    fn grottes_substance(&self, region: &Arc<Region>, lx: i32, lz: i32, job: &mut CaveJob) {
        let (Some(aq), Some(Some(d))) = (&self.aq, region.donnees.get()) else {
            return;
        };
        let Some(cel) = &d.cellules else {
            return;
        };
        let saut = d.sauts[(lz * self.r + lx) as usize];
        let garde: Arc<dyn std::any::Any + Send + Sync> = region.clone();
        let aqr = grottes::AquifereRegion {
            f: &aq.grottes,
            p: &cel.p,
            st: &cel.st,
            grille: cel.grille,
            saut,
            mer: aq.mer,
        };
        if let Err(e) = job.enfiler_substance(aqr, garde) {
            static SIGNALEE: OnceLock<()> = OnceLock::new();
            if SIGNALEE.set(()).is_ok() {
                log::error!("GPU grottes : substance : {e}");
            }
        }
    }

    fn region(&self, cle: (i32, i32)) -> Arc<Region> {
        let mut c = self.cache.lock().unwrap();
        c.horloge += 1;
        let h = c.horloge;
        if let Some((r, t)) = c.carte.get_mut(&cle) {
            *t = h;
            return r.clone();
        }
        if c.carte.len() >= c.capacite
            && let Some(vieille) = c.carte.iter().min_by_key(|(_, (_, t))| *t).map(|(k, _)| *k)
        {
            // Liberee quand le dernier worker qui la lit a fini
            c.carte.remove(&vieille);
            self.stats.evictions.fetch_add(1, Relaxed);
        }
        let r = Arc::new(Region::default());
        c.carte.insert(cle, (r.clone(), h));
        r
    }

    fn ressource(&self) -> Result<Ressources, String> {
        if let Some(r) = self.pool.lock().unwrap().pop() {
            return Ok(r);
        }
        let stream = flux_prioritaire(&self.ctx)?;
        let buf = stream.alloc_zeros::<f32>(self.n_chunk).ctx("tampon GPU")?;
        // Memoire verrouillee ordinaire : le defaut de cudarc (write-combined) est lent a relire
        let hote = unsafe { self.ctx.alloc_pinned_with_flags::<f32>(self.n_chunk, 0) }
            .ctx("tampon hote")?;
        let buf_codes = stream.alloc_zeros::<i8>(self.n_chunk).ctx("tampon GPU des codes")?;
        let hote_codes = unsafe { self.ctx.alloc_pinned_with_flags::<i8>(self.n_chunk, 0) }
            .ctx("tampon hote des codes")?;
        Ok(Ressources {
            stream,
            buf,
            hote,
            buf_codes,
            hote_codes,
        })
    }

    /// Toute la region sur le flux `st` : tables d'axes, grilles de coins, grille de blocs.
    fn calculer(&self, rx: i32, rz: i32, st: &Arc<CudaStream>) -> Result<DonneesRegion, String> {
        let min = [rx * self.r * 16, self.min_y, rz * self.r * 16];
        let mins: Vec<[i32; 3]> = self
            .pas_grilles
            .iter()
            .enumerate()
            .map(|(k, pas)| {
                if k == 0 {
                    min
                } else {
                    std::array::from_fn(|i| min[i].div_euclid(pas[i]) * pas[i])
                }
            })
            .collect();
        let mut mem = st
            .alloc_zeros::<f32>(self.taille)
            .ctx("memoire de region")?;
        for (f, gr, nb) in &self.f_ax {
            let mut b = st.launch_builder(f);
            b.arg(&mut mem)
                .arg(&mins[*gr][0])
                .arg(&mins[*gr][1])
                .arg(&mins[*gr][2]);
            unsafe { b.launch(config(*nb)) }.ctx("noyau d'axes")?;
        }
        for (k, k2d, k3d, n3d) in &self.f_gr {
            if let Some((f, nb)) = k2d {
                let mut b = st.launch_builder(f);
                b.arg(&mut mem)
                    .arg(&mins[*k][0])
                    .arg(&mins[*k][1])
                    .arg(&mins[*k][2]);
                unsafe { b.launch(config(*nb)) }.ctx("passe 2D")?;
            }
            let mut b = st.launch_builder(k3d);
            b.arg(&mut mem)
                .arg(&mins[*k][0])
                .arg(&mins[*k][1])
                .arg(&mins[*k][2]);
            unsafe { b.launch(config(*n3d)) }.ctx("grille 3D")?;
        }
        let (codes, sauts, cellules) = match &self.aq {
            Some(aq) => {
                let (c, s, cel) = self.aquiferes(aq, &mem, min, st)?;
                (Some(c), s, Some(cel))
            }
            None => {
                st.synchronize().ctx("calcul de region")?;
                (None, Vec::new(), None)
            }
        };
        Ok(DonneesRegion {
            mem,
            codes,
            cellules,
            sauts,
        })
    }

    /// Codes des blocs de la region (memes grilles que aquifer_blocks, sur toute la region).
    fn aquiferes(&self, aq: &Aquiferes, mem: &CudaSlice<f32>, min: [i32; 3], st: &Arc<CudaStream>) -> Result<(CudaSlice<i8>, Vec<i32>, Cellules), String> {
        let h = (self.n_chunk / 256) as i32;
        let gx = 16 * self.r;
        let (x0, z0) = (min[0], min[2]);
        let (x1, z1) = (x0 + gx - 1, z0 + gx - 1);
        let (y0, y1) = (self.min_y, self.min_y + h - 1);
        let (cx0, cy0, cz0) = ((x0 - 5) >> 4, (y0 + 1).div_euclid(12) - 1, (z0 - 5) >> 4);
        let (cx1, cy1, cz1) = (((x1 - 5) >> 4) + 1, (y1 + 1).div_euclid(12) + 1, ((z1 - 5) >> 4) + 1);
        let (ncx, ncy, ncz) = (cx1 - cx0 + 1, cy1 - cy0 + 1, cz1 - cz0 + 1);
        let (qx0, qz0) = ((cx0 * 16 - 48) >> 2, (cz0 * 16 - 16) >> 2);
        let (qx1, qz1) = ((cx1 * 16 + 9 + 16) >> 2, (cz1 * 16 + 9 + 16) >> 2);
        let (nqx, nqz) = (qx1 - qx0 + 1, qz1 - qz0 + 1);
        let nch = self.r * self.r;
        let (chx0, chz0) = (x0 >> 4, z0 >> 4);
        let mut s = st.alloc_zeros::<i32>((nqx * nqz) as usize).ctx("surface")?;
        let mut p = st.alloc_zeros::<i32>((3 * ncx * ncy * ncz) as usize).ctx("positions")?;
        let mut stt = st.alloc_zeros::<i32>((2 * ncx * ncy * ncz) as usize).ctx("statuts")?;
        let mut saut = st.alloc_zeros::<i32>(nch as usize).ctx("sauts")?;
        let mut codes = st.alloc_zeros::<i8>(self.n_chunk * nch as usize).ctx("codes")?;
        let (lo, hi) = aq.positionnel;
        {
            let mut l = st.launch_builder(&aq.surface);
            l.arg(&mut s).arg(&qx0).arg(&qz0).arg(&nqx).arg(&nqz);
            unsafe { l.launch(config((nqx * nqz) as usize)) }.ctx("surface")?;
        }
        {
            let mut l = st.launch_builder(&aq.cellules);
            l.arg(&mut p).arg(&mut stt).arg(&s).arg(&qx0).arg(&qz0).arg(&nqx).arg(&cx0).arg(&cy0).arg(&cz0);
            l.arg(&ncx).arg(&ncy).arg(&ncz).arg(&lo).arg(&hi).arg(&aq.mer);
            unsafe { l.launch(config((ncx * ncy * ncz) as usize)) }.ctx("cellules")?;
        }
        {
            let mut l = st.launch_builder(&aq.saut);
            l.arg(&mut saut).arg(&s).arg(&qx0).arg(&qz0).arg(&nqx).arg(&chx0).arg(&chz0).arg(&self.r).arg(&nch);
            l.arg(&self.min_y).arg(&h);
            unsafe { l.launch(config(nch as usize)) }.ctx("sauts")?;
        }
        {
            let mut l = st.launch_builder(&aq.blocs);
            l.arg(mem).arg(&mut codes).arg(&p).arg(&stt).arg(&saut).arg(&x0).arg(&z0).arg(&gx).arg(&gx).arg(&self.r);
            l.arg(&self.min_y).arg(&h).arg(&cx0).arg(&cy0).arg(&cz0).arg(&ncx).arg(&ncy).arg(&ncz).arg(&aq.mer);
            unsafe { l.launch(config(self.n_chunk * nch as usize)) }.ctx("blocs")?;
        }
        st.synchronize().ctx("aquiferes de la region")?;
        let sauts = st.clone_dtoh(&saut).ctx("retour des sauts")?;
        Ok((
            codes,
            sauts,
            Cellules {
                p,
                st: stt,
                grille: [cx0, cy0, cz0, ncx, ncy, ncz],
            },
        ))
    }
}

/// Evalue des racines du graphe en une liste de points sur le GPU, exactement comme
/// `DensitySampler::sample_value` de Horizon (verification et banc d'essai).
/// Resultat : une liste de valeurs par racine, dans l'ordre des points.
pub fn evaluate_points(
    compiler: &DensityCompiler,
    seed: i64,
    roots: &[FunctionId],
    points: &[[i32; 3]],
) -> Result<Vec<Vec<f32>>, String> {
    let src = traduction::generer_points(compiler, roots, seed).map_err(|e| e.to_string())?;
    if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP") {
        let _ = std::fs::write(f, &src);
    }
    let ctx = contexte()?;
    let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
    let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
    let module = ctx.load_module(ptx(&src, arch)?).ctx("chargement du module")?;
    let st = ctx.new_stream().ctx("flux CUDA")?;
    let plat: Vec<i32> = points.iter().flatten().copied().collect();
    let p = st.clone_htod(&plat).ctx("envoi des points")?;
    let n = points.len();
    let mut sorties = Vec::new();
    for k in 0..roots.len() {
        let f = module.load_function(&format!("kpt_{k}")).ctx("noyau de points")?;
        let mut out = st.alloc_zeros::<f32>(n.max(1)).ctx("sortie")?;
        let ni = n as i32;
        let mut b = st.launch_builder(&f);
        b.arg(&p).arg(&mut out).arg(&ni);
        unsafe { b.launch(config(n.max(1))) }.ctx("evaluation")?;
        st.synchronize().ctx("evaluation")?;
        let mut v = st.clone_dtoh(&out).ctx("retour")?;
        v.truncate(n);
        sorties.push(v);
    }
    Ok(sorties)
}

/// Parametres d'un calcul d'aquiferes sur GPU (jalon de verification) : un rectangle de chunks.
pub struct AquiferBatch<'a> {
    /// Racines dans l'ordre de AquiferField::ALL.
    pub roots: [FunctionId; 6],
    /// Etat (lo, hi) de la fabrique positionnelle des aquiferes.
    pub positional: (u64, u64),
    pub sea_level: i32,
    pub min_y: i32,
    pub height: i32,
    /// Premier chunk (x, z) et nombre de chunks (x, z).
    pub chunk0: (i32, i32),
    pub chunks: (i32, i32),
    /// Densite finale de chaque chunk a la suite (chunk = z * nx + x), dans l'ordre de `fill_noise`.
    pub density: &'a [f32],
}

/// Codes de chaque bloc (0 bloc par defaut, 1 air, 2 fluide par defaut, 3 lave, +4 mise a jour planifiee,
/// -1 hors grille) et skip_sampling_above_y de chaque chunk.
pub fn aquifer_blocks(compiler: &DensityCompiler, seed: i64, b: &AquiferBatch) -> Result<(Vec<i8>, Vec<i32>), String> {
    let src = traduction::generer_aquifere(compiler, b.roots, seed).map_err(|e| e.to_string())?;
    if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP") {
        let _ = std::fs::write(f, &src);
    }
    let ctx = contexte()?;
    let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
    let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
    let module = ctx.load_module(ptx(&src, arch)?).ctx("chargement du module")?;
    let st = ctx.new_stream().ctx("flux CUDA")?;
    let (nchx, nchz) = b.chunks;
    let nch = nchx * nchz;
    let (x0, z0) = (b.chunk0.0 * 16, b.chunk0.1 * 16);
    let (x1, z1) = ((b.chunk0.0 + nchx) * 16 - 1, (b.chunk0.1 + nchz) * 16 - 1);
    let (y0, y1) = (b.min_y, b.min_y + b.height - 1);
    // Grille des cellules (NoiseBasedAquifer::new sur tout le rectangle)
    let (cx0, cy0, cz0) = ((x0 - 5) >> 4, (y0 + 1).div_euclid(12) - 1, (z0 - 5) >> 4);
    let (cx1, cy1, cz1) = (((x1 - 5) >> 4) + 1, (y1 + 1).div_euclid(12) + 1, ((z1 - 5) >> 4) + 1);
    let (ncx, ncy, ncz) = (cx1 - cx0 + 1, cy1 - cy0 + 1, cz1 - cz0 + 1);
    // Colonnes de surface : decalages de compute_fluid (x de -48 a +16, z de -16 a +16) autour des cellules
    let (qx0, qz0) = ((cx0 * 16 - 48) >> 2, (cz0 * 16 - 16) >> 2);
    let (qx1, qz1) = ((cx1 * 16 + 9 + 16) >> 2, (cz1 * 16 + 9 + 16) >> 2);
    let (nqx, nqz) = (qx1 - qx0 + 1, qz1 - qz0 + 1);
    let mut s = st.alloc_zeros::<i32>((nqx * nqz) as usize).ctx("surface")?;
    let mut p = st.alloc_zeros::<i32>((3 * ncx * ncy * ncz) as usize).ctx("positions")?;
    let mut stt = st.alloc_zeros::<i32>((2 * ncx * ncy * ncz) as usize).ctx("statuts")?;
    let mut saut = st.alloc_zeros::<i32>(nch as usize).ctx("sauts")?;
    let dens = st.clone_htod(b.density).ctx("densites")?;
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
        let k = f("kaq_cellules")?;
        let mut l = st.launch_builder(&k);
        l.arg(&mut p).arg(&mut stt).arg(&s).arg(&qx0).arg(&qz0).arg(&nqx).arg(&cx0).arg(&cy0).arg(&cz0);
        l.arg(&ncx).arg(&ncy).arg(&ncz).arg(&lo).arg(&hi).arg(&b.sea_level);
        unsafe { l.launch(config((ncx * ncy * ncz) as usize)) }.ctx("cellules")?;
    }
    {
        let k = f("kaq_saut")?;
        let mut l = st.launch_builder(&k);
        l.arg(&mut saut).arg(&s).arg(&qx0).arg(&qz0).arg(&nqx).arg(&b.chunk0.0).arg(&b.chunk0.1).arg(&nchx).arg(&nch);
        l.arg(&b.min_y).arg(&b.height);
        unsafe { l.launch(config(nch as usize)) }.ctx("sauts")?;
    }
    {
        let k = f("kaq_blocs")?;
        let mut l = st.launch_builder(&k);
        l.arg(&dens).arg(&mut codes).arg(&p).arg(&stt).arg(&saut).arg(&b.chunk0.0).arg(&b.chunk0.1).arg(&nchx).arg(&nch);
        l.arg(&b.min_y).arg(&b.height).arg(&cx0).arg(&cy0).arg(&cz0).arg(&ncx).arg(&ncy).arg(&ncz).arg(&b.sea_level);
        unsafe { l.launch(config(b.density.len())) }.ctx("blocs")?;
    }
    st.synchronize().ctx("aquiferes")?;
    Ok((st.clone_dtoh(&codes).ctx("retour des codes")?, st.clone_dtoh(&saut).ctx("retour des sauts")?))
}


/// Evalue des bruits (`Noise::get3`) en une liste de points sur le GPU (verification de la surface).
pub fn evaluate_noises(
    compiler: &DensityCompiler,
    noises: &[&dyn horizon_core::noise::Noise],
    points: &[[f64; 3]],
) -> Result<Vec<Vec<f32>>, String> {
    let src = traduction::generer_bruits(compiler, noises).map_err(|e| e.to_string())?;
    let ctx = contexte()?;
    let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
    let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
    let module = ctx.load_module(ptx(&src, arch)?).ctx("chargement du module")?;
    let st = ctx.new_stream().ctx("flux CUDA")?;
    let plat: Vec<f64> = points.iter().flatten().copied().collect();
    let p = st.clone_htod(&plat).ctx("envoi des points")?;
    let n = points.len();
    let mut sorties = Vec::new();
    for k in 0..noises.len() {
        let f = module.load_function(&format!("knz_{k}")).ctx("noyau de bruit")?;
        let mut out = st.alloc_zeros::<f32>(n.max(1)).ctx("sortie")?;
        let ni = n as i32;
        let mut b = st.launch_builder(&f);
        b.arg(&p).arg(&mut out).arg(&ni);
        unsafe { b.launch(config(n.max(1))) }.ctx("evaluation")?;
        st.synchronize().ctx("evaluation")?;
        let mut v = st.clone_dtoh(&out).ctx("retour")?;
        v.truncate(n);
        sorties.push(v);
    }
    Ok(sorties)
}

/// Tirages d'une fabrique positionnelle Xoroshiro (lo, hi) en chaque position : premier next_float,
/// premier next_double, trois next_float successifs.
pub fn positional_draws(
    compiler: &DensityCompiler,
    positional: (u64, u64),
    points: &[[i32; 3]],
) -> Result<(Vec<f32>, Vec<f64>, Vec<f32>), String> {
    let src = traduction::generer_bruits(compiler, &[]).map_err(|e| e.to_string())?;
    let ctx = contexte()?;
    let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
    let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
    let module = ctx.load_module(ptx(&src, arch)?).ctx("chargement du module")?;
    let st = ctx.new_stream().ctx("flux CUDA")?;
    let plat: Vec<i32> = points.iter().flatten().copied().collect();
    let p = st.clone_htod(&plat).ctx("envoi des points")?;
    let n = points.len().max(1);
    let mut f = st.alloc_zeros::<f32>(n).ctx("sortie")?;
    let mut d = st.alloc_zeros::<f64>(n).ctx("sortie")?;
    let mut f3 = st.alloc_zeros::<f32>(3 * n).ctx("sortie")?;
    let k = module.load_function("ktirages").ctx("ktirages")?;
    let (ni, lo, hi) = (points.len() as i32, positional.0, positional.1);
    let mut b = st.launch_builder(&k);
    b.arg(&p).arg(&mut f).arg(&mut d).arg(&mut f3).arg(&ni).arg(&lo).arg(&hi);
    unsafe { b.launch(config(n)) }.ctx("tirages")?;
    st.synchronize().ctx("tirages")?;
    Ok((
        st.clone_dtoh(&f).ctx("retour")?,
        st.clone_dtoh(&d).ctx("retour")?,
        st.clone_dtoh(&f3).ctx("retour")?,
    ))
}


/// Jalon S2 de la surface : quarts choisis par le zoom des biomes (graine deja obfusquee) et
/// temperatures ajustees a la hauteur pour chaque (temperature de base, modificateur gele).
#[allow(clippy::type_complexity)]
pub fn surface_s2_check(
    compiler: &DensityCompiler,
    zoom_seed: i64,
    points: &[[i32; 3]],
    biomes: &[(f32, bool)],
    sea_level: i32,
    hauteur: &horizon_core::noise::SimplexNoise,
    gel: &horizon_core::noise::NoiseStack,
    info: &horizon_core::noise::SimplexNoise,
) -> Result<(Vec<[i32; 3]>, Vec<Vec<f32>>), String> {
    let src = traduction::generer_surface_s2(compiler, hauteur, gel, info).map_err(|e| e.to_string())?;
    let ctx = contexte()?;
    let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
    let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
    let module = ctx.load_module(ptx(&src, arch)?).ctx("chargement du module")?;
    let st = ctx.new_stream().ctx("flux CUDA")?;
    let plat: Vec<i32> = points.iter().flatten().copied().collect();
    let p = st.clone_htod(&plat).ctx("envoi des points")?;
    let n = points.len();
    let mut q = st.alloc_zeros::<i32>(3 * n.max(1)).ctx("quarts")?;
    {
        let k = module.load_function("ksf_quarts").ctx("ksf_quarts")?;
        let ni = n as i32;
        let mut b = st.launch_builder(&k);
        b.arg(&p).arg(&mut q).arg(&ni).arg(&zoom_seed);
        unsafe { b.launch(config(n.max(1))) }.ctx("zoom")?;
    }
    let bases: Vec<f32> = biomes.iter().map(|b| b.0).collect();
    let geles: Vec<i32> = biomes.iter().map(|b| i32::from(b.1)).collect();
    let bd = st.clone_htod(&bases).ctx("bases")?;
    let gd = st.clone_htod(&geles).ctx("geles")?;
    let mut t = st.alloc_zeros::<f32>((n * biomes.len()).max(1)).ctx("temperatures")?;
    {
        let k = module.load_function("ksf_temperatures").ctx("ksf_temperatures")?;
        let (ni, nb) = (n as i32, biomes.len() as i32);
        let mut b = st.launch_builder(&k);
        b.arg(&p).arg(&bd).arg(&gd).arg(&nb).arg(&mut t).arg(&ni).arg(&sea_level);
        unsafe { b.launch(config((n * biomes.len()).max(1))) }.ctx("temperatures")?;
    }
    st.synchronize().ctx("jalon S2")?;
    let qv = st.clone_dtoh(&q).ctx("retour")?;
    let tv = st.clone_dtoh(&t).ctx("retour")?;
    let quarts = (0..n).map(|i| [qv[3 * i], qv[3 * i + 1], qv[3 * i + 2]]).collect();
    let temps = (0..biomes.len()).map(|k| tv[k * n..(k + 1) * n].to_vec()).collect();
    Ok((quarts, temps))
}


/// Regle de surface sous forme neutre (traduite depuis `MaterialRule` de horizon-world).
#[derive(Clone, Debug)]
pub enum SurfaceRule {
    Empty,
    Block(u32),
    Sequence(Vec<usize>),
    Condition { condition: usize, rule: usize },
    Bandlands,
    /// `positive_y` : les y hors desquels la densite n'est jamais positive (rien n'y est place).
    OreVein { blocks: [u32; 3], raw_chance: f32, density: FunctionId, richness: FunctionId, gap: FunctionId, positive_y: [i32; 2] },
}
/// Condition de surface sous forme neutre (ancres deja resolues, noms deja traduits en index).
#[derive(Clone, Debug)]
pub enum SurfaceCondition {
    Hole,
    Steep,
    Biomes(Vec<u32>),
    Temperature,
    AbovePreliminary,
    Noise { noise: usize, min: f64, max: f64, is_3d: bool },
    YAbove { anchor: i32, multiplier: i32, add_stone: bool },
    Water { offset: i32, multiplier: i32, add_stone: bool },
    StoneDepth { offset: i32, add_surface: bool, secondary_range: i32, ceiling: bool },
    Gradient { low: i32, high: i32, positional: (u64, u64) },
    Not(usize),
}
/// Tout ce que la surface d'un monde utilise (jalon S3).
pub struct SurfaceSetup<'a> {
    pub rules: Vec<SurfaceRule>,
    pub conditions: Vec<SurfaceCondition>,
    pub root: usize,
    pub noises: Vec<&'a dyn horizon_core::noise::Noise>,
    pub surface_noise: &'a dyn horizon_core::noise::Noise,
    pub secondary_noise: &'a dyn horizon_core::noise::Noise,
    pub clay_noise: &'a dyn horizon_core::noise::Noise,
    pub bands: [u32; 192],
    pub root_positional: (u64, u64),
    pub ore_positional: (u64, u64),
    pub surface_level: FunctionId,
    /// Par biome : temperature de base et modificateur gele.
    pub temperatures: Vec<(f32, bool)>,
    pub temperature_noises: (&'a horizon_core::noise::SimplexNoise, &'a horizon_core::noise::NoiseStack, &'a horizon_core::noise::SimplexNoise),
    pub zoom_seed: i64,
    pub sea_level: i32,
    pub min_y: i32,
    pub height: i32,
    /// Piliers des badlands erodes et icebergs (jalon S4) ; `None` : colonnes sans extensions.
    pub extensions: Option<SurfaceExtensions<'a>>,
    /// Les etats du remplissage par code (& 3) : bloc par defaut, air, fluide par defaut, lave.
    pub fill_states: [u32; 4],
    /// Les etats a fluide que la regle peut ecrire (l'eau des marais) : marques de post-traitement.
    pub fluid_states: Vec<u32>,
}
/// Les extensions de la surface (`material_extensions.rs` de Horizon).
pub struct SurfaceExtensions<'a> {
    pub eroded_badlands: Option<u32>,
    /// Au plus deux biomes (frozen_ocean, deep_frozen_ocean).
    pub frozen_oceans: Vec<u32>,
    /// badlands_surface, badlands_pillar, badlands_pillar_roof, iceberg_surface, iceberg_pillar,
    /// iceberg_pillar_roof.
    pub noises: [&'a dyn horizon_core::noise::Noise; 6],
    pub default_block: u32,
    pub packed_ice: u32,
    pub snow: u32,
}

/// Un programme de grilles (filons, surface preliminaire) charge sur le GPU.
struct Volume {
    f_ax: Vec<Lancement>,
    f_gr: Vec<(usize, Option<(CudaFunction, usize)>, CudaFunction, usize)>,
    pas: Vec<[i32; 3]>,
    taille: usize,
}
impl Volume {
    fn charger(ctx: &Arc<CudaContext>, arch: &'static str, prog: traduction::Programme) -> Result<Self, String> {
        let module = ctx.load_module(ptx(&prog.src, arch)?).ctx("module d'un filon")?;
        let f = |nom: &str| module.load_function(nom).ctx(nom);
        let mut f_ax = Vec::new();
        for &(k, gr, nb) in &prog.axes {
            f_ax.push((f(&format!("kax_{k}"))?, gr, nb));
        }
        let mut f_gr = Vec::new();
        for k in (1..prog.grilles.len()).chain(std::iter::once(0)) {
            let gr = prog.grilles[k];
            let k2d = if prog.bords[k] > 0 { Some((f(&format!("k2d_{k}"))?, gr.colonnes() * prog.bords[k])) } else { None };
            f_gr.push((k, k2d, f(&format!("k3d_{k}"))?, gr.n()));
        }
        Ok(Self { f_ax, f_gr, pas: prog.grilles.iter().map(|g| g.pas).collect(), taille: prog.taille })
    }
    /// Toutes les grilles du programme sur l'origine `min` ; la grille des blocs est au debut de `mem`, en
    /// ((z * CGX) + x) * CGY + y.
    fn grille(&self, st: &Arc<CudaStream>, mem: &mut CudaSlice<f32>, min: [i32; 3]) -> Result<(), String> {
        let mins: Vec<[i32; 3]> = self.pas.iter().enumerate().map(|(k, pas)| if k == 0 { min } else { std::array::from_fn(|i| min[i].div_euclid(pas[i]) * pas[i]) }).collect();
        st.memset_zeros(mem).ctx("memoire d'un programme")?;
        for (f, gr, nb) in &self.f_ax {
            let mut b = st.launch_builder(f);
            b.arg(&mut *mem).arg(&mins[*gr][0]).arg(&mins[*gr][1]).arg(&mins[*gr][2]);
            unsafe { b.launch(config(*nb)) }.ctx("noyau d'axes")?;
        }
        for (k, k2d, k3d, n3d) in &self.f_gr {
            if let Some((f, nb)) = k2d {
                let mut b = st.launch_builder(f);
                b.arg(&mut *mem).arg(&mins[*k][0]).arg(&mins[*k][1]).arg(&mins[*k][2]);
                unsafe { b.launch(config(*nb)) }.ctx("passe 2D")?;
            }
            let mut b = st.launch_builder(k3d);
            b.arg(&mut *mem).arg(&mins[*k][0]).arg(&mins[*k][1]).arg(&mins[*k][2]);
            unsafe { b.launch(config(*n3d)) }.ctx("grille 3D")?;
        }
        Ok(())
    }
}

/// Les tampons d'un appel de la surface, gardes d'un chunk a l'autre (un jeu par worker a la fois).
struct RessourcesSurface {
    st: Arc<CudaStream>,
    pl: CudaSlice<f32>,
    cd: CudaSlice<i8>,
    bd: CudaSlice<u16>,
    /// Les blocs d'iceberg (etat + 1).
    glace: CudaSlice<u32>,
    /// Par bloc : above, below, water ; par colonne : 10 entiers (ksf_tetes) et le bruit secondaire.
    meta: CudaSlice<i32>,
    coli: CudaSlice<i32>,
    cold: CudaSlice<f64>,
    fin: CudaSlice<u32>,
    err: CudaSlice<i32>,
    hote: cudarc::driver::PinnedHostSlice<u32>,
    hote_err: cudarc::driver::PinnedHostSlice<i32>,
    /// Sections compactes, cartes de hauteur, marques et erreur (`enable_sections`)
    sec: CudaSlice<u32>,
    hote_sec: cudarc::driver::PinnedHostSlice<u32>,
    /// Marques de chaque colonne (ksec_hauteurs pour ksec_marques)
    nm: CudaSlice<u32>,
}
/// La construction des sections compactes du remplissage (`SurfaceGpu::enable_sections`).
struct SectionsGpu {
    _module: Arc<cudarc::driver::CudaModule>,
    k_sections: CudaFunction,
    k_hauteurs: CudaFunction,
    k_marques: CudaFunction,
    flags: CudaSlice<u8>,
    nflags: u32,
    air: u32,
    test: bool,
}

#[derive(Default)]
struct StatsSurface {
    chunks: AtomicU64,
    ns: AtomicU64,
    replis: AtomicU64,
    /// Sections compactes : chunks poses, repasses par les etats bruts, ecarts au CPU (mode test)
    sections: AtomicU64,
    sections_replis: AtomicU64,
    sections_ecarts: AtomicU64,
    /// HORIZON_GPU_SURFACE_PROFIL : filons, surface preliminaire, colonnes, blocs, envois et retour (ns)
    phases: [AtomicU64; 5],
}

/// La surface d'un monde sur GPU : etats finaux d'un chunk apres remplissage et surface.
pub struct SurfaceGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<cudarc::driver::CudaModule>,
    k_tetes: CudaFunction,
    k_descente: CudaFunction,
    k_blocs: CudaFunction,
    volumes: Vec<Volume>,
    niveau: Volume,
    height: i32,
    pool: Mutex<Vec<RessourcesSurface>>,
    stats: StatsSurface,
    /// Par region de NIVEAU_R x NIVEAU_R chunks, calculee une fois pour ses chunks ; les plus anciennes
    /// sont oubliees au-dela de `capacite`.
    regions: Mutex<Regions>,
    capacite: usize,
    /// Les grilles des filons d'une region : y de depart et hauteur.
    vlo: i32,
    vh: i32,
    sections: OnceLock<Option<SectionsGpu>>,
}
/// La surface preliminaire d'une region (plan y = 0, en z puis x, sur l'hote) et les densites de ses
/// filons (sur le GPU : une grille par racine, en ((z * SF_VR) + x) * SF_VH + y - SF_VLO).
struct RegionSurface {
    plan: Vec<f32>,
    filons: CudaSlice<f32>,
}
type CelluleRegion = Arc<std::sync::OnceLock<Result<Arc<RegionSurface>, String>>>;
#[derive(Default)]
struct Regions {
    carte: std::collections::HashMap<(i32, i32), CelluleRegion>,
    ordre: std::collections::VecDeque<(i32, i32)>,
}
impl Drop for SurfaceGpu {
    fn drop(&mut self) {
        let n = self.stats.chunks.load(Relaxed);
        if n > 0 {
            eprintln!(
                "GPU surface : {n} chunks ({} us en moyenne par appel), {} replis CPU ; sections compactes : {} chunks, {} par les etats bruts, {} ecarts au CPU (mode test)",
                self.stats.ns.load(Relaxed) / n / 1000,
                self.stats.replis.load(Relaxed),
                self.stats.sections.load(Relaxed),
                self.stats.sections_replis.load(Relaxed),
                self.stats.sections_ecarts.load(Relaxed),
            );
            if std::env::var_os("HORIZON_GPU_SURFACE_PROFIL").is_some() {
                let p = self.stats.phases.each_ref().map(|a| a.load(Relaxed) / n / 1000);
                eprintln!("GPU surface : par chunk, filons {} us, surface preliminaire {} us, colonnes {} us, blocs {} us, envois et retour {} us", p[0], p[1], p[2], p[3], p[4]);
            }
        }
    }
}
impl std::fmt::Debug for SurfaceGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SurfaceGpu({} filons)", self.volumes.len())
    }
}
impl SurfaceGpu {
    pub fn new(compiler: &DensityCompiler, setup: &SurfaceSetup, seed: i64) -> Result<Self, String> {
        let prog = traduction::generer_surface(compiler, setup, seed).map_err(|e| e.to_string())?;
        if let Some(f) = std::env::var_os("HORIZON_GPU_DUMP_SURFACE") {
            let f = std::path::PathBuf::from(f);
            let _ = std::fs::write(&f, &prog.src);
            for (k, v) in prog.volumes.iter().enumerate() {
                let _ = std::fs::write(f.with_extension(format!("filon{k}.cu")), &v.src);
            }
            let _ = std::fs::write(f.with_extension("niveau.cu"), &prog.niveau.src);
        }
        let ctx = contexte()?;
        let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
        let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
        let module = ctx.load_module(ptx(&prog.src, arch)?).ctx("module de surface")?;
        let k_tetes = module.load_function("ksf_tetes").ctx("ksf_tetes")?;
        let k_descente = module.load_function("ksf_descente").ctx("ksf_descente")?;
        let k_blocs = module.load_function("ksf_blocs").ctx("ksf_blocs")?;
        let mut volumes = Vec::new();
        for v in prog.volumes {
            volumes.push(Volume::charger(&ctx, arch, v)?);
        }
        let niveau = Volume::charger(&ctx, arch, prog.niveau)?;
        let capacite = std::env::var("HORIZON_GPU_SURFACE_REGIONS").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
        Ok(Self {
            ctx,
            _module: module,
            k_tetes,
            k_descente,
            k_blocs,
            volumes,
            niveau,
            height: setup.height,
            pool: Mutex::new(Vec::new()),
            stats: StatsSurface::default(),
            regions: Mutex::new(Regions::default()),
            capacite,
            vlo: prog.filons_y.0,
            vh: prog.filons_y.1,
            sections: OnceLock::new(),
        })
    }
    /// HORIZON_GPU_SECTIONS : le GPU construit aussi les sections compactes du remplissage, ses cartes de
    /// hauteur et ses marques (sections.cu), que `fill_noise_sections` pose telles quelles. `flags` : par
    /// etat, 1 air, 2 fluide, 4 bloque le mouvement (`TerrainChunk::section_flags`) ; `test` : les etats
    /// reviennent aussi, pour comparer. Une fois par dimension ; faux si le module ne se charge pas.
    pub fn enable_sections(&self, flags: &[u8], air: u32, test: bool) -> bool {
        self.sections
            .get_or_init(|| {
                let r = (|| -> Result<SectionsGpu, String> {
                    let (maj, min) = self.ctx.compute_capability().ctx("capacite du GPU")?;
                    let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
                    let module = self
                        .ctx
                        .load_module(ptx(include_str!("sections.cu"), arch)?)
                        .ctx("module des sections")?;
                    let st = self.ctx.new_stream().ctx("flux CUDA")?;
                    let flags_d = st.clone_htod(flags).ctx("drapeaux des etats")?;
                    st.synchronize().ctx("drapeaux des etats")?;
                    Ok(SectionsGpu {
                        k_sections: module.load_function("ksec_sections").ctx("ksec_sections")?,
                        k_hauteurs: module.load_function("ksec_hauteurs").ctx("ksec_hauteurs")?,
                        k_marques: module.load_function("ksec_marques").ctx("ksec_marques")?,
                        _module: module,
                        flags: flags_d,
                        nflags: flags.len() as u32,
                        air,
                        test,
                    })
                })();
                r.map_err(|e| log::error!("GPU sections : pas de sections sur GPU : {e}")).ok()
            })
            .is_some()
    }
    /// Un chunk dont les sections compactes du GPU sont posees (`ok`) ou passent par les etats bruts.
    pub fn note_sections(&self, ok: bool) {
        if ok {
            self.stats.sections.fetch_add(1, Relaxed);
        } else {
            self.stats.sections_replis.fetch_add(1, Relaxed);
        }
    }
    /// Mode test : des sections compactes qui different de celles du CPU.
    pub fn note_sections_mismatch(&self) {
        self.stats.sections_ecarts.fetch_add(1, Relaxed);
    }
    /// Mots u32 de la sortie des sections d'un chunk : `SECTION_WORDS` par section, 512 hauteurs (ocean
    /// puis surface), 2 compteurs et 2 x `SECTION_MARKS` marques, puis le mot d'erreur.
    pub fn sections_len(&self) -> usize {
        (self.height as usize / 16) * SECTION_WORDS + 512 + 2 + 2 * SECTION_MARKS + 1
    }
    /// La region (rx, rz) : surface preliminaire et filons, calcules au premier chunk qui la demande.
    fn region(&self, st: &Arc<CudaStream>, rx: i32, rz: i32) -> Result<Arc<RegionSurface>, String> {
        let cellule = {
            let mut p = self.regions.lock().unwrap();
            match p.carte.get(&(rx, rz)) {
                Some(c) => c.clone(),
                None => {
                    let c: CelluleRegion = Arc::new(std::sync::OnceLock::new());
                    p.carte.insert((rx, rz), c.clone());
                    p.ordre.push_back((rx, rz));
                    // Liberee quand le dernier chunk qui la lit a fini
                    while p.ordre.len() > self.capacite {
                        if let Some(v) = p.ordre.pop_front() {
                            p.carte.remove(&v);
                        }
                    }
                    c
                }
            }
        };
        cellule
            .get_or_init(|| {
                let vr = 16 * traduction::NIVEAU_R;
                let n = (vr * vr) as usize;
                let mut mem = st.alloc_zeros::<f32>(self.niveau.taille).ctx("memoire de la surface preliminaire")?;
                self.niveau.grille(st, &mut mem, [rx * vr, 0, rz * vr])?;
                let plan = st.clone_dtoh(&mem.slice(0..n)).ctx("retour de la surface preliminaire")?;
                let vn = n * self.vh as usize;
                let mut filons = st.alloc_zeros::<f32>(vn * self.volumes.len().max(1)).ctx("filons de la region")?;
                for (k, v) in self.volumes.iter().enumerate() {
                    let mut m = st.alloc_zeros::<f32>(v.taille).ctx("memoire d'un filon")?;
                    v.grille(st, &mut m, [rx * vr, self.vlo, rz * vr])?;
                    st.memcpy_dtod(&m.slice(0..vn), &mut filons.slice_mut(k * vn..(k + 1) * vn)).ctx("copie d'un filon")?;
                }
                // Lue ensuite depuis les flux des autres workers
                st.synchronize().ctx("region de la surface")?;
                Ok(Arc::new(RegionSurface { plan, filons }))
            })
            .clone()
    }
    /// Nombre de biomes attendus par `chunk_states` : les quarts des 3 x 3 chunks autour du chunk.
    pub fn biome_count(&self) -> usize {
        9 * 16 * (self.height as usize / 4)
    }
    /// Le chunk est passe par le CPU.
    pub fn note_cpu_fallback(&self) {
        self.stats.replis.fetch_add(1, Relaxed);
    }
    fn ressource(&self) -> Result<RessourcesSurface, String> {
        if let Some(r) = self.pool.lock().unwrap().pop() {
            return Ok(r);
        }
        let n = 256 * self.height as usize;
        let st = flux_prioritaire(&self.ctx)?;
        Ok(RessourcesSurface {
            pl: st.alloc_zeros::<f32>(256).ctx("surface preliminaire")?,
            cd: st.alloc_zeros::<i8>(n).ctx("codes")?,
            bd: st.alloc_zeros::<u16>(self.biome_count()).ctx("biomes")?,
            glace: st.alloc_zeros::<u32>(n).ctx("icebergs")?,
            meta: st.alloc_zeros::<i32>(3 * n).ctx("descente")?,
            coli: st.alloc_zeros::<i32>(10 * 256).ctx("colonnes")?,
            cold: st.alloc_zeros::<f64>(256).ctx("colonnes")?,
            fin: st.alloc_zeros::<u32>(n).ctx("etats finaux")?,
            err: st.alloc_zeros::<i32>(1).ctx("erreur")?,
            // Memoire verrouillee ordinaire (le defaut de cudarc, write-combined, est lent a relire)
            hote: unsafe { self.ctx.alloc_pinned_with_flags::<u32>(n, 0) }.ctx("tampon hote")?,
            hote_err: unsafe { self.ctx.alloc_pinned_with_flags::<i32>(1, 0) }.ctx("tampon hote")?,
            sec: st.alloc_zeros::<u32>(self.sections_len()).ctx("sections")?,
            nm: st.alloc_zeros::<u32>(512).ctx("marques des colonnes")?,
            hote_sec: unsafe { self.ctx.alloc_pinned_with_flags::<u32>(self.sections_len(), 0) }.ctx("tampon hote")?,
            st,
        })
    }
    /// Les etats finaux du chunk (remplissage puis surface, extensions comprises) dans l'ordre de
    /// fill_noise : l'etat sur 30 bits, le bit 31 quand le remplissage planifie une mise a jour de fluide
    /// (code + 4 sur de l'eau ou de la lave), le bit 30 quand la surface y ecrit un etat a fluide. `codes` :
    /// les codes du remplissage GPU ; `biomes` : les quarts des 3 x 3 chunks (dz puis dx de -1 a +1, chaque
    /// chunk en sections). `Ok(false)` : une condition n'a pas pu etre evaluee, le chunk doit passer par le
    /// CPU. `GpuFinalDensity::chunk_blocks_surface` fait de meme sans renvoyer les codes au GPU.
    pub fn chunk_states(&self, chunk_x: i32, chunk_z: i32, codes: &[i8], biomes: &[u16], out: &mut [u32]) -> Result<bool, String> {
        let n = 256 * self.height as usize;
        if codes.len() != n || out.len() != n || biomes.len() != self.biome_count() {
            return Err("tailles inattendues".into());
        }
        let t = Instant::now();
        let mut res = self.ressource()?;
        let st = res.st.clone();
        let r = (|| {
            st.memcpy_htod(codes, &mut res.cd).ctx("codes")?;
            let _region = self.enfiler(&mut res, &st, None, chunk_x, chunk_z, biomes, false)?;
            st.synchronize().ctx("attente de la surface")?;
            self.lire(&mut res, out, None)
        })();
        self.rendre(res, r.is_ok(), t);
        r
    }
    /// La surface du chunk mise en file sur `st`, apres le noyau qui y ecrit ses codes (`cd`, sinon les
    /// codes deja envoyes dans `res`). Les etats se lisent par `lire` une fois `st` attendu ; la region
    /// renvoyee doit rester en vie jusque-la.
    fn enfiler(
        &self,
        res: &mut RessourcesSurface,
        st: &Arc<CudaStream>,
        cd: Option<&CudaSlice<i8>>,
        chunk_x: i32,
        chunk_z: i32,
        biomes: &[u16],
        sections: bool,
    ) -> Result<Arc<RegionSurface>, String> {
        let n = 256 * self.height as usize;
        let profil = cd.is_none() && std::env::var_os("HORIZON_GPU_SURFACE_PROFIL").is_some();
        let mut t = Instant::now();
        let mut phase = |k: usize| -> Result<(), String> {
            if profil {
                st.synchronize().ctx("profil")?;
                self.stats.phases[k].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                t = Instant::now();
            }
            Ok(())
        };
        let r = traduction::NIVEAU_R;
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
        st.memcpy_htod(&pl[..], &mut res.pl).ctx("surface preliminaire")?;
        phase(1)?;
        st.memcpy_htod(biomes, &mut res.bd).ctx("biomes")?;
        st.memset_zeros(&mut res.err).ctx("erreur")?;
        phase(4)?;
        st.memset_zeros(&mut res.glace).ctx("icebergs")?;
        st.memset_zeros(&mut res.meta).ctx("descente")?;
        let cd: &CudaSlice<i8> = match cd {
            Some(c) => c,
            None => &res.cd,
        };
        let mut b = st.launch_builder(&self.k_tetes);
        b.arg(cd)
            .arg(&res.bd)
            .arg(&res.pl)
            .arg(&mut res.glace)
            .arg(&mut res.coli)
            .arg(&mut res.cold)
            .arg(&mut res.err)
            .arg(&chunk_x)
            .arg(&chunk_z);
        // Une warp par colonne
        unsafe { b.launch(config(256 * 32)) }.ctx("colonnes de la surface")?;
        let mut b = st.launch_builder(&self.k_descente);
        b.arg(cd).arg(&mut res.coli).arg(&mut res.meta);
        unsafe { b.launch(config(256 * 32)) }.ctx("descente de la surface")?;
        phase(2)?;
        let mut b = st.launch_builder(&self.k_blocs);
        b.arg(cd)
            .arg(&res.bd)
            .arg(&region.filons)
            .arg(&res.glace)
            .arg(&res.meta)
            .arg(&res.coli)
            .arg(&res.cold)
            .arg(&mut res.fin)
            .arg(&mut res.err)
            .arg(&chunk_x)
            .arg(&chunk_z)
            .arg(&ox)
            .arg(&oz);
        unsafe { b.launch(config(n)) }.ctx("blocs de la surface")?;
        phase(3)?;
        let sec = if sections { self.sections.get().and_then(Option::as_ref) } else { None };
        if let Some(sg) = sec {
            let ns = self.height as usize / 16;
            let (a, b) = (ns * SECTION_WORDS, ns * SECTION_WORDS + 512);
            let fin_mq = b + 2 + 2 * SECTION_MARKS;
            st.memset_zeros(&mut res.sec.slice_mut(fin_mq..fin_mq + 1)).ctx("erreur des sections")?;
            let h = self.height;
            let cap = SECTION_MARKS as i32;
            {
                let hm = res.sec.slice(a..b);
                let mut l = st.launch_builder(&sg.k_hauteurs);
                l.arg(&res.fin).arg(&sg.flags).arg(&sg.nflags).arg(&sg.air).arg(&h).arg(&hm).arg(&mut res.nm);
                unsafe { l.launch(LaunchConfig { grid_dim: (256, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }) }
                    .ctx("hauteurs des sections")?;
            }
            {
                let mq = res.sec.slice(b..fin_mq);
                let er = res.sec.slice(fin_mq..fin_mq + 1);
                let mut l = st.launch_builder(&sg.k_marques);
                l.arg(&res.fin).arg(&sg.air).arg(&h).arg(&res.nm).arg(&mq).arg(&cap).arg(&er);
                unsafe { l.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }
                    .ctx("marques des sections")?;
            }
            {
                let out = res.sec.slice(0..a);
                let er = res.sec.slice(fin_mq..fin_mq + 1);
                let mut l = st.launch_builder(&sg.k_sections);
                l.arg(&res.fin).arg(&sg.flags).arg(&sg.nflags).arg(&sg.air).arg(&h).arg(&out).arg(&er);
                unsafe { l.launch(LaunchConfig { grid_dim: (ns as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }
                    .ctx("sections")?;
            }
            st.memcpy_dtoh(&res.sec, &mut res.hote_sec).ctx("retour des sections")?;
            if sg.test {
                st.memcpy_dtoh(&res.fin, &mut res.hote).ctx("retour des etats")?;
            }
        } else {
            st.memcpy_dtoh(&res.fin, &mut res.hote).ctx("retour des etats")?;
        }
        st.memcpy_dtoh(&res.err, &mut res.hote_err).ctx("retour de l'erreur")?;
        if profil {
            st.synchronize().ctx("profil")?;
            self.stats.phases[4].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        }
        Ok(region)
    }
    /// Les etats d'une surface mise en file, son flux attendu (suivi d'evenements coupe : la memoire
    /// verrouillee ne s'attend pas seule).
    fn lire(&self, res: &mut RessourcesSurface, out: &mut [u32], sections: Option<&mut Vec<u32>>) -> Result<bool, String> {
        let sec = self.sections.get().and_then(Option::as_ref);
        match (sections, sec) {
            (Some(v), Some(sg)) => {
                let lu = res.hote_sec.as_slice().ctx("lecture des sections")?;
                if lu[lu.len() - 1] == 0 {
                    v.clear();
                    v.extend_from_slice(lu);
                    if sg.test {
                        out.copy_from_slice(res.hote.as_slice().ctx("lecture des etats")?);
                    }
                } else {
                    // Trop de marques ou de palettes : les etats bruts, pour le CPU
                    v.clear();
                    res.st.memcpy_dtoh(&res.fin, &mut res.hote).ctx("retour des etats")?;
                    res.st.synchronize().ctx("retour des etats")?;
                    out.copy_from_slice(res.hote.as_slice().ctx("lecture des etats")?);
                }
            }
            (Some(v), None) => {
                v.clear();
                out.copy_from_slice(res.hote.as_slice().ctx("lecture des etats")?);
            }
            (None, _) => out.copy_from_slice(res.hote.as_slice().ctx("lecture des etats")?),
        }
        Ok(res.hote_err.as_slice().ctx("lecture de l'erreur")?[0] == 0)
    }
    /// Rend les tampons (pas ceux d'un appel en erreur, dont le flux peut etre casse).
    fn rendre(&self, res: RessourcesSurface, ok: bool, t: Instant) {
        if ok {
            self.pool.lock().unwrap().push(res);
            self.stats.chunks.fetch_add(1, Relaxed);
            self.stats.ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        }
    }
}

/// Un blob de minerai (OreFeature.doPlace) a calculer : ses `count` spheres `[x, y, z, r]` (avant
/// elagage) a partir de `first` dans les spheres du lot, et sa boite (coin bas, cote en X et Z, hauteur).
#[derive(Clone, Copy, Debug)]
pub struct OreBlob {
    pub first: u32,
    pub count: u32,
    pub start: [i32; 3],
    pub size_xz: i32,
    pub size_y: i32,
}

#[derive(Default)]
struct StatsMinerais {
    lots: AtomicU64,
    blobs: AtomicU64,
    ns_envoi: AtomicU64,
    ns_attente: AtomicU64,
    replis: AtomicU64,
    erreurs: AtomicU64,
    ecarts: AtomicU64,
}

/// Les cellules des blobs de minerai sur GPU (`HORIZON_GPU_ORES`) : l'elagage des spheres de
/// OreFeature.doPlace, puis le masque des cellules de chaque rangee (x, y) de la boite du blob.
/// Le lot d'un chunk est envoye au debut de sa decoration et lu a l'etape des minerais.
pub struct OreGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<cudarc::driver::CudaModule>,
    k_blobs: CudaFunction,
    pool: Mutex<Vec<RessourcesMinerais>>,
    stats: StatsMinerais,
}
/// Les tampons d'un lot, en mots de 32 bits : en entree les blobs (8 mots chacun) puis les spheres
/// (8 mots chacune), en sortie les erreurs (1 mot par blob) puis les rangees.
struct RessourcesMinerais {
    st: Arc<CudaStream>,
    /// Capacites : blobs, spheres, rangees
    cap: [usize; 3],
    entree: CudaSlice<u32>,
    sortie: CudaSlice<u32>,
    h_entree: PinnedHostSlice<u32>,
    h_sortie: PinnedHostSlice<u32>,
}
/// Un lot de blobs envoye au GPU, a lire apres `OreGpu::wait` et a rendre par `OreGpu::finish`.
pub struct OreJob {
    res: Option<RessourcesMinerais>,
    /// Premiere rangee de chaque blob, et le total a la fin
    rangees: Vec<usize>,
    attendu: bool,
    ok: bool,
    /// Le retour lu, une fois attendu (adresse et longueur) : `as_slice` est un appel au pilote
    vue: (usize, usize),
}
impl Drop for OreJob {
    fn drop(&mut self) {
        // Les copies du flux ecrivent dans ces tampons : ils ne sont liberes qu'apres elles
        if let Some(r) = &self.res
            && !self.attendu
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
        let n = self.stats.lots.load(Relaxed);
        if n > 0 {
            eprintln!(
                "GPU minerais : {n} lots, {} blobs par lot, envoi {} us, attente {} us par lot ; {} features repassees par le CPU, {} blobs en erreur, {} ecarts au CPU (mode test)",
                self.stats.blobs.load(Relaxed) / n,
                self.stats.ns_envoi.load(Relaxed) / n / 1000,
                self.stats.ns_attente.load(Relaxed) / n / 1000,
                self.stats.replis.load(Relaxed),
                self.stats.erreurs.load(Relaxed),
                self.stats.ecarts.load(Relaxed),
            );
        }
    }
}
impl OreGpu {
    pub fn new() -> Result<Self, String> {
        let ctx = contexte()?;
        let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
        let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
        let module = ctx
            .load_module(ptx(include_str!("minerais.cu"), arch)?)
            .ctx("module des minerais")?;
        let k_blobs = module.load_function("kmin_blobs").ctx("kmin_blobs")?;
        Ok(Self {
            ctx,
            _module: module,
            k_blobs,
            pool: Mutex::new(Vec::new()),
            stats: StatsMinerais::default(),
        })
    }
    /// Une feature planifiee est repassee par le CPU.
    pub fn note_cpu_fallback(&self) {
        self.stats.replis.fetch_add(1, Relaxed);
    }
    /// Mode test : un blob dont les cellules different de celles du CPU.
    pub fn note_mismatch(&self) {
        self.stats.ecarts.fetch_add(1, Relaxed);
    }
    fn ressource(&self, besoin: [usize; 3]) -> Result<RessourcesMinerais, String> {
        let mut pool = self.pool.lock().unwrap();
        if let Some(k) = pool.iter().position(|r| (0..3).all(|i| r.cap[i] >= besoin[i])) {
            return Ok(pool.swap_remove(k));
        }
        // Trop petites (ou aucune) : une plus grande remplace l'une d'elles
        let ancienne = pool.pop();
        drop(pool);
        let cap: [usize; 3] = std::array::from_fn(|i| {
            let defaut = [256, 8192, 65536][i];
            besoin[i]
                .max(defaut)
                .max(ancienne.as_ref().map_or(0, |r| r.cap[i]))
                .next_power_of_two()
        });
        let st = match ancienne {
            Some(r) => r.st.clone(),
            None => self.ctx.new_stream().ctx("flux CUDA")?,
        };
        let [b, s, m] = cap;
        Ok(RessourcesMinerais {
            entree: st.alloc_zeros::<u32>(8 * b + 8 * s).ctx("blobs")?,
            sortie: st.alloc_zeros::<u32>(b + m).ctx("rangees")?,
            h_entree: unsafe { self.ctx.alloc_pinned_with_flags::<u32>(8 * b + 8 * s, 0) }
                .ctx("tampon hote")?,
            h_sortie: unsafe { self.ctx.alloc_pinned_with_flags::<u32>(b + m, 0) }
                .ctx("tampon hote")?,
            st,
            cap,
        })
    }
    /// Envoie les blobs d'un chunk ; le calcul et le retour se font sur un flux a part.
    pub fn submit(&self, blobs: &[OreBlob], spheres: &[[f64; 4]]) -> Result<OreJob, String> {
        let t = Instant::now();
        let nb = blobs.len();
        let mut rangees = Vec::with_capacity(nb + 1);
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
            rangees.push(total);
            total += (b.size_xz * b.size_y) as usize;
        }
        rangees.push(total);
        if nb == 0 {
            return Ok(OreJob {
                res: None,
                rangees,
                attendu: true,
                ok: true,
                vue: (0, 0),
            });
        }
        let mut res = self.ressource([nb, spheres.len(), total])?;
        let (n_entree, n_sortie) = (8 * nb + 8 * spheres.len(), nb + total);
        {
            let h = res.h_entree.as_mut_slice().ctx("tampon hote")?;
            let (tetes, reste) = h.split_at_mut(4 * nb);
            let (debuts, sp) = reste.split_at_mut(4 * nb);
            for (i, b) in blobs.iter().enumerate() {
                tetes[4 * i..4 * i + 4].copy_from_slice(&[
                    b.first,
                    b.count,
                    rangees[i] as u32,
                    (b.size_xz | (b.size_y << 8)) as u32,
                ]);
                debuts[4 * i..4 * i + 4].copy_from_slice(&[
                    b.start[0] as u32,
                    b.start[1] as u32,
                    b.start[2] as u32,
                    0,
                ]);
            }
            // Les doubles en petit-boutiste, comme sur le GPU
            for (mots, v) in sp.chunks_exact_mut(2).zip(spheres.iter().flatten()) {
                let b = v.to_bits();
                mots[0] = b as u32;
                mots[1] = (b >> 32) as u32;
            }
        }
        let st = res.st.clone();
        st.memcpy_htod(
            &res.h_entree.as_slice().ctx("tampon hote")?[..n_entree],
            &mut res.entree.slice_mut(0..n_entree),
        )
        .ctx("envoi des blobs")?;
        let nbi = nb as i32;
        let mut l = st.launch_builder(&self.k_blobs);
        l.arg(&res.entree).arg(&mut res.sortie).arg(&nbi);
        let cfg = LaunchConfig {
            grid_dim: (nb as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { l.launch(cfg) }.ctx("blobs de minerai")?;
        st.memcpy_dtoh(
            &res.sortie.slice(0..n_sortie),
            &mut res.h_sortie.as_mut_slice().ctx("tampon hote")?[..n_sortie],
        )
        .ctx("retour des rangees")?;
        self.stats.lots.fetch_add(1, Relaxed);
        self.stats.blobs.fetch_add(nb as u64, Relaxed);
        self.stats.ns_envoi.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        Ok(OreJob {
            res: Some(res),
            rangees,
            attendu: false,
            ok: true,
            vue: (0, 0),
        })
    }
    /// Attend le lot (une fois) ; `false` si le GPU a echoue (le lot repasse alors par le CPU).
    pub fn wait(&self, job: &mut OreJob) -> bool {
        if !job.attendu {
            let t = Instant::now();
            job.ok = job.res.as_ref().is_some_and(|r| r.st.synchronize().is_ok());
            if job.ok
                && let Some(s) = job.res.as_ref().and_then(|r| r.h_sortie.as_slice().ok())
            {
                job.vue = (s.as_ptr() as usize, s.len());
            } else {
                job.ok = false;
            }
            job.attendu = true;
            self.stats.ns_attente.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        }
        job.ok
    }
    /// Les rangees du blob `i` d'un lot attendu (rangee `x * size_y + y`, bit z), ou None si le blob
    /// est en erreur (une cellule hors de sa boite) ou le lot pas attendu.
    pub fn rows<'a>(&self, job: &'a OreJob, i: usize) -> Option<&'a [u32]> {
        if !job.attendu || !job.ok {
            return None;
        }
        job.res.as_ref()?;
        // Le tampon du lot, lu par `wait` : il vit tant que le lot
        let sortie = unsafe { std::slice::from_raw_parts(job.vue.0 as *const u32, job.vue.1) };
        let nb = job.rangees.len() - 1;
        if *sortie.get(i)? != 0 {
            self.stats.erreurs.fetch_add(1, Relaxed);
            return None;
        }
        sortie.get(nb + job.rangees[i]..nb + *job.rangees.get(i + 1)?)
    }
    /// Rend les tampons d'un lot, apres l'avoir attendu s'il ne l'a pas ete.
    pub fn finish(&self, mut job: OreJob) {
        self.wait(&mut job);
        if job.ok
            && let Some(r) = job.res.take()
        {
            self.pool.lock().unwrap().push(r);
        }
    }
}
