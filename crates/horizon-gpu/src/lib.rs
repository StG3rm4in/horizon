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
//! Reglages : `HORIZON_GPU_REGION` (cote d'une region en chunks, 8), `HORIZON_GPU_REGIONS_MAX`
//! (regions gardees en memoire GPU, 128), `HORIZON_GPU_CACHE` (dossier du PTX compile).
//! Au demarrage, un auto-test compare trois chunks au CPU ; a la moindre difference ou erreur CUDA,
//! le GPU est desactive et tout repasse par le CPU.
mod traduction;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use horizon_core::density::graph::{DensityCompiler, FunctionId};
use horizon_core::density::{DensitySampler, SamplerContext};
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
        // Attente de la fin d'un calcul : HORIZON_GPU_SYNC = block (le thread dort), spin, yield ou
        // auto (heuristique CUDA, souvent spin : le coeur tourne a vide pendant l'attente).
        use cudarc::driver::sys::CUctx_flags::*;
        let attente = match std::env::var("HORIZON_GPU_SYNC").as_deref() {
            Ok("block") => Some(CU_CTX_SCHED_BLOCKING_SYNC),
            Ok("spin") => Some(CU_CTX_SCHED_SPIN),
            Ok("yield") => Some(CU_CTX_SCHED_YIELD),
            _ => None,
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
        let partage = Arc::new(Partage {
            nom: nom.to_owned(),
            moteur: OnceLock::new(),
        });
        let p = partage.clone();
        let travail = move || {
            let t = Instant::now();
            let m = Moteur::demarrer(prog, r, capacite, min_y, height)
                .and_then(|m| m.auto_test(&*cpu).map(|()| m));
            match m {
                Ok(m) => {
                    let msg = format!(
                        "GPU final_density ({}) : pret en {:.1} s, auto-test identique au CPU",
                        p.nom,
                        t.elapsed().as_secs_f64()
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

#[derive(Default)]
struct Region {
    mem: OnceLock<Option<CudaSlice<f32>>>,
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
}

#[derive(Default)]
struct Stats {
    chunks: AtomicU64,
    ns_copie: AtomicU64,
    regions: AtomicU64,
    ns_regions: AtomicU64,
    evictions: AtomicU64,
    replis: AtomicU64,
}

type Lancement = (CudaFunction, usize, usize);

struct Moteur {
    ctx: Arc<CudaContext>,
    /// Tables d'axes : (noyau, grille, threads)
    f_ax: Vec<Lancement>,
    /// Grilles dans l'ordre de lancement : (grille, passe 2D, noyau 3D, points)
    f_gr: Vec<(usize, Option<(CudaFunction, usize)>, CudaFunction, usize)>,
    f_chunk: CudaFunction,
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
        Ok(Self {
            f_chunk: f("kchunk")?,
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
            "GPU final_density : {n} chunks (copie {:.0} us en moyenne), {reg} regions calculees ({:.2} ms en moyenne), {} evictions, {} chunks sur CPU (structures, melange)",
            s.ns_copie.load(Relaxed) as f64 / 1e3 / n.max(1) as f64,
            s.ns_regions.load(Relaxed) as f64 / 1e6 / reg.max(1) as f64,
            s.evictions.load(Relaxed),
            s.replis.load(Relaxed)
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
        let mem = region.mem.get_or_init(|| {
            let t = Instant::now();
            let m = self.calculer(rx, rz, &res.stream);
            self.stats.regions.fetch_add(1, Relaxed);
            self.stats
                .ns_regions
                .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            m.map_err(|e| log::error!("GPU final_density : region {rx} {rz} : {e}"))
                .ok()
        });
        let mem = mem.as_ref().ok_or("calcul de region en echec")?;
        let t = Instant::now();
        let mut b = res.stream.launch_builder(&self.f_chunk);
        b.arg(mem).arg(&mut res.buf).arg(&lx).arg(&lz);
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
        let stream = self.ctx.new_stream().ctx("flux CUDA")?;
        let buf = stream.alloc_zeros::<f32>(self.n_chunk).ctx("tampon GPU")?;
        // Memoire verrouillee ordinaire : le defaut de cudarc (write-combined) est lent a relire
        let hote = unsafe { self.ctx.alloc_pinned_with_flags::<f32>(self.n_chunk, 0) }
            .ctx("tampon hote")?;
        Ok(Ressources { stream, buf, hote })
    }

    /// Toute la region sur le flux `st` : tables d'axes, grilles de coins, grille de blocs.
    fn calculer(&self, rx: i32, rz: i32, st: &Arc<CudaStream>) -> Result<CudaSlice<f32>, String> {
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
        st.synchronize().ctx("calcul de region")?;
        Ok(mem)
    }
}
