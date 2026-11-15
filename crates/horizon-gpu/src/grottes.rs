//! Masque des carvers sur GPU (`HORIZON_GPU_CAVES`) : voir GROTTES.md.
//!
//! Le CPU planifie les carvers de chaque origine une fois (tirages qui ne dependent pas du chunk
//! cible) et envoie, au debut d'un chunk, les elements qui peuvent l'atteindre. Le GPU trace les
//! tunnels et les canyons, rasterise leurs ellipsoides dans le masque du chunk (`kgr_trace`), puis,
//! des que la region du chunk est sur le GPU, calcule la substance de chaque cellule creusee avec
//! l'aquifere (`kgr_substance`, dans le module des aquiferes). Le tout tourne sur le flux du lot,
//! hors du chemin critique du worker ; le CPU lit le resultat au moment de creuser.
use super::{Contexte, contexte, ptx};
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
/// Mots de masque au plus (256 colonnes de 384 cellules : 1 536 mots)
pub const CAVE_MASK_WORDS: usize = 1536;

/// Un element planifie d'un carver pour un chunk cible, tel que `kgr_trace` le lit (16 mots).
#[derive(Clone, Copy, Debug, Default)]
pub struct CaveItem {
    /// Tunnel : graine de son LegacyRandom ; canyon : etat de son flux apres la table des largeurs
    pub seed: u64,
    pub position: [f64; 3],
    /// Tunnel : vertical_scale ; canyon : y_scale
    pub scale: f64,
    /// Tunnel : multiplicateurs de rayon du run ; salle : ses rayons
    pub horizontal: f64,
    pub vertical: f64,
    /// Grotte : floor_level
    pub floor: f64,
    pub kind: u32,
    /// Appel de carver (budget de 2^24 cellules testees)
    pub call: u32,
    pub end: i32,
    /// Canyon : debut de sa table dans les largeurs du lot
    pub width: u32,
    pub thickness: f32,
    pub yaw: f32,
    pub pitch: f32,
    /// Canyon, horizontal_radius_factor : 0 constant (`factor[0]`), 1 uniforme
    pub factor_kind: u32,
    pub factor: [f32; 2],
    /// Canyon : vertical_radius_default_factor, vertical_radius_center_factor
    pub vertical_factors: [f32; 2],
}
impl CaveItem {
    fn mots(&self) -> [u64; 16] {
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

/// Le masque d'un chunk : bornes de CarvingMask (ymin, ymax) et minimum de la generation.
#[derive(Clone, Copy, Debug)]
pub struct CaveTarget {
    pub chunk: [i32; 2],
    pub min_y: i32,
    pub max_y: i32,
    pub generation_min_y: i32,
    /// Longueur de la table des largeurs d'un canyon (hauteur de la generation)
    pub generation_height: i32,
}

#[derive(Default)]
struct StatsGrottes {
    lots: AtomicU64,
    elements: AtomicU64,
    cellules: AtomicU64,
    cellules_max: AtomicU64,
    ns_envoi: AtomicU64,
    ns_attente: AtomicU64,
    substances: AtomicU64,
    replis: AtomicU64,
    erreurs: AtomicU64,
    ecarts_masque: AtomicU64,
    ecarts_substance: AtomicU64,
}

pub struct CaveGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<cudarc::driver::CudaModule>,
    k_trace: CudaFunction,
    sinus: CudaSlice<f32>,
    pool: Mutex<Vec<RessourcesGrottes>>,
    /// Codes de substance copies par lot (octets), agrandi si un chunk en a plus
    cap_codes: AtomicU64,
    stats: StatsGrottes,
}
/// Un seul tampon par lot, en mots de 64 bits : elements (16 mots chacun), largeurs (2 par mot), puis
/// la sortie : masque, compteurs, drapeaux (a zero), puis les codes de substance (8 par mot). L'aller
/// copie elements, largeurs et zeros d'un coup ; le retour copie sortie et codes d'un coup.
struct RessourcesGrottes {
    st: Arc<CudaStream>,
    /// Capacite en mots
    cap: usize,
    tampon: CudaSlice<u64>,
    h_entree: PinnedHostSlice<u64>,
    h_sortie: PinnedHostSlice<u64>,
}

/// Le lot d'un chunk : envoye par `CaveGpu::submit`, complete par l'aquifere du GPU au remplissage
/// du chunk, lu par `CaveGpu::wait` au moment de creuser, rendu par `CaveGpu::finish`.
pub struct CaveJob {
    res: Option<RessourcesGrottes>,
    cible: CaveTarget,
    nw: usize,
    nc: usize,
    /// Debut de la sortie dans le tampon
    e: usize,
    /// Mots de codes que le tampon peut recevoir
    cw: usize,
    hauteur: i32,
    /// Codes de substance demandes (0 : pas d'aquifere GPU pour ce chunk)
    codes: usize,
    attendu: bool,
    ok: bool,
    /// La region dont l'aquifere est lu, gardee jusqu'a la fin du flux du lot
    region: Option<Arc<dyn Any + Send + Sync>>,
}
impl Drop for CaveJob {
    fn drop(&mut self) {
        // Les copies du flux ecrivent dans ces tampons : ils ne sont liberes qu'apres elles
        if let Some(r) = &self.res
            && !self.attendu
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
        let n = s.lots.load(Relaxed);
        if n > 0 {
            eprintln!(
                "GPU grottes : {n} lots, {} elements et {} cellules creusees par lot (max {}), envoi {} us, attente {} us par lot ; {} avec l'aquifere GPU ; {} chunks repasses par le CPU, {} lots en erreur ; ecarts au CPU (mode test) : {} masques, {} substances",
                s.elements.load(Relaxed) / n,
                s.cellules.load(Relaxed) / n,
                s.cellules_max.load(Relaxed),
                s.ns_envoi.load(Relaxed) / n / 1000,
                s.ns_attente.load(Relaxed) / n / 1000,
                s.substances.load(Relaxed),
                s.replis.load(Relaxed),
                s.erreurs.load(Relaxed),
                s.ecarts_masque.load(Relaxed),
                s.ecarts_substance.load(Relaxed),
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
    /// `sine` : la table de sinus de Horizon (`horizon_core::geometry::trig::sine_table`).
    pub fn new(sine: &[f32]) -> Result<Self, String> {
        if sine.len() != 65536 {
            return Err("table de sinus".into());
        }
        let ctx = contexte()?;
        let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
        let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
        let module = ctx
            .load_module(ptx(include_str!("grottes.cu"), arch)?)
            .ctx("module des grottes")?;
        let k_trace = module.load_function("kgr_trace").ctx("kgr_trace")?;
        let st = ctx.new_stream().ctx("flux CUDA")?;
        let sinus = st.clone_htod(sine).ctx("table de sinus")?;
        st.synchronize().ctx("table de sinus")?;
        Ok(Self {
            ctx,
            _module: module,
            k_trace,
            sinus,
            pool: Mutex::new(Vec::new()),
            cap_codes: AtomicU64::new(16384),
            stats: StatsGrottes::default(),
        })
    }
    /// Un chunk dont les grottes passent par le CPU (lot en erreur, budget, substance manquante).
    pub fn note_cpu_fallback(&self) {
        self.stats.replis.fetch_add(1, Relaxed);
    }
    /// Mode test : un masque ou des substances qui different du CPU.
    pub fn note_mismatch(&self, masque: bool) {
        if masque {
            self.stats.ecarts_masque.fetch_add(1, Relaxed);
        } else {
            self.stats.ecarts_substance.fetch_add(1, Relaxed);
        }
    }
    fn ressource(&self, besoin: usize) -> Result<RessourcesGrottes, String> {
        let mut pool = self.pool.lock().unwrap();
        if let Some(k) = pool.iter().position(|r| r.cap >= besoin) {
            return Ok(pool.swap_remove(k));
        }
        let ancienne = pool.pop();
        drop(pool);
        let cap = besoin
            .max(8192)
            .max(ancienne.as_ref().map_or(0, |r| r.cap))
            .next_power_of_two();
        let st = match ancienne {
            Some(r) => r.st.clone(),
            None => self.ctx.new_stream().ctx("flux CUDA")?,
        };
        let pinned = |n: usize| unsafe { self.ctx.alloc_pinned_with_flags::<u64>(n, 0) }.ctx("tampon hote");
        Ok(RessourcesGrottes {
            tampon: st.alloc_zeros::<u64>(cap).ctx("tampon des grottes")?,
            h_entree: pinned(cap)?,
            h_sortie: pinned(cap)?,
            st,
            cap,
        })
    }
    /// Envoie les elements d'un chunk (`calls` appels de carver) et lance leur trace sur le flux du lot.
    pub fn submit(&self, cible: CaveTarget, items: &[CaveItem], widths: &[f32], calls: usize) -> Result<CaveJob, String> {
        let t = Instant::now();
        let hauteur = cible.max_y - cible.min_y + 1;
        if !(1..=384).contains(&hauteur) {
            return Err("hauteur du masque".into());
        }
        let nw = (256 * hauteur as usize).div_ceil(64);
        let (ni, nl) = (16 * items.len(), widths.len().div_ceil(2));
        let e = ni + nl;
        let tete = nw + calls + 1;
        let cw = (self.cap_codes.load(Relaxed) as usize).div_ceil(8);
        let mut res = self.ressource(e + tete + cw)?;
        let st = res.st.clone();
        {
            let h = res.h_entree.as_mut_slice().ctx("tampon hote")?;
            for (m, it) in h[..ni].chunks_exact_mut(16).zip(items) {
                m.copy_from_slice(&it.mots());
            }
            for (m, w) in h[ni..e].iter_mut().zip(widths.chunks(2)) {
                *m = u64::from(w[0].to_bits()) | (u64::from(w.get(1).map_or(0, |v| v.to_bits())) << 32);
            }
            h[e..e + tete].fill(0);
        }
        st.memcpy_htod(
            &res.h_entree.as_slice().ctx("tampon hote")?[..e + tete],
            &mut res.tampon.slice_mut(0..e + tete),
        )
        .ctx("envoi des elements")?;
        if !items.is_empty() {
            let (nwi, nci, h) = (nw as i32, calls as i32, hauteur);
            let [cx, cz] = cible.chunk;
            let nlarg = cible.generation_height;
            let entree = res.tampon.slice(0..ni);
            let largeurs = res.tampon.slice(ni..e.max(ni + 1));
            let sortie = res.tampon.slice(e..e + tete);
            let mut l = st.launch_builder(&self.k_trace);
            l.arg(&entree).arg(&largeurs).arg(&self.sinus).arg(&sortie);
            l.arg(&nwi).arg(&nci).arg(&cx).arg(&cz).arg(&cible.min_y).arg(&cible.max_y).arg(&h);
            l.arg(&cible.generation_min_y).arg(&nlarg);
            unsafe { l.launch(config(items.len(), 128)) }.ctx("trace des carvers")?;
        }
        self.stats.lots.fetch_add(1, Relaxed);
        self.stats.elements.fetch_add(items.len() as u64, Relaxed);
        self.stats.ns_envoi.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        Ok(CaveJob {
            res: Some(res),
            cible,
            nw,
            nc: calls,
            e,
            cw,
            hauteur,
            codes: 0,
            attendu: false,
            ok: true,
            region: None,
        })
    }
    /// Attend le lot (une fois) et copie son masque s'il ne l'a pas ete ; `false` si le GPU a echoue.
    pub fn wait(&self, job: &mut CaveJob) -> bool {
        if job.attendu {
            return job.ok;
        }
        let t = Instant::now();
        job.ok = (|| -> Result<(), String> {
            let res = job.res.as_mut().ok_or("lot sans tampons")?;
            if job.codes == 0 {
                let (e, n) = (job.e, job.nw + job.nc + 1);
                res.st
                    .memcpy_dtoh(&res.tampon.slice(e..e + n), &mut res.h_sortie.as_mut_slice().ctx("tampon hote")?[..n])
                    .ctx("retour du masque")?;
            }
            res.st.synchronize().ctx("attente des grottes")
        })()
        .map_err(|e| {
            self.stats.erreurs.fetch_add(1, Relaxed);
            log::error!("GPU grottes : chunk {:?} : {e}", job.cible.chunk);
        })
        .is_ok();
        job.attendu = true;
        job.region = None;
        self.stats.ns_attente.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        job.ok
    }
    /// Le masque d'un lot attendu (mots de CarvingMask), ou None si le GPU l'a signale en erreur ou si
    /// un appel de carver depasse son budget (le CPU leve alors son erreur lui-meme).
    pub fn mask<'a>(&self, job: &'a CaveJob) -> Option<&'a [u64]> {
        if !job.attendu || !job.ok {
            return None;
        }
        let s = job.res.as_ref()?.h_sortie.as_slice().ok()?;
        let (nw, nc) = (job.nw, job.nc);
        if s[nw + nc] != 0 || s[nw..nw + nc].iter().any(|&c| c > 1 << 24) {
            self.stats.erreurs.fetch_add(1, Relaxed);
            return None;
        }
        let m = &s[..nw];
        let n: u64 = m.iter().map(|w| u64::from(w.count_ones())).sum();
        self.stats.cellules.fetch_add(n, Relaxed);
        self.stats.cellules_max.fetch_max(n, Relaxed);
        Some(m)
    }
    /// Les substances des cellules creusees d'un lot attendu, dans l'ordre des bits du masque (codes
    /// de l'aquifere GPU : 1 air, 2 fluide, 3 lave, 8 bloc garde, + 4 mise a jour planifiee), ou None
    /// si l'aquifere GPU n'a pas ete lance ou si le masque a plus de cellules que les codes copies.
    pub fn codes<'a>(&self, job: &'a CaveJob, cellules: usize) -> Option<&'a [i8]> {
        if !job.attendu || !job.ok || job.codes == 0 {
            return None;
        }
        if cellules > job.codes {
            self.cap_codes.fetch_max((cellules as u64).next_power_of_two(), Relaxed);
            return None;
        }
        let tete = job.nw + job.nc + 1;
        let mots = &job.res.as_ref()?.h_sortie.as_slice().ok()?[tete..tete + job.cw];
        // Les codes sont les octets des mots qui suivent la sortie
        let c = unsafe { std::slice::from_raw_parts(mots.as_ptr() as *const i8, cellules) };
        if c.iter().any(|&v| v < 0) {
            return None;
        }
        self.stats.substances.fetch_add(1, Relaxed);
        Some(c)
    }
    /// Rend les tampons d'un lot, apres l'avoir attendu s'il ne l'a pas ete.
    pub fn finish(&self, mut job: CaveJob) {
        self.wait(&mut job);
        if job.ok
            && let Some(r) = job.res.take()
        {
            self.pool.lock().unwrap().push(r);
        }
    }
}

/// L'aquifere d'une region, pour la substance des cellules creusees d'un de ses chunks.
pub(crate) struct AquifereRegion<'a> {
    pub f: &'a CudaFunction,
    pub p: &'a CudaSlice<i32>,
    pub st: &'a CudaSlice<i32>,
    pub grille: [i32; 6],
    pub saut: i32,
    pub mer: i32,
}
impl CaveJob {
    /// Enfile sur le flux du lot, apres sa trace, la substance de ses cellules creusees et la copie
    /// du resultat. `region` garde la memoire de l'aquifere jusqu'a la fin du flux.
    pub(crate) fn enfiler_substance(
        &mut self,
        aq: AquifereRegion<'_>,
        region: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), String> {
        if self.attendu || self.codes != 0 {
            return Ok(());
        }
        let res = self.res.as_mut().ok_or("lot sans tampons")?;
        let st = res.st.clone();
        let (e, tete, cw) = (self.e, self.nw + self.nc + 1, self.cw);
        let cap = 8 * cw;
        let (nw, ci, h) = (self.nw as i32, cap as i32, self.hauteur);
        let (x0, z0) = (self.cible.chunk[0] * 16, self.cible.chunk[1] * 16);
        let g = aq.grille;
        let masque = res.tampon.slice(e..e + self.nw);
        let codes = res.tampon.slice(e + tete..e + tete + cw);
        let mut l = st.launch_builder(aq.f);
        l.arg(&masque).arg(&codes).arg(&nw).arg(&ci).arg(&self.cible.min_y).arg(&h);
        l.arg(&x0).arg(&z0).arg(aq.p).arg(aq.st).arg(&aq.saut);
        l.arg(&g[0]).arg(&g[1]).arg(&g[2]).arg(&g[3]).arg(&g[4]).arg(&g[5]).arg(&aq.mer);
        unsafe { l.launch(config(self.nw.div_ceil(32), 256)) }.ctx("substance des grottes")?;
        st.memcpy_dtoh(
            &res.tampon.slice(e..e + tete + cw),
            &mut res.h_sortie.as_mut_slice().ctx("tampon hote")?[..tete + cw],
        )
        .ctx("retour du masque et des substances")?;
        self.codes = cap;
        self.region = Some(region);
        Ok(())
    }
    pub fn target(&self) -> CaveTarget {
        self.cible
    }
}
