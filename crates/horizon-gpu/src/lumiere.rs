//! Lumiere des chunks sur GPU (`HORIZON_GPU_LIGHT`) : voir LUMIERE.md.
//!
//! Un lot est la lumiere d'un chunk (`light_chunk` de Horizon) : le CPU copie les sections compactes
//! des 3 x 3 chunks (palette et mots) ; le GPU les decode, cherche les sources du ciel, remplit le ciel
//! et les blocs jusqu'au point fixe et met en forme les sections du chunk central. Le tout tourne sur le
//! flux du lot ; un evenement a attente bloquante en signale la fin.
//!
//! Cache des sections (`HORIZON_GPU_LIGHT_CACHE`, voir LUMIERE.md) : une section deja envoyee par un lot
//! reste sur le GPU, et les lots suivants (ses 8 voisins) la lisent la au lieu de la recopier.
use super::{Contexte, contexte, ptx};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

const W: usize = 48;
const WW: usize = W * W;
/// Octets de sortie par section : 8 d'en-tete et 4096 de donnees
const PAS: usize = 4104;
/// Lots en cours au plus (14 Mo de GPU chacun a 24 sections) : au-dela, le CPU eclaire.
const LOTS_MAX: usize = 32;
/// Mots d'un emplacement du cache des sections : les infos de la palette (deux par mot), puis les mots.
const EMPLACEMENT: usize = 1152;
/// Emplacements du cache par defaut : 16 384, soit 151 Mo de GPU.
const EMPLACEMENTS: usize = 16384;
/// Drapeau de klu_etats : une section du cache differe de sa copie en clair (mode test du cache).
const LU_E_CACHE: u32 = 8;

/// La lumiere d'une section du chunk central, comme `Light::from_slice` la range.
#[derive(Clone, Copy, Debug)]
pub enum LightSection<'a> {
    /// Une valeur (ciel << 4 | bloc)
    Uniform(u8),
    /// Ciel variable (nibbles, bloc pair dans le nibble bas) et la lumiere de bloc
    Sky(&'a [u8], u8),
    /// Bloc variable (nibbles) et la lumiere du ciel
    Block(&'a [u8], u8),
    /// Les deux : 4096 octets ciel << 4 | bloc
    Full(&'a [u8]),
}

/// Une section d'un lot pour [`LightGpu::submit_sections`], dans l'ordre (chunk dz * 3 + dx, section).
pub enum LightInput<'a, S> {
    /// Rien a lire : pas de chunk, une section d'air, ou au-dessus du haut du lot.
    Absent,
    /// Un seul etat : son info de lumiere.
    Uniform(u32),
    /// Les etats compacts : `bits` par cellule dans `words`, par la palette `palette` (ids d'etats) ou
    /// par la palette globale (None). `section` les designe pour le cache : leur contenu ne doit pas
    /// changer tant qu'un `Arc` vers lui reste la meme allocation (copie sur ecriture, `Arc::make_mut`).
    Packed {
        section: &'a Arc<S>,
        bits: u8,
        palette: Option<&'a [u32]>,
        words: &'a [u64],
    },
}

#[derive(Default)]
struct StatsLumiere {
    lots: AtomicU64,
    ns_envoi: AtomicU64,
    ns_attente: AtomicU64,
    octets: AtomicU64,
    erreurs: AtomicU64,
    ecarts: AtomicU64,
    replis: AtomicU64,
    trouvees: AtomicU64,
    remplies: AtomicU64,
    en_clair: AtomicU64,
    ecarts_cache: AtomicU64,
}

/// Ou en est un emplacement du cache.
#[derive(Clone)]
enum Etat {
    Libre,
    /// Pris par un lot dont le remplissage n'est pas encore enfile : les autres lots l'envoient en clair.
    Reserve,
    /// Le remplissage est enfile, cet evenement enregistre apres lui sur le flux du lot qui remplit
    /// (un enregistrement plus tard du meme evenement reste apres lui) : les autres lots l'attendent.
    EnCours(Arc<CudaEvent>),
    /// Le lot qui l'a rempli est fini.
    Pret,
}
struct Emplacement {
    /// L'adresse de l'allocation de la section (`Arc::as_ptr`)
    cle: usize,
    /// Garde cette adresse a la section : tant qu'il vit, aucune autre allocation ne la prend, et
    /// `Arc::make_mut` deplace la section au lieu de la changer en place.
    garde: Weak<dyn Any + Send + Sync>,
    liste: bool,
    bits: u8,
    npal: usize,
    mots: usize,
    epingles: u32,
    /// Seconde chance de l'eviction
    vu: bool,
    etat: Etat,
}
impl Emplacement {
    fn libre() -> Self {
        Self {
            cle: 0,
            garde: Weak::<()>::new(),
            liste: false,
            bits: 0,
            npal: 0,
            mots: 0,
            epingles: 0,
            vu: false,
            etat: Etat::Libre,
        }
    }
}
struct EtatCache {
    emplacements: Vec<Emplacement>,
    carte: HashMap<usize, u32>,
    libres: Vec<u32>,
    aiguille: usize,
    /// L'adresse de la table d'infos qui a converti les palettes en cache
    info: usize,
}
impl EtatCache {
    fn new(n: usize) -> Self {
        Self {
            emplacements: (0..n).map(|_| Emplacement::libre()).collect(),
            carte: HashMap::with_capacity(n),
            libres: (0..n as u32).rev().collect(),
            aiguille: 0,
            info: 0,
        }
    }
    /// Un emplacement a remplir : un libre, sinon le premier, a l'aiguille, qu'aucun lot n'epingle et qui
    /// est pret (seconde chance a ceux lus depuis le dernier passage, dont la section vit encore).
    fn prendre(&mut self) -> Option<u32> {
        if let Some(s) = self.libres.pop() {
            return Some(s);
        }
        let n = self.emplacements.len();
        for _ in 0..2 * n {
            let i = self.aiguille;
            self.aiguille = (i + 1) % n;
            let em = &mut self.emplacements[i];
            if em.epingles != 0 || !matches!(em.etat, Etat::Pret | Etat::Libre) {
                continue;
            }
            if em.vu && em.garde.strong_count() > 0 {
                em.vu = false;
                continue;
            }
            let (cle, pret) = (em.cle, matches!(em.etat, Etat::Pret));
            *em = Emplacement::libre();
            if pret && self.carte.get(&cle) == Some(&(i as u32)) {
                self.carte.remove(&cle);
            }
            return Some(i as u32);
        }
        None
    }
    /// La fin d'un lot : ses emplacements desepingles, ceux qu'il a remplis prets (ou oublies si son
    /// flux a echoue).
    fn liberer(&mut self, epingles: &[u32], remplis: &[u32], flux_ok: bool) {
        for &s in remplis {
            let em = &mut self.emplacements[s as usize];
            if flux_ok {
                em.etat = Etat::Pret;
            } else {
                let cle = em.cle;
                em.etat = Etat::Libre;
                em.garde = Weak::<()>::new();
                if self.carte.get(&cle) == Some(&s) {
                    self.carte.remove(&cle);
                }
            }
        }
        for &s in epingles {
            let em = &mut self.emplacements[s as usize];
            em.epingles = em.epingles.saturating_sub(1);
        }
    }
}
struct CacheSections {
    dev: CudaSlice<u64>,
    etat: Mutex<EtatCache>,
    /// Mode test : chaque section trouvee est aussi envoyee en clair, et le GPU compare
    test: bool,
}

pub struct LightGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<cudarc::driver::CudaModule>,
    k_etats: CudaFunction,
    k_ciel: CudaFunction,
    k_remplir: CudaFunction,
    k_sortie: CudaFunction,
    k_cache: CudaFunction,
    info: CudaSlice<u32>,
    masques: CudaSlice<u8>,
    types: i32,
    /// Sections d'une colonne
    nsec: usize,
    pool: Mutex<Vec<RessLumiere>>,
    allouees: AtomicUsize,
    stats: StatsLumiere,
    cache: Option<Arc<CacheSections>>,
    /// Le CACHE de klu_etats sans cache
    vide: CudaSlice<u64>,
}
struct RessLumiere {
    st: Arc<CudaStream>,
    fin: CudaEvent,
    /// Enregistre apres le remplissage du cache par ce lot
    rempli: Arc<CudaEvent>,
    entree: CudaSlice<u64>,
    h_entree: PinnedHostSlice<u64>,
    etats: CudaSlice<u16>,
    ciel: CudaSlice<u8>,
    blocs: CudaSlice<u8>,
    fonds: CudaSlice<i32>,
    f0: CudaSlice<u32>,
    f1: CudaSlice<u32>,
    semis: CudaSlice<u32>,
    sortie: CudaSlice<u8>,
    h_sortie: PinnedHostSlice<u8>,
}
/// La lumiere d'un chunk en cours sur le GPU : `LightGpu::wait`, puis `sections`, puis `finish`.
pub struct LightJob {
    res: Option<RessLumiere>,
    attendu: bool,
    ok: bool,
    /// Le flux du lot a fini sans erreur CUDA (les remplissages du cache sont faits)
    flux_ok: bool,
    cache: Option<Arc<CacheSections>>,
    epingles: Vec<u32>,
    remplis: Vec<u32>,
}
impl LightJob {
    /// Rend au cache les emplacements du lot (une fois, son flux fini).
    fn rendre_cache(&mut self) {
        if let Some(c) = self.cache.take() {
            c.etat
                .lock()
                .unwrap()
                .liberer(&self.epingles, &self.remplis, self.flux_ok);
        }
    }
}
impl Drop for LightJob {
    fn drop(&mut self) {
        if let Some(r) = &self.res
            && !self.attendu
        {
            self.flux_ok = r.st.synchronize().is_ok();
        }
        self.rendre_cache();
    }
}
impl std::fmt::Debug for LightGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LightGpu")
    }
}
impl Drop for LightGpu {
    fn drop(&mut self) {
        if self.stats.lots.load(Relaxed) > 0 {
            eprintln!("{}", self.bilan());
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

/// Ce que `lancer` fait d'un lot : le remplissage du cache a enfiler, les evenements a attendre, les
/// emplacements a rendre a la fin.
#[derive(Default)]
struct Plan {
    attentes: Vec<Arc<CudaEvent>>,
    epingles: Vec<u32>,
    remplis: Vec<u32>,
}

impl LightGpu {
    /// `info` : l'info de lumiere de chaque etat (`emission << 16 | type << 4 | attenuation`) ;
    /// `masks[a * types + b]` : les directions ou les faces des types a et b bloquent la lumiere ;
    /// `sections` : sections d'une colonne.
    pub fn new(info: &[u32], masks: &[u8], types: usize, sections: usize) -> Result<Self, String> {
        if masks.len() != types * types || info.is_empty() || !(1..=254).contains(&sections) {
            return Err("tables de lumiere".into());
        }
        let ctx = contexte()?;
        let (maj, min) = ctx.compute_capability().ctx("capacite du GPU")?;
        let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
        let module = ctx
            .load_module(ptx(include_str!("lumiere.cu"), arch)?)
            .ctx("module de la lumiere")?;
        let f = |n: &str| module.load_function(n).ctx(n);
        let st = ctx.new_stream().ctx("flux CUDA")?;
        let info_d = st.clone_htod(info).ctx("infos de lumiere")?;
        let masques = st.clone_htod(masks).ctx("masques de lumiere")?;
        // HORIZON_GPU_LIGHT_CACHE : 0 sans cache, un nombre d'emplacements, `test` (le cache et sa
        // verification) ; par defaut EMPLACEMENTS.
        let reglage = std::env::var("HORIZON_GPU_LIGHT_CACHE").ok();
        let test = reglage.as_deref() == Some("test");
        let n = match reglage.as_deref() {
            None | Some("1") | Some("test") => EMPLACEMENTS,
            Some(v) => v.parse().unwrap_or(EMPLACEMENTS),
        };
        let cache = if n == 0 {
            None
        } else {
            Some(Arc::new(CacheSections {
                dev: st.alloc_zeros::<u64>(n * EMPLACEMENT).ctx("cache des sections")?,
                etat: Mutex::new(EtatCache::new(n)),
                test,
            }))
        };
        let vide = st.alloc_zeros::<u64>(1).ctx("cache vide")?;
        st.synchronize().ctx("tables de lumiere")?;
        Ok(Self {
            k_etats: f("klu_etats")?,
            k_ciel: f("klu_ciel")?,
            k_remplir: f("klu_remplir")?,
            k_sortie: f("klu_sortie")?,
            k_cache: f("klu_cache")?,
            _module: module,
            ctx,
            info: info_d,
            masques,
            types: types as i32,
            nsec: sections,
            pool: Mutex::new(Vec::new()),
            allouees: AtomicUsize::new(0),
            stats: StatsLumiere::default(),
            cache,
            vide,
        })
    }
    /// Sections d'une colonne : celles d'un chunk que ce GPU eclaire.
    pub fn section_count(&self) -> usize {
        self.nsec
    }
    fn bilan(&self) -> String {
        let s = &self.stats;
        let n = s.lots.load(Relaxed).max(1);
        let (t, r, c) = (
            s.trouvees.load(Relaxed),
            s.remplies.load(Relaxed),
            s.en_clair.load(Relaxed),
        );
        format!(
            "GPU lumiere : {} chunks, {} Ko envoyes, envoi {} us, attente {} us par chunk ; {} lots en cours au plus ; {} en erreur, {} repasses par le CPU, {} ecarts au CPU (mode test) ; cache : {} % des sections trouvees, {} remplies, {} en clair, {} ecarts (mode test)",
            s.lots.load(Relaxed),
            s.octets.load(Relaxed) / n / 1024,
            s.ns_envoi.load(Relaxed) / n / 1000,
            s.ns_attente.load(Relaxed) / n / 1000,
            self.allouees.load(Relaxed),
            s.erreurs.load(Relaxed),
            s.replis.load(Relaxed),
            s.ecarts.load(Relaxed),
            100 * t / (t + r + c).max(1),
            r,
            c,
            s.ecarts_cache.load(Relaxed),
        )
    }
    /// Mots de l'entree d'un lot : en-tete, descripteurs, mots et palettes des 9 x `sections` sections,
    /// puis la liste des remplissages du cache et les copies du mode test.
    pub fn input_words(&self) -> usize {
        3 + 18 * self.nsec + 9 * self.nsec * (1024 + 128) + 36 * self.nsec
    }
    pub fn note_cpu_fallback(&self) {
        self.stats.replis.fetch_add(1, Relaxed);
    }
    pub fn note_mismatch(&self) {
        self.stats.ecarts.fetch_add(1, Relaxed);
    }
    fn ressource(&self) -> Result<RessLumiere, String> {
        if let Some(r) = self.pool.lock().unwrap().pop() {
            return Ok(r);
        }
        if self.allouees.fetch_add(1, Relaxed) >= LOTS_MAX {
            self.allouees.fetch_sub(1, Relaxed);
            return Err("tous les lots sont en cours".into());
        }
        let st = self.ctx.new_stream().ctx("flux CUDA")?;
        use cudarc::driver::sys::CUevent_flags::*;
        let fin = self
            .ctx
            .new_event(Some(CU_EVENT_BLOCKING_SYNC))
            .ctx("evenement")?;
        let rempli = Arc::new(
            self.ctx
                .new_event(Some(CU_EVENT_DISABLE_TIMING))
                .ctx("evenement")?,
        );
        let cellules = WW * 16 * self.nsec;
        let n_entree = self.input_words();
        let n_sortie = 8 + PAS * self.nsec;
        Ok(RessLumiere {
            entree: st.alloc_zeros::<u64>(n_entree).ctx("entree")?,
            h_entree: unsafe { self.ctx.alloc_pinned_with_flags::<u64>(n_entree, 0) }.ctx("tampon hote")?,
            etats: st.alloc_zeros::<u16>(cellules).ctx("etats")?,
            ciel: st.alloc_zeros::<u8>(cellules).ctx("ciel")?,
            blocs: st.alloc_zeros::<u8>(cellules).ctx("blocs")?,
            fonds: st.alloc_zeros::<i32>(WW).ctx("fonds")?,
            f0: st.alloc_zeros::<u32>(cellules).ctx("front")?,
            f1: st.alloc_zeros::<u32>(cellules).ctx("front")?,
            semis: st.alloc_zeros::<u32>(cellules).ctx("semis")?,
            sortie: st.alloc_zeros::<u8>(n_sortie).ctx("sortie")?,
            h_sortie: unsafe { self.ctx.alloc_pinned_with_flags::<u8>(n_sortie, 0) }.ctx("tampon hote")?,
            st,
            fin,
            rempli,
        })
    }
    /// Lance la lumiere d'un chunk : `h` (multiple de 16) cellules de haut, `present` les 9 chunks
    /// (bit dz * 3 + dx). `fill` ecrit, a partir du mot 3 de l'entree, 2 mots par (chunk, section
    /// sous h) puis les mots et palettes qu'ils designent, et rend le nombre de mots ecrits (None :
    /// trop grand). Sans le cache des sections.
    pub fn submit(
        &self,
        h: usize,
        sky: bool,
        present: u16,
        fill: impl FnOnce(&mut [u64]) -> Option<usize>,
    ) -> Result<LightJob, String> {
        if h == 0 || h % 16 != 0 || h > 16 * self.nsec {
            return Err("hauteur de lumiere".into());
        }
        self.lancer(h, sky, present, |e| fill(e).map(|n| (n, 0, 0, 0)), Plan::default())
    }
    /// [`submit`](Self::submit) d'un lot decrit section par section : les sections deja sur le GPU
    /// sont lues dans le cache, les autres envoyees en clair et mises dans le cache pour les lots
    /// suivants. `info` convertit les palettes (ids d'etats) en infos de lumiere ; `inputs` a 9 x h / 16
    /// entrees, chunk dz * 3 + dx puis section de bas en haut.
    pub fn submit_sections<S: Any + Send + Sync>(
        &self,
        h: usize,
        sky: bool,
        present: u16,
        info: &[u32],
        inputs: &[LightInput<'_, S>],
    ) -> Result<LightJob, String> {
        if h == 0 || h % 16 != 0 || h > 16 * self.nsec {
            return Err("hauteur de lumiere".into());
        }
        let nsh = h / 16;
        if inputs.len() != 9 * nsh {
            return Err("sections de lumiere".into());
        }
        #[derive(Clone, Copy)]
        enum Sorte {
            Absente,
            Uniforme(u32),
            EnClair,
            Cache(u32),
            Remplir(u32),
        }
        let mut plan = Plan::default();
        let mut sortes = Vec::with_capacity(inputs.len());
        let test = self.cache.as_ref().is_some_and(|c| c.test);
        {
            let mut etat = self.cache.as_ref().map(|c| c.etat.lock().unwrap());
            let adresse = info.as_ptr() as usize;
            if let Some(e) = etat.as_mut()
                && e.info != adresse
            {
                // Une autre table d'infos : le cache repart de zero, s'il n'est pas lu en ce moment
                if e.emplacements.iter().all(|em| em.epingles == 0) {
                    let n = e.emplacements.len();
                    **e = EtatCache::new(n);
                    e.info = adresse;
                } else {
                    etat = None;
                }
            }
            for input in inputs {
                let sorte = match input {
                    LightInput::Absent => Sorte::Absente,
                    LightInput::Uniform(v) => Sorte::Uniforme(*v),
                    LightInput::Packed {
                        section,
                        bits,
                        palette,
                        words,
                    } => match etat.as_mut() {
                        None => Sorte::EnClair,
                        Some(e) => {
                            let cle = Arc::as_ptr(section) as *const () as usize;
                            let npal = palette.map_or(0, |p| p.len());
                            let taille = npal.div_ceil(2) + words.len();
                            match e.carte.get(&cle).copied() {
                                Some(s) => {
                                    let em = &mut e.emplacements[s as usize];
                                    let meme = em.liste == palette.is_some()
                                        && em.bits == *bits
                                        && em.npal == npal
                                        && em.mots == words.len();
                                    match (&em.etat, meme) {
                                        (Etat::Pret, true) => {
                                            em.epingles += 1;
                                            em.vu = true;
                                            plan.epingles.push(s);
                                            Sorte::Cache(s)
                                        }
                                        (Etat::EnCours(ev), true) => {
                                            let ev = ev.clone();
                                            em.epingles += 1;
                                            em.vu = true;
                                            plan.epingles.push(s);
                                            if !plan.attentes.iter().any(|a| Arc::ptr_eq(a, &ev)) {
                                                plan.attentes.push(ev);
                                            }
                                            Sorte::Cache(s)
                                        }
                                        _ => Sorte::EnClair,
                                    }
                                }
                                None if taille <= EMPLACEMENT && npal <= 0xFFFF => match e.prendre() {
                                    Some(s) => {
                                        e.emplacements[s as usize] = Emplacement {
                                            cle,
                                            garde: Arc::downgrade(section) as Weak<dyn Any + Send + Sync>,
                                            liste: palette.is_some(),
                                            bits: *bits,
                                            npal,
                                            mots: words.len(),
                                            epingles: 1,
                                            vu: false,
                                            etat: Etat::Reserve,
                                        };
                                        e.carte.insert(cle, s);
                                        plan.epingles.push(s);
                                        plan.remplis.push(s);
                                        Sorte::Remplir(s)
                                    }
                                    None => Sorte::EnClair,
                                },
                                None => Sorte::EnClair,
                            }
                        }
                    },
                };
                sortes.push(sorte);
            }
        }
        let (mut trouvees, mut remplies, mut en_clair) = (0u64, 0u64, 0u64);
        for s in &sortes {
            match s {
                Sorte::Cache(_) => trouvees += 1,
                Sorte::Remplir(_) => remplies += 1,
                Sorte::EnClair => en_clair += 1,
                _ => {}
            }
        }
        self.stats.trouvees.fetch_add(trouvees, Relaxed);
        self.stats.remplies.fetch_add(remplies, Relaxed);
        self.stats.en_clair.fetch_add(en_clair, Relaxed);
        // Une section en clair a partir de `at` : les infos de sa palette (deux par mot), puis ses mots ;
        // le debut de la palette et celui des mots.
        let en_clair_a = |e: &mut [u64], at: &mut usize, palette: Option<&[u32]>, words: &[u64]| -> Option<(usize, usize)> {
            let pal = *at;
            if let Some(list) = palette {
                if list.len() > 0xFFFF || *at + list.len().div_ceil(2) + words.len() > e.len() {
                    return None;
                }
                for (j, pair) in list.chunks(2).enumerate() {
                    let a = *info.get(pair[0] as usize)?;
                    let b = match pair.get(1) {
                        Some(&id) => *info.get(id as usize)?,
                        None => 0,
                    };
                    e[pal + j] = u64::from(a) | u64::from(b) << 32;
                }
                *at += list.len().div_ceil(2);
            } else if *at + words.len() > e.len() {
                return None;
            }
            let mots = *at;
            e[mots..mots + words.len()].copy_from_slice(words);
            *at += words.len();
            Some((pal, mots))
        };
        let fill = |e: &mut [u64]| -> Option<(usize, usize, u32, u32)> {
            let mut at = 3 + 18 * nsh;
            let mut copies: Vec<(usize, usize, usize)> = Vec::new();
            let mut verif: Vec<(usize, usize, usize)> = Vec::new();
            for (k, (input, sorte)) in inputs.iter().zip(&sortes).enumerate() {
                let kk = 3 + 2 * k;
                match (input, *sorte) {
                    (_, Sorte::Absente) => {
                        e[kk] = 0;
                        e[kk + 1] = 0;
                    }
                    (_, Sorte::Uniforme(v)) => {
                        e[kk] = 1 | u64::from(v) << 32;
                        e[kk + 1] = 0;
                    }
                    (LightInput::Packed { bits, palette, words, .. }, Sorte::EnClair | Sorte::Remplir(_)) => {
                        let (pal, mots) = en_clair_a(e, &mut at, *palette, words)?;
                        let b = u64::from(*bits);
                        match palette {
                            Some(list) => {
                                e[kk] = 2 | b << 8 | (list.len() as u64) << 16;
                                e[kk + 1] = mots as u64 | (pal as u64) << 32;
                            }
                            None => {
                                e[kk] = 3 | b << 8;
                                e[kk + 1] = mots as u64;
                            }
                        }
                        if let Sorte::Remplir(s) = *sorte {
                            let debut = if palette.is_some() { pal } else { mots };
                            copies.push((debut, at - debut, s as usize * EMPLACEMENT));
                        }
                    }
                    (LightInput::Packed { bits, palette, words, .. }, Sorte::Cache(s)) => {
                        let base = s as usize * EMPLACEMENT;
                        let b = u64::from(*bits);
                        match palette {
                            Some(list) => {
                                let np = list.len();
                                e[kk] = 4 | b << 8 | (np as u64) << 16;
                                e[kk + 1] = (base + np.div_ceil(2)) as u64 | (base as u64) << 32;
                            }
                            None => {
                                e[kk] = 5 | b << 8;
                                e[kk + 1] = base as u64;
                            }
                        }
                        if test {
                            let (pal, mots) = en_clair_a(e, &mut at, *palette, words)?;
                            verif.push((k, mots, pal));
                        }
                    }
                    _ => return None,
                }
            }
            let copies_at = at;
            if at + 2 * copies.len() > e.len() {
                return None;
            }
            for (i, &(src, len, dst)) in copies.iter().enumerate() {
                e[at + 2 * i] = src as u64 | (len as u64) << 32;
                e[at + 2 * i + 1] = dst as u64;
            }
            at += 2 * copies.len();
            let mut verif_at = 0;
            if test {
                let n = 2 * inputs.len();
                if at + n > e.len() {
                    return None;
                }
                verif_at = at;
                e[at..at + n].fill(0);
                for &(k, mots, pal) in &verif {
                    e[at + 2 * k] = mots as u64;
                    e[at + 2 * k + 1] = pal as u64;
                }
                at += n;
            }
            Some((at, copies_at, copies.len() as u32, verif_at as u32))
        };
        let (epingles, remplis) = (plan.epingles.clone(), plan.remplis.clone());
        let lot = self.lancer(h, sky, present, fill, plan);
        if lot.is_err()
            && let Some(c) = &self.cache
        {
            c.etat.lock().unwrap().liberer(&epingles, &remplis, false);
        }
        lot
    }
    /// Envoie un lot : l'entree que `fill` ecrit (mots ecrits, debut des remplissages du cache, leur
    /// nombre, debut des copies du mode test), le remplissage du cache, l'attente des remplissages des
    /// autres lots qu'il lit, puis les quatre noyaux de la lumiere.
    fn lancer(
        &self,
        h: usize,
        sky: bool,
        present: u16,
        fill: impl FnOnce(&mut [u64]) -> Option<(usize, usize, u32, u32)>,
        plan: Plan,
    ) -> Result<LightJob, String> {
        let t = Instant::now();
        let mut res = self.ressource()?;
        let cache = self.cache.as_ref().filter(|_| !plan.epingles.is_empty());
        let r = (|| -> Result<(), String> {
            let (n, copies_at, ncopies, verif) = {
                let e = res.h_entree.as_mut_slice().ctx("tampon hote")?;
                e[0] = h as u64 | (self.nsec as u64) << 16 | u64::from(sky) << 31;
                e[1] = u64::from(present);
                e[2] = 0;
                fill(e).ok_or("entree de lumiere trop grande")?
            };
            let st = res.st.clone();
            st.memcpy_htod(&res.h_entree.as_slice().ctx("tampon hote")?[..n], &mut res.entree.slice_mut(0..n))
                .ctx("envoi des sections")?;
            let dev_cache = self.cache.as_ref().map_or(&self.vide, |c| &c.dev);
            if ncopies > 0 {
                let at = copies_at as u32;
                let mut l = st.launch_builder(&self.k_cache);
                l.arg(&res.entree).arg(dev_cache).arg(&at).arg(&ncopies);
                unsafe { l.launch(config(ncopies as usize, 128)) }.ctx("remplissage du cache")?;
            }
            if !plan.remplis.is_empty()
                && let Some(c) = &self.cache
            {
                res.rempli.record(&st).ctx("evenement du cache")?;
                let mut e = c.etat.lock().unwrap();
                for &s in &plan.remplis {
                    e.emplacements[s as usize].etat = Etat::EnCours(res.rempli.clone());
                }
            }
            for ev in &plan.attentes {
                st.wait(ev).ctx("attente du cache")?;
            }
            let nsh = (h / 16) as i32;
            let cellules = (WW * h) as u32;
            let ninfo = self.info.len() as u32;
            let compteurs = res.entree.slice(2..3);
            {
                let mut l = st.launch_builder(&self.k_etats);
                l.arg(&res.entree).arg(dev_cache).arg(&self.info).arg(&mut res.etats).arg(&mut res.blocs);
                l.arg(&mut res.semis).arg(&compteurs).arg(&cellules).arg(&nsh).arg(&ninfo).arg(&verif);
                unsafe { l.launch(config(9 * nsh as usize, 256)) }.ctx("etats")?;
            }
            {
                let mut l = st.launch_builder(&self.k_ciel);
                l.arg(&res.entree).arg(&res.etats).arg(&mut res.ciel).arg(&mut res.fonds).arg(&self.masques);
                l.arg(&self.types);
                unsafe { l.launch(config(WW.div_ceil(128), 128)) }.ctx("ciel")?;
            }
            {
                let mut l = st.launch_builder(&self.k_remplir);
                l.arg(&res.entree).arg(&res.etats).arg(&mut res.ciel).arg(&mut res.blocs).arg(&res.fonds);
                l.arg(&self.masques).arg(&self.types).arg(&mut res.f0).arg(&mut res.f1).arg(&cellules);
                l.arg(&res.semis).arg(&compteurs);
                unsafe { l.launch(config(1, 1024)) }.ctx("remplissage")?;
            }
            {
                let mut l = st.launch_builder(&self.k_sortie);
                l.arg(&res.entree).arg(&res.ciel).arg(&res.blocs).arg(&mut res.sortie).arg(&compteurs);
                unsafe { l.launch(config(self.nsec, 256)) }.ctx("sortie")?;
            }
            let m = 8 + PAS * self.nsec;
            st.memcpy_dtoh(&res.sortie.slice(0..m), &mut res.h_sortie.as_mut_slice().ctx("tampon hote")?[..m])
                .ctx("retour de la lumiere")?;
            res.fin.record(&st).ctx("evenement")?;
            self.stats.octets.fetch_add(8 * n as u64, Relaxed);
            Ok(())
        })();
        if let Err(e) = r {
            let _ = res.st.synchronize();
            self.pool.lock().unwrap().push(res);
            return Err(e);
        }
        let n = self.stats.lots.fetch_add(1, Relaxed) + 1;
        self.stats.ns_envoi.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        if n.is_power_of_two() && n >= 1024 || n % 65536 == 0 {
            log::info!("{}", self.bilan());
        }
        Ok(LightJob {
            res: Some(res),
            attendu: false,
            ok: true,
            flux_ok: true,
            cache: cache.cloned(),
            epingles: plan.epingles,
            remplis: plan.remplis,
        })
    }
    /// Attend la fin du lot (le thread dort) ; `false` si le GPU a echoue ou signale une erreur.
    pub fn wait(&self, job: &mut LightJob) -> bool {
        if job.attendu {
            return job.ok;
        }
        let t = Instant::now();
        let (flux, drapeaux) = match job.res.as_ref() {
            Some(r) => {
                let flux = r.fin.synchronize().is_ok();
                let drapeaux = flux
                    .then(|| r.h_sortie.as_slice().ok().map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])))
                    .flatten();
                (flux, drapeaux)
            }
            None => (false, None),
        };
        job.flux_ok = flux;
        job.ok = flux && drapeaux == Some(0);
        if drapeaux.is_some_and(|d| d & LU_E_CACHE != 0) {
            self.stats.ecarts_cache.fetch_add(1, Relaxed);
        }
        if !job.ok {
            self.stats.erreurs.fetch_add(1, Relaxed);
        }
        job.attendu = true;
        self.stats.ns_attente.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        job.ok
    }
    /// Les sections du chunk central d'un lot attendu sans erreur, de bas en haut.
    pub fn sections<'a>(&self, job: &'a LightJob) -> Option<Vec<LightSection<'a>>> {
        if !job.attendu || !job.ok {
            return None;
        }
        let s = job.res.as_ref()?.h_sortie.as_slice().ok()?;
        Some(
            (0..self.nsec)
                .map(|i| {
                    let o = &s[8 + i * PAS..8 + (i + 1) * PAS];
                    let (genre, premier, d) = (o[0], o[1], &o[8..]);
                    match genre {
                        1 => LightSection::Sky(&d[..2048], premier & 15),
                        2 => LightSection::Block(&d[..2048], premier >> 4),
                        3 => LightSection::Full(&d[..4096]),
                        _ => LightSection::Uniform(premier),
                    }
                })
                .collect(),
        )
    }
    /// Rend les tampons d'un lot (et ses emplacements du cache).
    pub fn finish(&self, mut job: LightJob) {
        self.wait(&mut job);
        job.rendre_cache();
        if let Some(r) = job.res.take() {
            self.pool.lock().unwrap().push(r);
        }
    }
}
