//! Chunk light on the GPU (`HORIZON_GPU_LIGHT`): see LIGHT.md.
//!
//! A batch is the light of one chunk (Horizon's `light_chunk`): the CPU copies the compact sections
//! of the 3 x 3 chunks (palette and words); the GPU decodes them, finds the sky sources, fills the
//! sky and the blocks up to the fixed point and formats the sections of the central chunk.
//! Everything runs on the batch's stream; an event with blocking sync signals the end.
//!
//! Section cache (`HORIZON_GPU_LIGHT_CACHE`, see LIGHT.md): a section a batch has already sent
//! stays on the GPU, and the next batches (its 8 neighbours) read it there instead of copying it
//! again.
use super::{Context, context, ptx};
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
/// Output bytes per section: 8 of header and 4096 of data
const STRIDE: usize = 4104;
/// Batches in flight at most (14 MB of GPU memory each at 24 sections): beyond that, the CPU
/// lights.
const BATCHES_MAX: usize = 32;
/// Words of a section cache slot: the palette infos (two per word), then the words.
const SLOT_WORDS: usize = 1152;
/// Default cache slots: 16,384, i.e. 151 MB of GPU memory.
const SLOTS: usize = 16384;
/// klt_states flag: a cache section differs from its inline copy (cache test mode).
const LT_E_CACHE: u32 = 8;

/// The light of a section of the central chunk, as `Light::from_slice` stores it.
#[derive(Clone, Copy, Debug)]
pub enum LightSection<'a> {
    /// One value (sky << 4 | block)
    Uniform(u8),
    /// Variable sky (nibbles, even block in the low nibble) and the block light
    Sky(&'a [u8], u8),
    /// Variable block (nibbles) and the sky light
    Block(&'a [u8], u8),
    /// Both: 4096 bytes sky << 4 | block
    Full(&'a [u8]),
}

/// A section of a batch for [`LightGpu::submit_sections`], in order (chunk dz * 3 + dx, section).
pub enum LightInput<'a, S> {
    /// Nothing to read: no chunk, an air section, or above the top of the batch.
    Absent,
    /// A single state: its light info.
    Uniform(u32),
    /// The compact states: `bits` per cell in `words`, through the palette `palette` (state ids) or
    /// the global palette (None). `section` identifies them for the cache: their content must not
    /// change while an `Arc` to it stays the same allocation (copy on write, `Arc::make_mut`).
    Packed {
        section: &'a Arc<S>,
        bits: u8,
        palette: Option<&'a [u32]>,
        words: &'a [u64],
    },
}

#[derive(Default)]
struct LightStats {
    batches: AtomicU64,
    ns_send: AtomicU64,
    ns_wait: AtomicU64,
    octets: AtomicU64,
    errors: AtomicU64,
    mismatches: AtomicU64,
    fallbacks: AtomicU64,
    found: AtomicU64,
    filled: AtomicU64,
    inline: AtomicU64,
    cache_mismatches: AtomicU64,
}

/// Where a cache slot stands.
#[derive(Clone)]
enum State {
    Free,
    /// Taken by a batch whose fill is not enqueued yet: the other batches send the section inline.
    Reserved,
    /// The fill is enqueued, this event recorded after it on the stream of the batch that fills (a
    /// later record of the same event stays after it): the other batches wait for it.
    Filling(Arc<CudaEvent>),
    /// The batch that filled it is finished.
    Ready,
}
struct Slot {
    /// The address of the section's allocation (`Arc::as_ptr`)
    key: usize,
    /// Keeps this address for the section: while it lives, no other allocation takes it, and
    /// `Arc::make_mut` moves the section instead of changing it in place.
    guard: Weak<dyn Any + Send + Sync>,
    list: bool,
    bits: u8,
    npal: usize,
    words: usize,
    pins: u32,
    /// Second chance of the eviction
    seen: bool,
    state: State,
}
impl Slot {
    fn free() -> Self {
        Self {
            key: 0,
            guard: Weak::<()>::new(),
            list: false,
            bits: 0,
            npal: 0,
            words: 0,
            pins: 0,
            seen: false,
            state: State::Free,
        }
    }
}
struct CacheState {
    slots: Vec<Slot>,
    lookup: HashMap<usize, u32>,
    free_slots: Vec<u32>,
    needle: usize,
    /// The address of the info table that converted the palettes in the cache
    info: usize,
}
impl CacheState {
    fn new(n: usize) -> Self {
        Self {
            slots: (0..n).map(|_| Slot::free()).collect(),
            lookup: HashMap::with_capacity(n),
            free_slots: (0..n as u32).rev().collect(),
            needle: 0,
            info: 0,
        }
    }
    /// A slot to fill: a free one, otherwise the first one, from the clock hand, that no batch pins
    /// and that is ready (second chance for those read since the last pass, whose section is still
    /// alive).
    fn take(&mut self) -> Option<u32> {
        if let Some(s) = self.free_slots.pop() {
            return Some(s);
        }
        let n = self.slots.len();
        for _ in 0..2 * n {
            let i = self.needle;
            self.needle = (i + 1) % n;
            let entry = &mut self.slots[i];
            if entry.pins != 0 || !matches!(entry.state, State::Ready | State::Free) {
                continue;
            }
            if entry.seen && entry.guard.strong_count() > 0 {
                entry.seen = false;
                continue;
            }
            let (key, ready) = (entry.key, matches!(entry.state, State::Ready));
            *entry = Slot::free();
            if ready && self.lookup.get(&key) == Some(&(i as u32)) {
                self.lookup.remove(&key);
            }
            return Some(i as u32);
        }
        None
    }
    /// The end of a batch: its slots unpinned, the ones it filled ready (or forgotten if its stream
    /// failed).
    fn release(&mut self, pins: &[u32], filled_slots: &[u32], stream_ok: bool) {
        for &s in filled_slots {
            let entry = &mut self.slots[s as usize];
            if stream_ok {
                entry.state = State::Ready;
            } else {
                let key = entry.key;
                entry.state = State::Free;
                entry.guard = Weak::<()>::new();
                if self.lookup.get(&key) == Some(&s) {
                    self.lookup.remove(&key);
                }
            }
        }
        for &s in pins {
            let entry = &mut self.slots[s as usize];
            entry.pins = entry.pins.saturating_sub(1);
        }
    }
}
struct CacheSections {
    dev: CudaSlice<u64>,
    state: Mutex<CacheState>,
    /// Test mode: each section found is also sent inline, and the GPU compares
    test: bool,
}

pub struct LightGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<cudarc::driver::CudaModule>,
    k_states: CudaFunction,
    k_sky: CudaFunction,
    k_fill: CudaFunction,
    k_output: CudaFunction,
    k_cache: CudaFunction,
    info: CudaSlice<u32>,
    masks_d: CudaSlice<u8>,
    types: i32,
    /// Sections of a column
    nsec: usize,
    pool: Mutex<Vec<LightResources>>,
    allocated: AtomicUsize,
    stats: LightStats,
    cache: Option<Arc<CacheSections>>,
    /// The CACHE of klt_states without a cache
    empty: CudaSlice<u64>,
}
struct LightResources {
    st: Arc<CudaStream>,
    end: CudaEvent,
    /// Recorded after this batch's cache fill
    fill_done: Arc<CudaEvent>,
    input: CudaSlice<u64>,
    h_input: PinnedHostSlice<u64>,
    states: CudaSlice<u16>,
    sky: CudaSlice<u8>,
    blocks: CudaSlice<u8>,
    floors: CudaSlice<i32>,
    f0: CudaSlice<u32>,
    f1: CudaSlice<u32>,
    seeds: CudaSlice<u32>,
    output: CudaSlice<u8>,
    h_output: PinnedHostSlice<u8>,
}
/// The light of a chunk in flight on the GPU: `LightGpu::wait`, then `sections`, then `finish`.
pub struct LightJob {
    res: Option<LightResources>,
    expected: bool,
    ok: bool,
    /// The batch's stream finished without a CUDA error (the cache fills are done)
    stream_ok: bool,
    cache: Option<Arc<CacheSections>>,
    pins: Vec<u32>,
    filled_slots: Vec<u32>,
}
impl LightJob {
    /// Returns the batch's slots to the cache (once, its stream finished).
    fn put_back_cache(&mut self) {
        if let Some(c) = self.cache.take() {
            c.state
                .lock()
                .unwrap()
                .release(&self.pins, &self.filled_slots, self.stream_ok);
        }
    }
}
impl Drop for LightJob {
    fn drop(&mut self) {
        if let Some(r) = &self.res
            && !self.expected
        {
            self.stream_ok = r.st.synchronize().is_ok();
        }
        self.put_back_cache();
    }
}
impl std::fmt::Debug for LightGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LightGpu")
    }
}
impl Drop for LightGpu {
    fn drop(&mut self) {
        if self.stats.batches.load(Relaxed) > 0 {
            eprintln!("{}", self.summary());
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

/// What `launch` does with a batch: the cache fill to enqueue, the events to wait for, the slots to
/// return at the end.
#[derive(Default)]
struct Plan {
    waits: Vec<Arc<CudaEvent>>,
    pins: Vec<u32>,
    filled_slots: Vec<u32>,
}

impl LightGpu {
    /// `info`: the light info of each state (`emission << 16 | type << 4 | attenuation`); `masks[a
    /// * types + b]`: the directions where the faces of types a and b block the light; `sections`:
    /// sections of a column.
    pub fn new(info: &[u32], masks: &[u8], types: usize, sections: usize) -> Result<Self, String> {
        if masks.len() != types * types || info.is_empty() || !(1..=254).contains(&sections) {
            return Err("light tables".into());
        }
        let ctx = context()?;
        let (major, min) = ctx.compute_capability().ctx("GPU compute capability")?;
        let arch: &'static str = Box::leak(format!("sm_{major}{min}").into_boxed_str());
        let module = ctx
            .load_module(ptx(include_str!("light.cu"), arch)?)
            .ctx("light module")?;
        let f = |n: &str| module.load_function(n).ctx(n);
        let st = ctx.new_stream().ctx("CUDA stream")?;
        let info_d = st.clone_htod(info).ctx("light infos")?;
        let masks_d = st.clone_htod(masks).ctx("light masks")?;
        // HORIZON_GPU_LIGHT_CACHE: 0 without a cache, a number of slots, `test` (the cache and its
        // check); SLOTS by default.
        let setting = std::env::var("HORIZON_GPU_LIGHT_CACHE").ok();
        let test = setting.as_deref() == Some("test");
        let n = match setting.as_deref() {
            None | Some("1") | Some("test") => SLOTS,
            Some(v) => v.parse().unwrap_or(SLOTS),
        };
        let cache = if n == 0 {
            None
        } else {
            Some(Arc::new(CacheSections {
                dev: st.alloc_zeros::<u64>(n * SLOT_WORDS).ctx("section cache")?,
                state: Mutex::new(CacheState::new(n)),
                test,
            }))
        };
        let empty = st.alloc_zeros::<u64>(1).ctx("empty cache")?;
        st.synchronize().ctx("light tables")?;
        Ok(Self {
            k_states: f("klt_states")?,
            k_sky: f("klt_sky")?,
            k_fill: f("klt_fill")?,
            k_output: f("klt_output")?,
            k_cache: f("klt_cache")?,
            _module: module,
            ctx,
            info: info_d,
            masks_d,
            types: types as i32,
            nsec: sections,
            pool: Mutex::new(Vec::new()),
            allocated: AtomicUsize::new(0),
            stats: LightStats::default(),
            cache,
            empty,
        })
    }
    /// Sections of a column: those of a chunk this GPU lights.
    pub fn section_count(&self) -> usize {
        self.nsec
    }
    fn summary(&self) -> String {
        let s = &self.stats;
        let n = s.batches.load(Relaxed).max(1);
        let (t, r, c) = (
            s.found.load(Relaxed),
            s.filled.load(Relaxed),
            s.inline.load(Relaxed),
        );
        format!(
            "GPU light: {} chunks, {} KB sent, upload {} us, wait {} us per chunk; {} batches in flight at most; {} failed, {} fell back to the CPU, {} mismatches with the CPU (test mode); cache: {} % of the sections found, {} filled, {} inline, {} mismatches (test mode)",
            s.batches.load(Relaxed),
            s.octets.load(Relaxed) / n / 1024,
            s.ns_send.load(Relaxed) / n / 1000,
            s.ns_wait.load(Relaxed) / n / 1000,
            self.allocated.load(Relaxed),
            s.errors.load(Relaxed),
            s.fallbacks.load(Relaxed),
            s.mismatches.load(Relaxed),
            100 * t / (t + r + c).max(1),
            r,
            c,
            s.cache_mismatches.load(Relaxed),
        )
    }
    /// Words of a batch's input: header, descriptors, words and palettes of the 9 x `sections`
    /// sections, then the list of cache fills and the test mode copies.
    pub fn input_words(&self) -> usize {
        3 + 18 * self.nsec + 9 * self.nsec * (1024 + 128) + 36 * self.nsec
    }
    pub fn note_cpu_fallback(&self) {
        self.stats.fallbacks.fetch_add(1, Relaxed);
    }
    pub fn note_mismatch(&self) {
        self.stats.mismatches.fetch_add(1, Relaxed);
    }
    fn resource(&self) -> Result<LightResources, String> {
        if let Some(r) = self.pool.lock().unwrap().pop() {
            return Ok(r);
        }
        if self.allocated.fetch_add(1, Relaxed) >= BATCHES_MAX {
            self.allocated.fetch_sub(1, Relaxed);
            return Err("all batches are in flight".into());
        }
        let st = self.ctx.new_stream().ctx("CUDA stream")?;
        use cudarc::driver::sys::CUevent_flags::*;
        let end = self
            .ctx
            .new_event(Some(CU_EVENT_BLOCKING_SYNC))
            .ctx("event")?;
        let fill_done = Arc::new(
            self.ctx
                .new_event(Some(CU_EVENT_DISABLE_TIMING))
                .ctx("event")?,
        );
        let cells = WW * 16 * self.nsec;
        let n_input = self.input_words();
        let n_output = 8 + STRIDE * self.nsec;
        Ok(LightResources {
            input: st.alloc_zeros::<u64>(n_input).ctx("input")?,
            h_input: unsafe { self.ctx.alloc_pinned_with_flags::<u64>(n_input, 0) }.ctx("host buffer")?,
            states: st.alloc_zeros::<u16>(cells).ctx("states")?,
            sky: st.alloc_zeros::<u8>(cells).ctx("sky")?,
            blocks: st.alloc_zeros::<u8>(cells).ctx("blocks")?,
            floors: st.alloc_zeros::<i32>(WW).ctx("floors")?,
            f0: st.alloc_zeros::<u32>(cells).ctx("front")?,
            f1: st.alloc_zeros::<u32>(cells).ctx("front")?,
            seeds: st.alloc_zeros::<u32>(cells).ctx("seeds")?,
            output: st.alloc_zeros::<u8>(n_output).ctx("output")?,
            h_output: unsafe { self.ctx.alloc_pinned_with_flags::<u8>(n_output, 0) }.ctx("host buffer")?,
            st,
            end,
            fill_done,
        })
    }
    /// Launches the light of a chunk: `h` (a multiple of 16) cells high, `present` the 9 chunks
    /// (bit dz * 3 + dx). `fill` writes, from word 3 of the input, 2 words per (chunk, section
    /// below h) then the words and palettes they point to, and returns the number of words written
    /// (None: too big). Without the section cache.
    pub fn submit(
        &self,
        h: usize,
        sky: bool,
        present: u16,
        fill: impl FnOnce(&mut [u64]) -> Option<usize>,
    ) -> Result<LightJob, String> {
        if h == 0 || h % 16 != 0 || h > 16 * self.nsec {
            return Err("light height".into());
        }
        self.launch(h, sky, present, |e| fill(e).map(|n| (n, 0, 0, 0)), Plan::default())
    }
    /// [`submit`](Self::submit) of a batch described section by section: the sections already on
    /// the GPU are read from the cache, the others are sent inline and put in the cache for the
    /// next batches. `info` converts the palettes (state ids) to light infos; `inputs` has 9 x h /
    /// 16 entries, chunk dz * 3 + dx then section from bottom to top.
    pub fn submit_sections<S: Any + Send + Sync>(
        &self,
        h: usize,
        sky: bool,
        present: u16,
        info: &[u32],
        inputs: &[LightInput<'_, S>],
    ) -> Result<LightJob, String> {
        if h == 0 || h % 16 != 0 || h > 16 * self.nsec {
            return Err("light height".into());
        }
        let nsh = h / 16;
        if inputs.len() != 9 * nsh {
            return Err("light sections".into());
        }
        #[derive(Clone, Copy)]
        enum InputKind {
            Absent,
            Uniform(u32),
            Inline,
            Cache(u32),
            Fill(u32),
        }
        let mut plan = Plan::default();
        let mut input_kinds = Vec::with_capacity(inputs.len());
        let test = self.cache.as_ref().is_some_and(|c| c.test);
        {
            let mut state = self.cache.as_ref().map(|c| c.state.lock().unwrap());
            let address = info.as_ptr() as usize;
            if let Some(e) = state.as_mut()
                && e.info != address
            {
                // Another info table: the cache starts over, if it is not being read right now
                if e.slots.iter().all(|entry| entry.pins == 0) {
                    let n = e.slots.len();
                    **e = CacheState::new(n);
                    e.info = address;
                } else {
                    state = None;
                }
            }
            for input in inputs {
                let input_kind = match input {
                    LightInput::Absent => InputKind::Absent,
                    LightInput::Uniform(v) => InputKind::Uniform(*v),
                    LightInput::Packed {
                        section,
                        bits,
                        palette,
                        words,
                    } => match state.as_mut() {
                        None => InputKind::Inline,
                        Some(e) => {
                            let key = Arc::as_ptr(section) as *const () as usize;
                            let npal = palette.map_or(0, |p| p.len());
                            let size = npal.div_ceil(2) + words.len();
                            match e.lookup.get(&key).copied() {
                                Some(s) => {
                                    let entry = &mut e.slots[s as usize];
                                    let same = entry.list == palette.is_some()
                                        && entry.bits == *bits
                                        && entry.npal == npal
                                        && entry.words == words.len();
                                    match (&entry.state, same) {
                                        (State::Ready, true) => {
                                            entry.pins += 1;
                                            entry.seen = true;
                                            plan.pins.push(s);
                                            InputKind::Cache(s)
                                        }
                                        (State::Filling(ev), true) => {
                                            let ev = ev.clone();
                                            entry.pins += 1;
                                            entry.seen = true;
                                            plan.pins.push(s);
                                            if !plan.waits.iter().any(|a| Arc::ptr_eq(a, &ev)) {
                                                plan.waits.push(ev);
                                            }
                                            InputKind::Cache(s)
                                        }
                                        _ => InputKind::Inline,
                                    }
                                }
                                None if size <= SLOT_WORDS && npal <= 0xFFFF => match e.take() {
                                    Some(s) => {
                                        e.slots[s as usize] = Slot {
                                            key,
                                            guard: Arc::downgrade(section) as Weak<dyn Any + Send + Sync>,
                                            list: palette.is_some(),
                                            bits: *bits,
                                            npal,
                                            words: words.len(),
                                            pins: 1,
                                            seen: false,
                                            state: State::Reserved,
                                        };
                                        e.lookup.insert(key, s);
                                        plan.pins.push(s);
                                        plan.filled_slots.push(s);
                                        InputKind::Fill(s)
                                    }
                                    None => InputKind::Inline,
                                },
                                None => InputKind::Inline,
                            }
                        }
                    },
                };
                input_kinds.push(input_kind);
            }
        }
        let (mut found, mut filled, mut inline) = (0u64, 0u64, 0u64);
        for s in &input_kinds {
            match s {
                InputKind::Cache(_) => found += 1,
                InputKind::Fill(_) => filled += 1,
                InputKind::Inline => inline += 1,
                _ => {}
            }
        }
        self.stats.found.fetch_add(found, Relaxed);
        self.stats.filled.fetch_add(filled, Relaxed);
        self.stats.inline.fetch_add(inline, Relaxed);
        // A section inline from `at`: the infos of its palette (two per word), then its words; the
        // start of the palette and that of the words.
        let inline_at = |e: &mut [u64], at: &mut usize, palette: Option<&[u32]>, words: &[u64]| -> Option<(usize, usize)> {
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
            let pos = *at;
            e[pos..pos + words.len()].copy_from_slice(words);
            *at += words.len();
            Some((pal, pos))
        };
        let fill = |e: &mut [u64]| -> Option<(usize, usize, u32, u32)> {
            let mut at = 3 + 18 * nsh;
            let mut copies: Vec<(usize, usize, usize)> = Vec::new();
            let mut check: Vec<(usize, usize, usize)> = Vec::new();
            for (k, (input, input_kind)) in inputs.iter().zip(&input_kinds).enumerate() {
                let kk = 3 + 2 * k;
                match (input, *input_kind) {
                    (_, InputKind::Absent) => {
                        e[kk] = 0;
                        e[kk + 1] = 0;
                    }
                    (_, InputKind::Uniform(v)) => {
                        e[kk] = 1 | u64::from(v) << 32;
                        e[kk + 1] = 0;
                    }
                    (LightInput::Packed { bits, palette, words, .. }, InputKind::Inline | InputKind::Fill(_)) => {
                        let (pal, words_at) = inline_at(e, &mut at, *palette, words)?;
                        let b = u64::from(*bits);
                        match palette {
                            Some(list) => {
                                e[kk] = 2 | b << 8 | (list.len() as u64) << 16;
                                e[kk + 1] = words_at as u64 | (pal as u64) << 32;
                            }
                            None => {
                                e[kk] = 3 | b << 8;
                                e[kk + 1] = words_at as u64;
                            }
                        }
                        if let InputKind::Fill(s) = *input_kind {
                            let start = if palette.is_some() { pal } else { words_at };
                            copies.push((start, at - start, s as usize * SLOT_WORDS));
                        }
                    }
                    (LightInput::Packed { bits, palette, words, .. }, InputKind::Cache(s)) => {
                        let base = s as usize * SLOT_WORDS;
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
                            let (pal, words) = inline_at(e, &mut at, *palette, words)?;
                            check.push((k, words, pal));
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
            let mut check_at = 0;
            if test {
                let n = 2 * inputs.len();
                if at + n > e.len() {
                    return None;
                }
                check_at = at;
                e[at..at + n].fill(0);
                for &(k, words, pal) in &check {
                    e[at + 2 * k] = words as u64;
                    e[at + 2 * k + 1] = pal as u64;
                }
                at += n;
            }
            Some((at, copies_at, copies.len() as u32, check_at as u32))
        };
        let (pins, filled_slots) = (plan.pins.clone(), plan.filled_slots.clone());
        let launched = self.launch(h, sky, present, fill, plan);
        if launched.is_err()
            && let Some(c) = &self.cache
        {
            c.state.lock().unwrap().release(&pins, &filled_slots, false);
        }
        launched
    }
    /// Sends a batch: the input `fill` writes (words written, start of the cache fills, their
    /// number, start of the test mode copies), the cache fill, the wait for the fills of the other
    /// batches it reads, then the four light kernels.
    fn launch(
        &self,
        h: usize,
        sky: bool,
        present: u16,
        fill: impl FnOnce(&mut [u64]) -> Option<(usize, usize, u32, u32)>,
        plan: Plan,
    ) -> Result<LightJob, String> {
        let t = Instant::now();
        let mut res = self.resource()?;
        let cache = self.cache.as_ref().filter(|_| !plan.pins.is_empty());
        let r = (|| -> Result<(), String> {
            let (n, copies_at, ncopies, check) = {
                let e = res.h_input.as_mut_slice().ctx("host buffer")?;
                e[0] = h as u64 | (self.nsec as u64) << 16 | u64::from(sky) << 31;
                e[1] = u64::from(present);
                e[2] = 0;
                fill(e).ok_or("light input too big")?
            };
            let st = res.st.clone();
            st.memcpy_htod(&res.h_input.as_slice().ctx("host buffer")?[..n], &mut res.input.slice_mut(0..n))
                .ctx("section upload")?;
            let dev_cache = self.cache.as_ref().map_or(&self.empty, |c| &c.dev);
            if ncopies > 0 {
                let at = copies_at as u32;
                let mut l = st.launch_builder(&self.k_cache);
                l.arg(&res.input).arg(dev_cache).arg(&at).arg(&ncopies);
                unsafe { l.launch(config(ncopies as usize, 128)) }.ctx("cache fill")?;
            }
            if !plan.filled_slots.is_empty()
                && let Some(c) = &self.cache
            {
                res.fill_done.record(&st).ctx("cache event")?;
                let mut e = c.state.lock().unwrap();
                for &s in &plan.filled_slots {
                    e.slots[s as usize].state = State::Filling(res.fill_done.clone());
                }
            }
            for ev in &plan.waits {
                st.wait(ev).ctx("cache wait")?;
            }
            let nsh = (h / 16) as i32;
            let cells = (WW * h) as u32;
            let ninfo = self.info.len() as u32;
            let counters = res.input.slice(2..3);
            {
                let mut l = st.launch_builder(&self.k_states);
                l.arg(&res.input).arg(dev_cache).arg(&self.info).arg(&mut res.states).arg(&mut res.blocks);
                l.arg(&mut res.seeds).arg(&counters).arg(&cells).arg(&nsh).arg(&ninfo).arg(&check);
                unsafe { l.launch(config(9 * nsh as usize, 256)) }.ctx("states")?;
            }
            {
                let mut l = st.launch_builder(&self.k_sky);
                l.arg(&res.input).arg(&res.states).arg(&mut res.sky).arg(&mut res.floors).arg(&self.masks_d);
                l.arg(&self.types);
                unsafe { l.launch(config(WW.div_ceil(128), 128)) }.ctx("sky")?;
            }
            {
                let mut l = st.launch_builder(&self.k_fill);
                l.arg(&res.input).arg(&res.states).arg(&mut res.sky).arg(&mut res.blocks).arg(&res.floors);
                l.arg(&self.masks_d).arg(&self.types).arg(&mut res.f0).arg(&mut res.f1).arg(&cells);
                l.arg(&res.seeds).arg(&counters);
                unsafe { l.launch(config(1, 1024)) }.ctx("filling")?;
            }
            {
                let mut l = st.launch_builder(&self.k_output);
                l.arg(&res.input).arg(&res.sky).arg(&res.blocks).arg(&mut res.output).arg(&counters);
                unsafe { l.launch(config(self.nsec, 256)) }.ctx("output")?;
            }
            let m = 8 + STRIDE * self.nsec;
            st.memcpy_dtoh(&res.output.slice(0..m), &mut res.h_output.as_mut_slice().ctx("host buffer")?[..m])
                .ctx("light readback")?;
            res.end.record(&st).ctx("event")?;
            self.stats.octets.fetch_add(8 * n as u64, Relaxed);
            Ok(())
        })();
        if let Err(e) = r {
            let _ = res.st.synchronize();
            self.pool.lock().unwrap().push(res);
            return Err(e);
        }
        let n = self.stats.batches.fetch_add(1, Relaxed) + 1;
        self.stats.ns_send.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        if n.is_power_of_two() && n >= 1024 || n % 65536 == 0 {
            log::info!("{}", self.summary());
        }
        Ok(LightJob {
            res: Some(res),
            expected: false,
            ok: true,
            stream_ok: true,
            cache: cache.cloned(),
            pins: plan.pins,
            filled_slots: plan.filled_slots,
        })
    }
    /// Waits for the end of the batch (the thread sleeps); `false` if the GPU failed or flagged an
    /// error.
    pub fn wait(&self, job: &mut LightJob) -> bool {
        if job.expected {
            return job.ok;
        }
        let t = Instant::now();
        let (stream, flags) = match job.res.as_ref() {
            Some(r) => {
                let stream = r.end.synchronize().is_ok();
                let flags = stream
                    .then(|| r.h_output.as_slice().ok().map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])))
                    .flatten();
                (stream, flags)
            }
            None => (false, None),
        };
        job.stream_ok = stream;
        job.ok = stream && flags == Some(0);
        if flags.is_some_and(|d| d & LT_E_CACHE != 0) {
            self.stats.cache_mismatches.fetch_add(1, Relaxed);
        }
        if !job.ok {
            self.stats.errors.fetch_add(1, Relaxed);
        }
        job.expected = true;
        self.stats.ns_wait.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        job.ok
    }
    /// The sections of the central chunk of a batch waited for without an error, from bottom to
    /// top.
    pub fn sections<'a>(&self, job: &'a LightJob) -> Option<Vec<LightSection<'a>>> {
        if !job.expected || !job.ok {
            return None;
        }
        let s = job.res.as_ref()?.h_output.as_slice().ok()?;
        Some(
            (0..self.nsec)
                .map(|i| {
                    let o = &s[8 + i * STRIDE..8 + (i + 1) * STRIDE];
                    let (kind, first, d) = (o[0], o[1], &o[8..]);
                    match kind {
                        1 => LightSection::Sky(&d[..2048], first & 15),
                        2 => LightSection::Block(&d[..2048], first >> 4),
                        3 => LightSection::Full(&d[..4096]),
                        _ => LightSection::Uniform(first),
                    }
                })
                .collect(),
        )
    }
    /// Returns the buffers of a batch (and its cache slots).
    pub fn finish(&self, mut job: LightJob) {
        self.wait(&mut job);
        job.put_back_cache();
        if let Some(r) = job.res.take() {
            self.pool.lock().unwrap().push(r);
        }
    }
}
