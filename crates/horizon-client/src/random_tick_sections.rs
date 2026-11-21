//! Which sections of the game thread's chunks random-tick
//! (`LevelChunkSection.isRandomlyTicking`), memoized per section.
use crate::{
    blocks::BlockFacts,
    chunk::{Chunk, MIN_Y, Section},
    fast_hash::FxHashMap,
    section_mask::SectionMask,
};
use horizon_world::{
    level::{RandomTickMask, RandomTickMaskUpdate},
    registry::BlockStateId,
};
use std::ops::Range;
use std::rc::Rc;
use std::sync::{Arc, Weak};

/// A value snapshot, in ascending section order, with no per-chunk allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sections {
    mask: SectionMask,
    /// The minimum Y of section index 0 (the chunk's).
    base: i32,
}
impl Default for Sections {
    fn default() -> Self {
        Self {
            mask: SectionMask::EMPTY,
            base: MIN_Y,
        }
    }
}
impl Sections {
    /// The minimum Y of the lowest section in the set from block `min_y` up.
    pub fn from(self, min_y: i32) -> Option<i32> {
        let index = ((min_y - self.base) >> 4).max(0) as usize;
        Sections {
            mask: self.mask.from(index),
            ..self
        }
        .next()
    }
}
impl Iterator for Sections {
    type Item = i32;
    fn next(&mut self) -> Option<i32> {
        let index = self.mask.iter().next()?;
        self.mask.remove(index);
        Some(self.base + index as i32 * 16)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.mask.len();
        (len, Some(len))
    }
}
impl ExactSizeIterator for Sections {}

struct Cached {
    chunk: Weak<Chunk>,
    /// The section each bit of `sections` was read from.
    read: Vec<Weak<Section>>,
    sections: Sections,
    masks: ChunkMasks,
}
/// The [`RandomTickMask`]s of a chunk's ticking sections, each built at its section's first draw
/// (or ahead, [`RandomTickSections::prepare_masks`]) and dropped when the section is read again,
/// held compact one after the other in one table, lowest section first. The random tick loop
/// reads the chunk's entry, then this one allocation, a cache line or a few: an allocation of its
/// own per mask (536 bytes) cost a miss each, behind a miss for the list of them, and a count on
/// each as the loop handed it out. `Rc`, not `Arc`: the tick thread hands a mask out per ticking
/// section per tick, and an atomic count on a mask out of cache stalled the whole random tick
/// loop.
#[derive(Default)]
struct ChunkMasks {
    /// The sections whose mask `table` holds.
    built: SectionMask,
    /// [`RandomTickMask::push_compact`] of each section of `built`, lowest first (`None`: none).
    /// Laid out anew when a mask comes or goes; the masks handed out keep the table they were
    /// handed out from.
    table: Option<Rc<[u64]>>,
}
impl ChunkMasks {
    /// Each section of `built` and where its mask is in `table`, lowest first.
    fn entries(&self) -> impl Iterator<Item = (usize, Range<usize>)> + '_ {
        let table = self.table.as_deref().unwrap_or(&[]);
        self.built.iter().zip(RandomTickMask::ranges(table))
    }
    /// The mask that begins at `at` in `table` (an entry's start).
    fn mask_at(&self, at: usize) -> Option<RandomTickMask> {
        RandomTickMask::in_table(self.table.as_ref()?, at)
    }
    /// The mask of section `index`; `None` when it is not built.
    fn get(&self, index: usize) -> Option<RandomTickMask> {
        if !self.built.contains(index) {
            return None;
        }
        let (_, range) = self.entries().find(|(i, _)| *i == index)?;
        self.mask_at(range.start)
    }
    /// Drops the masks of the sections outside `keep`.
    fn retain(&mut self, keep: SectionMask) {
        if !(self.built & !keep).is_empty() {
            self.lay_out(keep, &[]);
        }
    }
    /// Adds `new`, sections and their compact masks, in place of any built for those sections.
    fn insert(&mut self, new: &[(usize, Vec<u64>)]) {
        self.lay_out(SectionMask::ALL, new);
    }
    /// Lays out a new table: the masks built of the sections of `keep`, and `new`. A mask of
    /// `new` that is not one compact mask is left out (its section stays unbuilt): the masks
    /// after it would be read from the wrong words.
    fn lay_out(&mut self, keep: SectionMask, new: &[(usize, Vec<u64>)]) {
        let old = self.table.take();
        let old = old.as_deref().unwrap_or(&[]);
        let mut masks: Vec<(usize, &[u64])> = (new.iter())
            .filter(|(_, mask)| RandomTickMask::is_compact(mask))
            .map(|(i, mask)| (*i, mask.as_slice()))
            .collect();
        let replaced: SectionMask = masks.iter().map(|&(i, _)| i).collect();
        let kept = keep & !replaced;
        masks.extend(
            (self.built.iter().zip(RandomTickMask::ranges(old)))
                .filter(|(i, _)| kept.contains(*i))
                .map(|(i, range)| (i, &old[range])),
        );
        masks.sort_by_key(|&(i, _)| i);
        masks.dedup_by_key(|(i, _)| *i);
        let mut table = Vec::with_capacity(masks.iter().map(|(_, mask)| mask.len()).sum());
        self.built = SectionMask::EMPTY;
        for (i, mask) in masks {
            self.built.insert(i);
            table.extend_from_slice(mask);
        }
        self.table = (!table.is_empty()).then(|| Rc::from(table));
    }
}
/// Vanilla keeps a count of randomly ticking blocks and fluids per section,
/// updated by every `setBlockState`, and ticks the sections where it is above
/// zero. Here a section's answer is read from its blocks after any edit:
/// resident chunks and sections are immutable behind `Arc` and edited through
/// `Arc::make_mut`, so a weak reference tells an edited or replaced one from
/// the one read, without retaining block data or forcing a deep copy on the
/// next edit, and prevents address reuse. An unchanged chunk costs one
/// pointer comparison; a changed one, a comparison per section and a read of
/// the sections that changed. Clear this cache whenever the gameplay facts
/// change (including resource reload).
#[derive(Default)]
pub struct RandomTickSections {
    chunks: FxHashMap<(i32, i32), Cached>,
}
impl RandomTickSections {
    pub fn clear(&mut self) {
        self.chunks.clear();
    }
    pub fn remove(&mut self, key: &(i32, i32)) {
        self.chunks.remove(key);
    }
    /// The sections of `chunk` holding a randomly ticking block or fluid.
    pub fn get(&mut self, key: (i32, i32), chunk: &Arc<Chunk>, facts: &BlockFacts) -> Sections {
        self.get_with(key, chunk, |id| facts.get(BlockStateId(id)).random_ticks)
    }
    fn get_with(
        &mut self,
        key: (i32, i32),
        chunk: &Arc<Chunk>,
        mut random: impl FnMut(u32) -> bool,
    ) -> Sections {
        Self::refresh(&mut self.chunks, key, chunk, &mut random).sections
    }
    /// The lowest ticking section of `chunk` ([`Self::get`]) from block
    /// `from_y` up: its minimum Y and its mask of the blocks a random tick can
    /// do anything for (`candidate`: `Behaviours::random_tick_candidate`),
    /// built at the section's first tick and kept until it changes (`None` if it cannot be: the
    /// section's draws then read every block, as vanilla's do). Only the section asked for is
    /// built.
    pub fn next_masked(
        &mut self,
        key: (i32, i32),
        chunk: &Arc<Chunk>,
        facts: &BlockFacts,
        candidate: impl FnMut(u32) -> bool,
        from_y: i32,
    ) -> Option<(i32, Option<RandomTickMask>)> {
        let random = |id| facts.get(BlockStateId(id)).random_ticks;
        self.next_masked_with(key, chunk, random, candidate, from_y)
    }
    /// [`Self::next_masked`], `random` telling the states that tick randomly.
    fn next_masked_with(
        &mut self,
        key: (i32, i32),
        chunk: &Arc<Chunk>,
        mut random: impl FnMut(u32) -> bool,
        candidate: impl FnMut(u32) -> bool,
        from_y: i32,
    ) -> Option<(i32, Option<RandomTickMask>)> {
        let cached = Self::refresh(&mut self.chunks, key, chunk, &mut random);
        let y = cached.sections.from(from_y)?;
        let index = ((y - cached.sections.base) >> 4) as usize;
        Some((y, Self::built_mask(cached, chunk, index, candidate)))
    }
    /// The mask of section `index` of `chunk` (`cached` refreshed from it), built now if it is
    /// not yet.
    fn built_mask(
        cached: &mut Cached,
        chunk: &Chunk,
        index: usize,
        candidate: impl FnMut(u32) -> bool,
    ) -> Option<RandomTickMask> {
        if !cached.masks.built.contains(index) {
            let mask = Self::compact_mask(&chunk.sections[index], candidate);
            cached.masks.insert(&[(index, mask)]);
        }
        cached.masks.get(index)
    }
    /// For [`ChunkTicking`]: the ticking sections of `chunk` from block `from_y` up, as
    /// [`Sections::from`] counts them, in ascending order, each with the mask
    /// [`Self::next_masked`] would hand out now (`None` when not built yet: it is built at the
    /// section's first draw, as there). Returns the minimum Y of the chunk's section 0.
    fn read_from(
        &mut self,
        key: (i32, i32),
        chunk: &Arc<Chunk>,
        mut random: impl FnMut(u32) -> bool,
        from_y: i32,
        out: &mut Vec<Ticking>,
    ) -> i32 {
        let cached = Self::refresh(&mut self.chunks, key, chunk, &mut random);
        let base = cached.sections.base;
        let first = ((from_y - base) >> 4).max(0) as usize;
        let mut built = cached.masks.entries().peekable();
        for index in cached.sections.mask.iter().filter(|&i| i >= first) {
            while built.next_if(|(i, _)| *i < index).is_some() {}
            let mask = (built.next_if(|(i, _)| *i == index))
                .and_then(|(_, range)| cached.masks.mask_at(range.start));
            out.push(Ticking {
                y: base + index as i32 * 16,
                index,
                mask,
            });
        }
        base
    }
    /// Reads what [`ChunkTicking::next`] reads first for each of the chunks `keys`: its entry
    /// here and its table of masks, the count and a word in each cache line (of the first eight:
    /// a chunk's masks mostly fit in a few). The loop over the chunks then finds them in the
    /// caches: these reads, which mostly miss, wait for each other here instead of each for the
    /// one before in the loop. Reads only; the sum is for `black_box`.
    pub fn read_ahead(&self, keys: &[(i32, i32)]) -> usize {
        let mut sum = 0usize;
        for key in keys {
            if let Some(cached) = self.chunks.get(key) {
                sum = sum.wrapping_add(cached.chunk.as_ptr() as usize);
                if let Some(table) = &cached.masks.table {
                    sum = sum.wrapping_add(Rc::strong_count(table));
                    // Eight words a line; the last word read may start a line of its own.
                    let head = &table[..table.len().min(64)];
                    for &word in head.iter().step_by(8).chain(head.last()) {
                        sum = sum.wrapping_add(word as usize);
                    }
                }
            }
        }
        sum
    }
    /// For [`ChunkTicking`]: the mask of section `index` of `chunk` as [`Self::next_masked`]
    /// hands it out, built now if it is not yet.
    fn mask_of(
        &mut self,
        key: (i32, i32),
        chunk: &Arc<Chunk>,
        mut random: impl FnMut(u32) -> bool,
        index: usize,
        candidate: impl FnMut(u32) -> bool,
    ) -> Option<RandomTickMask> {
        let cached = Self::refresh(&mut self.chunks, key, chunk, &mut random);
        Self::built_mask(cached, chunk, index, candidate)
    }
    /// Builds, on the tick's helper threads (`crate::parallel`), the masks [`Self::next_masked`]
    /// will build at the first draw in each ticking section of the chunks `keys` names, from their
    /// blocks now (`chunk_of`), so the random tick loop finds them built. A mask only tells the loop
    /// which drawn blocks to read: the draws and the ticks are the same either way, and an edit
    /// before the loop reaches a section drops its mask as usual.
    pub fn prepare_masks(
        &mut self,
        keys: &[(i32, i32)],
        chunk_of: impl Fn((i32, i32)) -> Option<Arc<Chunk>>,
        facts: &BlockFacts,
        candidate: &(impl Fn(u32) -> bool + Sync),
    ) {
        let random = |id| facts.get(BlockStateId(id)).random_ticks;
        self.prepare_masks_with(keys, chunk_of, random, candidate);
    }
    /// [`Self::prepare_masks`], `random` telling the states that tick randomly.
    fn prepare_masks_with(
        &mut self,
        keys: &[(i32, i32)],
        chunk_of: impl Fn((i32, i32)) -> Option<Arc<Chunk>>,
        mut random: impl FnMut(u32) -> bool,
        candidate: &(impl Fn(u32) -> bool + Sync),
    ) {
        let mut todo: Vec<((i32, i32), usize, Arc<Chunk>)> = Vec::new();
        for &key in keys {
            let Some(chunk) = chunk_of(key) else {
                continue;
            };
            let cached = Self::refresh(&mut self.chunks, key, &chunk, &mut random);
            for i in cached.sections.mask.iter() {
                if !cached.masks.built.contains(i) {
                    todo.push((key, i, Arc::clone(&chunk)));
                }
            }
        }
        // Made compact on the helpers too: what comes back is a few words a section, not 65.
        let masks = crate::parallel::par_map(&todo, 128, |part, out| {
            out.extend(
                (part.iter())
                    .map(|(_, i, chunk)| Self::compact_mask(&chunk.sections[*i], candidate)),
            );
        });
        // A chunk's sections are next to each other in `todo`: its table is laid out once.
        let mut done = (todo.into_iter().map(|(key, i, _)| (key, i)))
            .zip(masks)
            .peekable();
        let mut new = Vec::new();
        while let Some(((key, i), mask)) = done.next() {
            new.clear();
            new.push((i, mask));
            while let Some(((_, i), mask)) = done.next_if(|((k, _), _)| *k == key) {
                new.push((i, mask));
            }
            if let Some(cached) = self.chunks.get_mut(&key) {
                cached.masks.insert(&new);
            }
        }
    }
    /// The compact mask ([`RandomTickMask::push_compact`]) of [`Self::candidate_words`].
    fn compact_mask(section: &Section, candidate: impl FnMut(u32) -> bool) -> Vec<u64> {
        let mut mask = Vec::new();
        RandomTickMask::push_compact(&Self::candidate_words(section, candidate), &mut mask);
        mask
    }
    /// The blocks of `section` a random tick can do anything for (`candidate`), a bit each as
    /// [`RandomTickMask::from_words`] takes them.
    fn candidate_words(section: &Section, mut candidate: impl FnMut(u32) -> bool) -> [u64; 64] {
        let blocks = &section.blocks;
        let mut ids = [0u16; 4096];
        if let Some(palette) = blocks.unpack_ids(&mut ids) {
            // ServerLevel.tickChunk's per-state eligibility is constant
            // across a section: resolve it once per palette entry, not
            // once per block when an edited section rebuilds its mask.
            let mut candidates = [false; 256];
            for (i, &state) in palette.iter().enumerate() {
                candidates[i] = candidate(state);
            }
            // A word at a time, in a register: or-ing each block into the array made each wait
            // for the store of the one before.
            let mut words = [0u64; 64];
            for (word, row) in words.iter_mut().zip(ids.as_chunks::<64>().0) {
                *word = (row.iter().enumerate()).fold(0, |w, (j, &id)| {
                    w | u64::from(candidates[usize::from(id)]) << j
                });
            }
            words
        } else if !blocks.is_global() {
            // One state for the whole section: every block or none.
            let hit = candidate(blocks.get(0));
            [if hit { u64::MAX } else { 0 }; 64]
        } else {
            let mut states = [0u32; 4096];
            blocks.unpack(&mut states);
            RandomTickMask::words_of(&states, candidate)
        }
    }
    fn refresh<'a>(
        chunks: &'a mut FxHashMap<(i32, i32), Cached>,
        key: (i32, i32),
        chunk: &Arc<Chunk>,
        random: &mut impl FnMut(u32) -> bool,
    ) -> &'a mut Cached {
        let cached = chunks.entry(key).or_insert_with(|| Cached {
            chunk: Weak::new(),
            read: Vec::new(),
            sections: Sections::default(),
            masks: ChunkMasks::default(),
        });
        if cached.chunk.as_ptr() != Arc::as_ptr(chunk) {
            if cached.read.len() != chunk.sections.len() || cached.sections.base != chunk.min_y() {
                cached.read = chunk.sections.iter().map(|_| Weak::new()).collect();
                cached.masks = ChunkMasks::default();
                cached.sections = Sections {
                    mask: SectionMask::EMPTY,
                    base: chunk.min_y(),
                };
            }
            let mut read_again = SectionMask::EMPTY;
            for (i, (section, read)) in chunk.sections.iter().zip(&mut cached.read).enumerate() {
                if read.as_ptr() == Arc::as_ptr(section) {
                    continue;
                }
                // `isRandomlyTicking`: the blocks themselves, not the palette,
                // which keeps the states no block uses any more.
                read_again.insert(i);
                if section.blocks.has(&mut *random) {
                    cached.sections.mask.insert(i);
                } else {
                    cached.sections.mask.remove(i);
                }
                *read = Arc::downgrade(section);
            }
            // A section read again loses its mask, to be built from its blocks as they are.
            cached.masks.retain(!read_again);
            cached.chunk = Arc::downgrade(chunk);
        }
        cached
    }
}

/// A ticking section as [`ChunkTicking`] read it.
struct Ticking {
    y: i32,
    index: usize,
    /// Until it is handed out; `None` when it was not built at the reading.
    mask: Option<RandomTickMask>,
}

/// One chunk's ticking sections through its random tick
/// (`Behaviours::tick_chunk_random_masked`): read once from [`RandomTickSections`], then answered
/// from that reading for as long as the world stays as it was read (`stamp`: its block revision
/// and its chunk membership), with the answers [`RandomTickSections::next_masked`] gives. Asking
/// that at every section cost two map lookups each time, cache misses a few times per chunk; the
/// masks, handles into the chunk's one table, are now taken at the reading.
#[derive(Default)]
pub struct ChunkTicking {
    key: (i32, i32),
    stamp: Option<(u64, u64)>,
    /// The minimum Y of the chunk's section 0 at the reading.
    base: i32,
    /// The ticking sections from the ask that read them up, in ascending order.
    list: Vec<Ticking>,
    /// The section of the last answer since the reading.
    last: Option<usize>,
}
impl ChunkTicking {
    /// Starts the asks for chunk `key`.
    pub fn start(&mut self, key: (i32, i32)) {
        self.key = key;
        self.stamp = None;
        self.list.clear();
        self.last = None;
    }
    /// The lowest ticking section from block `from` up and its mask: `Keep` when it is the
    /// section of the last answer at the same `stamp`, which is when the old per-ask lookup
    /// said its answer still stood (the random tick asks from rising Ys). `chunk` (the chunk now)
    /// is called only to read the chunk again or to build a mask.
    pub fn next<'c>(
        &mut self,
        sections: &mut RandomTickSections,
        chunk: impl Fn() -> Option<&'c Arc<Chunk>>,
        facts: &BlockFacts,
        candidate: impl FnMut(u32) -> bool,
        from: i32,
        stamp: (u64, u64),
    ) -> Option<(i32, RandomTickMaskUpdate)> {
        let random = |id| facts.get(BlockStateId(id)).random_ticks;
        self.next_with(sections, chunk, random, candidate, from, stamp)
    }
    /// [`Self::next`], `random` telling the states that tick randomly.
    fn next_with<'c>(
        &mut self,
        sections: &mut RandomTickSections,
        chunk: impl Fn() -> Option<&'c Arc<Chunk>>,
        mut random: impl FnMut(u32) -> bool,
        candidate: impl FnMut(u32) -> bool,
        from: i32,
        stamp: (u64, u64),
    ) -> Option<(i32, RandomTickMaskUpdate)> {
        if self.stamp != Some(stamp) {
            self.stamp = Some(stamp);
            self.list.clear();
            self.last = None;
            self.base = sections.read_from(self.key, chunk()?, &mut random, from, &mut self.list);
        }
        let first = ((from - self.base) >> 4).max(0) as usize;
        let section = self.list.iter_mut().find(|s| s.index >= first)?;
        if self.last == Some(section.index) {
            return Some((section.y, RandomTickMaskUpdate::Keep));
        }
        self.last = Some(section.index);
        let mask = match section.mask.take() {
            None => sections.mask_of(self.key, chunk()?, random, section.index, candidate),
            mask => mask,
        };
        Some((section.y, RandomTickMaskUpdate::Replace(mask)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::SECTIONS;
    use crate::chunk::Section;

    #[test]
    fn palette_masks_preserve_positions_and_rng() {
        use horizon_core::position::BlockPos;
        use horizon_world::level::Ticks;

        // Alternating states defeat the old mask builder's consecutive-state
        // shortcut. Include single, local and global palettes and unused IDs.
        for distinct in [1, 2, 256, 300] {
            let states = std::array::from_fn::<_, 4096, _>(|i| (i % distinct) as u32);
            let mut section = Section {
                blocks: Section::from_blocks(&states),
                ..Default::default()
            };
            if distinct == 2 {
                section.set(0, 999);
                section.set(0, states[0]);
            }
            let candidate = |id| id % 3 == 0;
            let reference = RandomTickMask::of(&states, candidate);
            let mut calls = 0;
            let words = RandomTickSections::candidate_words(&section, |id| {
                calls += 1;
                candidate(id)
            });
            assert_eq!(words, RandomTickMask::words_of(&states, candidate));
            let optimized = RandomTickMask::from_words(words);
            assert_eq!(reference, optimized);
            if distinct == 2 {
                assert_eq!(calls, 3); // two live states and one stale palette entry
            }
            let (mut before, mut after) = (Ticks::new(12345), Ticks::new(12345));
            let mut expected = Vec::new();
            let mut actual = Vec::new();
            // Non-ticking sections consume no draws; each ticking section
            // gets three, even if none hits a candidate.
            for (y, ticking) in [(-64, false), (-48, true), (-32, false), (16, true)] {
                if !ticking {
                    continue;
                }
                for _ in 0..3 {
                    let p = before.block_random_pos(-32, y, 48, 15);
                    if reference.may_tick(p) {
                        expected.push(p);
                    }
                    let p = after.block_random_pos(-32, y, 48, 15);
                    if optimized.may_tick(p) {
                        actual.push(p);
                    }
                }
            }
            assert_eq!(expected, actual);
            assert_eq!(before.random_tick_value, after.random_tick_value);
            // Independent wrapping i32 LCG oracle, including coordinate bits.
            let mut value = 12345u32;
            let mut oracle = Vec::new();
            for y in [-48, 16] {
                for _ in 0..3 {
                    value = value.wrapping_mul(3).wrapping_add(1_013_904_223);
                    let p = BlockPos::new(
                        -32 + ((value >> 2) & 15) as i32,
                        y + ((value >> 18) & 15) as i32,
                        48 + ((value >> 10) & 15) as i32,
                    );
                    let i = ((p.y & 15) << 8 | (p.z & 15) << 4 | (p.x & 15)) as usize;
                    if candidate(states[i]) {
                        oracle.push(p);
                    }
                }
            }
            assert_eq!(actual, oracle);
            assert_eq!(after.random_tick_value as u32, value);
        }
    }

    #[test]
    fn a_tall_chunk_ticks_its_sections_near_the_top() {
        let layout = crate::chunk::ColumnLayout::new(-64, 2096).unwrap();
        let mut cache = RandomTickSections::default();
        let mut chunk = Arc::new(Chunk::with_layout(0, 0, layout));
        Arc::make_mut(&mut chunk).set(0, 2020, 0, 1);
        Arc::make_mut(&mut chunk).set(0, 100, 0, 1);
        let sections = cache.get_with((0, 0), &chunk, |id| id == 1);
        assert_eq!(sections.collect::<Vec<_>>(), [96, 2016]);
        assert_eq!(sections.from(200), Some(2016));
        // A vanilla chunk at the same key is read again.
        let short = Arc::new(Chunk::empty(0, 0));
        assert_eq!(cache.get_with((0, 0), &short, |id| id == 1).len(), 0);
    }
    #[test]
    fn edits_replacements_compaction_and_reload() {
        let mut cache = RandomTickSections::default();
        let mut chunk = Arc::new(Chunk::empty(0, 0));
        assert_eq!(cache.get_with((0, 0), &chunk, |id| id == 1).len(), 0);
        cache.get_with((0, 0), &chunk, |_| panic!("unchanged chunk rescanned"));
        let worker = chunk.clone();
        Arc::make_mut(&mut chunk).set(0, MIN_Y, 0, 1);
        assert_eq!(
            cache
                .get_with((0, 0), &chunk, |id| id == 1)
                .collect::<Vec<_>>(),
            [MIN_Y]
        );
        assert_eq!(worker.get(0, MIN_Y, 0), 0);
        // Once the last ticking block is gone the section stops ticking, as
        // vanilla's count drops to zero, though its palette still lists it.
        Arc::make_mut(&mut chunk).set(0, MIN_Y, 0, 0);
        assert!(chunk.sections[0].blocks.maybe_has(|id| id == 1));
        assert_eq!(cache.get_with((0, 0), &chunk, |id| id == 1).len(), 0);
        // Only the edited section is read again (its palette: air, then 1).
        Arc::make_mut(&mut chunk).set(0, MIN_Y + 16, 0, 1);
        let mut asked = Vec::new();
        let sections = cache.get_with((0, 0), &chunk, |id| {
            asked.push(id);
            id == 1
        });
        assert_eq!(
            (sections.collect::<Vec<_>>(), asked),
            (vec![MIN_Y + 16], vec![0, 1])
        );
        Arc::make_mut(&mut chunk).set(0, MIN_Y + 16, 0, 0);
        Arc::make_mut(&mut Arc::make_mut(&mut chunk).sections[1]).compact();
        assert_eq!(cache.get_with((0, 0), &chunk, |id| id == 1).len(), 0);
        Arc::make_mut(&mut chunk).sections[SECTIONS - 1] = Arc::new(Section::default());
        cache.clear(); // resource reload, with changed facts for the same state IDs
        assert_eq!(cache.get_with((0, 0), &chunk, |id| id == 0).len(), SECTIONS);
        cache.remove(&(0, 0));
        assert!(cache.chunks.is_empty());
    }

    #[test]
    fn replacement_at_same_coordinates_and_early_snapshot() {
        let mut cache = RandomTickSections::default();
        let mut chunk = Arc::new(Chunk::empty(-2, 3));
        let before_scheduled = cache.get_with((-2, 3), &chunk, |id| id == 1);
        Arc::make_mut(&mut chunk).set(0, MIN_Y + 16, 0, 1);
        assert_eq!(before_scheduled.len(), 0);
        assert_eq!(
            cache
                .get_with((-2, 3), &chunk, |id| id == 1)
                .collect::<Vec<_>>(),
            [MIN_Y + 16]
        );
        let weak = Arc::downgrade(&chunk);
        drop(chunk);
        assert!(weak.upgrade().is_none()); // cache does not retain unloaded block data
        let replacement = Arc::new(Chunk::empty(-2, 3));
        assert_eq!(cache.get_with((-2, 3), &replacement, |id| id == 1).len(), 0);
    }

    #[test]
    fn global_palette_and_ascending_order() {
        let mut chunk = Chunk::empty(0, 0);
        for i in [0, 9, SECTIONS - 1] {
            Arc::make_mut(&mut chunk.sections[i]).blocks =
                Section::from_blocks(&std::array::from_fn(|j| (j % 300) as u32));
        }
        let chunk = Arc::new(chunk);
        let mut cache = RandomTickSections::default();
        // Global palettes: the states the blocks hold, not the whole ID range.
        assert_eq!(cache.get_with((0, 0), &chunk, |id| id == 40000).len(), 0);
        cache.clear();
        assert_eq!(
            cache
                .get_with((0, 0), &chunk, |id| id == 299)
                .collect::<Vec<_>>(),
            [MIN_Y, MIN_Y + 9 * 16, MIN_Y + (SECTIONS as i32 - 1) * 16]
        );
        let sections = cache.get_with((0, 0), &chunk, |id| id == 299);
        let top = MIN_Y + (SECTIONS as i32 - 1) * 16;
        let from = |y| sections.from(y);
        assert_eq!(
            [
                from(MIN_Y - 16),
                from(MIN_Y),
                from(MIN_Y + 16),
                from(MIN_Y + 9 * 16)
            ],
            [
                Some(MIN_Y),
                Some(MIN_Y),
                Some(MIN_Y + 9 * 16),
                Some(MIN_Y + 9 * 16)
            ]
        );
        assert_eq!(
            [from(top), from(top + 16), from(top + 1000)],
            [Some(top), None, None]
        );
    }

    /// A chunk's table lays its masks out lowest section first, whatever order they come in,
    /// puts a new mask in place of an old one, leaves out what is not one compact mask and drops
    /// what it is told to; a mask handed out keeps the table it came from.
    #[test]
    fn chunk_masks_lay_out_lowest_section_first() {
        let compact = |words: [u64; 64]| {
            let mut mask = Vec::new();
            RandomTickMask::push_compact(&words, &mut mask);
            mask
        };
        // A ramp (word 0 empty, the others partial), every word full, one bit.
        let ramp: [u64; 64] = std::array::from_fn(|k| k as u64 * 0x0101);
        let dot: [u64; 64] = std::array::from_fn(|k| u64::from(k == 7) << 9);
        let (ramp, full, dot) = (compact(ramp), compact([u64::MAX; 64]), compact(dot));
        let built = |masks: &ChunkMasks| masks.built.iter().collect::<Vec<_>>();
        let mut masks = ChunkMasks::default();
        masks.insert(&[(5, ramp.clone()), (2, full.clone())]);
        masks.insert(&[(9, dot.clone())]);
        assert_eq!(built(&masks), [2, 5, 9]);
        let table = [full.clone(), ramp.clone(), dot.clone()].concat();
        assert_eq!(masks.table.as_deref(), Some(&table[..]));
        let held = masks.get(5);
        masks.retain(!SectionMask::single(2));
        assert_eq!(built(&masks), [5, 9]);
        let table = [ramp.clone(), dot.clone()].concat();
        assert_eq!(masks.table.as_deref(), Some(&table[..]));
        assert_eq!(held.as_ref().map(RandomTickMask::compact), Some(&ramp[..]));
        assert_eq!(held, masks.get(5));
        // A new mask for 9 takes the old one's place; [1, 0] is not a compact mask.
        masks.insert(&[(9, full.clone()), (3, vec![1, 0])]);
        assert_eq!(built(&masks), [5, 9]);
        let nine = masks.get(9);
        assert_eq!(nine.as_ref().map(RandomTickMask::compact), Some(&full[..]));
        assert_eq!(masks.get(3), None);
        masks.retain(SectionMask::EMPTY);
        assert!(masks.table.is_none() && masks.built.is_empty());
    }

    /// A chunk's masks share one table and answer for its blocks: built at the first ask, ahead
    /// (`prepare_masks`) or through `ChunkTicking`; after an edit drops one, the others stay as
    /// they were and a mask handed out before keeps answering for the blocks it was built from;
    /// built again, it answers for the new ones.
    #[test]
    fn a_chunks_masks_share_a_table_and_answer_for_its_blocks() {
        use horizon_core::position::BlockPos;

        let ticks = |id: u32| id % 3 == 1;
        let mut seed = 0x2545_f491_u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let scattered: [u32; 4096] = std::array::from_fn(|_| next() % 8);
        let layer: [u32; 4096] = std::array::from_fn(|i| if i >> 8 == 5 { 1 } else { 3 });
        let global: [u32; 4096] = std::array::from_fn(|i| (i % 300) as u32);
        let mut chunk = Chunk::empty(0, 0);
        // A list palette, a flat layer (whole words), one ticking state (every word full), one
        // that does not tick, and a global palette.
        for (i, states) in [
            (1, &scattered),
            (4, &layer),
            (6, &[4; 4096]),
            (7, &[3; 4096]),
            (9, &global),
        ] {
            Arc::make_mut(&mut chunk.sections[i]).blocks = Section::from_blocks(states);
        }
        let mut chunk = Arc::new(chunk);
        let check = |mask: Option<RandomTickMask>, chunk: &Arc<Chunk>, y: i32| {
            let mask = mask.expect("a ticking section has its mask");
            let mut states = [0u32; 4096];
            chunk.sections[((y - MIN_Y) >> 4) as usize].unpack(&mut states);
            for (i, &state) in states.iter().enumerate() {
                let i = i as i32;
                let pos = BlockPos::new(i & 15, y + (i >> 8), (i >> 4) & 15);
                assert_eq!(mask.may_tick(pos), ticks(state), "{pos:?}");
            }
        };
        let walk = |cache: &mut RandomTickSections, chunk: &Arc<Chunk>| {
            let mut from = MIN_Y;
            let mut ys = Vec::new();
            while let Some((y, mask)) = cache.next_masked_with((0, 0), chunk, ticks, ticks, from) {
                check(mask, chunk, y);
                ys.push(y);
                from = y + 16;
            }
            ys
        };
        // The built sections, and whether the table is their compact masks in that order.
        let layout = |cache: &RandomTickSections, chunk: &Arc<Chunk>| {
            let masks = &cache.chunks[&(0, 0)].masks;
            let table: Vec<u64> = (masks.built.iter())
                .flat_map(|i| RandomTickSections::compact_mask(&chunk.sections[i], ticks))
                .collect();
            let built: Vec<_> = masks.built.iter().collect();
            (built, masks.table.as_deref() == Some(&table[..]))
        };
        let ticking = [1, 4, 6, 9].map(|i| MIN_Y + 16 * i);
        let mut cache = RandomTickSections::default();
        assert_eq!(walk(&mut cache, &chunk), ticking);
        assert_eq!(layout(&cache, &chunk), (vec![1, 4, 6, 9], true));
        // The layer's words and the single state's are all full: two summaries each.
        let masks = &cache.chunks[&(0, 0)].masks;
        assert_eq!(masks.get(4).map(|m| m.compact().len()), Some(2));
        assert_eq!(masks.get(6).map(|m| m.compact().len()), Some(2));
        let held = masks.get(4);
        // An edit of the layer drops its mask, and only its.
        let hole = BlockPos::new(3, MIN_Y + 4 * 16 + 5, 7);
        Arc::make_mut(&mut chunk).set(hole.x, hole.y, hole.z, 3);
        assert_eq!(
            cache.get_with((0, 0), &chunk, ticks).collect::<Vec<_>>(),
            ticking
        );
        assert_eq!(layout(&cache, &chunk), (vec![1, 6, 9], true));
        for i in [1, 6, 9] {
            check(
                cache.chunks[&(0, 0)].masks.get(i),
                &chunk,
                MIN_Y + 16 * i as i32,
            );
        }
        assert_eq!(held.as_ref().map(|m| m.may_tick(hole)), Some(true));
        assert_eq!(walk(&mut cache, &chunk), ticking);
        assert_eq!(layout(&cache, &chunk), (vec![1, 4, 6, 9], true));
        assert_eq!(held.map(|m| m.may_tick(hole)), Some(true));
        // Built ahead, all at once; a chunk that is not there is passed over.
        let mut ahead = RandomTickSections::default();
        let chunk_of = |key: (i32, i32)| (key == (0, 0)).then(|| chunk.clone());
        ahead.prepare_masks_with(&[(0, 0), (5, 5)], chunk_of, ticks, &ticks);
        assert_eq!(layout(&ahead, &chunk), (vec![1, 4, 6, 9], true));
        for y in ticking {
            let asked =
                ahead.next_masked_with((0, 0), &chunk, ticks, |_| unreachable!("built ahead"), y);
            check(asked.and_then(|(_, mask)| mask), &chunk, y);
        }
        // And through `ChunkTicking`, from one reading.
        let mut reader = ChunkTicking::default();
        reader.start((0, 0));
        let (mut from, mut ys) = (MIN_Y, Vec::new());
        while let Some((y, update)) =
            reader.next_with(&mut cache, || Some(&chunk), ticks, ticks, from, (0, 0))
        {
            let RandomTickMaskUpdate::Replace(mask) = update else {
                panic!("a section reached at {y} for the first time is handed its mask");
            };
            check(mask, &chunk, y);
            // The same section asked again keeps its mask.
            let again = reader.next_with(&mut cache, || Some(&chunk), ticks, ticks, y, (0, 0));
            assert!(matches!(again, Some((_, RandomTickMaskUpdate::Keep))));
            ys.push(y);
            from = y + 16;
        }
        assert_eq!(ys, ticking);
    }
}

#[cfg(test)]
mod tick_parity {
    use super::*;
    use crate::chunk::SECTIONS;
    use horizon_core::{
        position::{BlockPos, ChunkPos},
        random::{LegacyRandom, RandomSource},
    };
    use horizon_world::level::{LevelAccess, LevelEvent, RandomTickMaskUpdate, Ticks, data};
    use std::{cell::RefCell, path::PathBuf};

    #[derive(Clone)]
    struct Level {
        chunks: FxHashMap<(i32, i32), Arc<Chunk>>,
        random: LegacyRandom,
        reads: RefCell<Vec<BlockPos>>,
        events: Vec<LevelEvent>,
        time: i64,
        speed: i32,
        /// Counts the block changes, as `World::revision` does (for `ChunkTicking`).
        revision: u64,
    }
    impl horizon_world::level::BlockEntityNeighbors for Level {}
    impl LevelAccess for Level {
        fn block_state(&self, pos: BlockPos) -> BlockStateId {
            self.reads.borrow_mut().push(pos);
            self.chunks
                .get(&(pos.x >> 4, pos.z >> 4))
                .map_or(BlockStateId(0), |c| c.block(pos.x, pos.y, pos.z))
        }
        fn set_raw(&mut self, pos: BlockPos, state: BlockStateId) -> Option<BlockStateId> {
            let chunk = self.chunks.get_mut(&(pos.x >> 4, pos.z >> 4))?;
            let previous = Arc::make_mut(chunk).set(pos.x, pos.y, pos.z, state.0);
            if previous != state.0 {
                self.revision += 1;
            }
            Some(BlockStateId(previous))
        }
        fn is_loaded(&self, pos: BlockPos) -> bool {
            self.chunks.contains_key(&(pos.x >> 4, pos.z >> 4))
        }
        fn random(&mut self) -> &mut LegacyRandom {
            &mut self.random
        }
        fn game_time(&self) -> i64 {
            self.time
        }
        fn game_rules(&self) -> horizon_world::level::GameRules {
            horizon_world::level::GameRules {
                random_tick_speed: self.speed,
                ..Default::default()
            }
        }
        fn emit(&mut self, event: LevelEvent) {
            self.events.push(event);
        }
    }

    #[test]
    fn actual_random_ticks_match_a_block_scan() {
        let root = std::env::var_os("HORIZON_RESOURCES")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                let local = manifest
                    .join("../../.local/horizon-client")
                    .join(horizon_core::VANILLA_VERSION);
                if local.join("gameplay-facts.json").exists() {
                    local
                } else {
                    // The branch worktree lives below the main checkout.
                    manifest
                        .join("../../../../.local/horizon-client")
                        .join(horizon_core::VANILLA_VERSION)
                }
            });
        if !root.join("gameplay-facts.json").exists() {
            eprintln!(
                "skipping resource-backed random-tick parity: {}",
                root.display()
            );
            return;
        }
        let registry = data::load_registry(&root).unwrap();
        let (mut facts, behaviours) = BlockFacts::load_pair(&root, &registry).unwrap();
        let grass = registry.default_state("minecraft:grass_block").unwrap();
        let stone = registry.default_state("minecraft:stone").unwrap();
        let chunk = |x| {
            let mut c = Chunk::empty(x, 0);
            c.lit = true;
            // Every sampled block can run real behaviour and change to dirt.
            for i in [4, 7] {
                Arc::make_mut(&mut c.sections[i]).blocks =
                    crate::chunk::Section::from_blocks(&[grass.0; 4096]);
            }
            Arc::new(c)
        };
        let mut original = Level {
            chunks: [
                ((-1, 0), chunk(-1)),
                ((0, 0), chunk(0)),
                ((1, 0), Arc::new(Chunk::empty(1, 0))),
            ]
            .into_iter()
            .collect(),
            random: LegacyRandom::new(42),
            reads: RefCell::default(),
            events: Vec::new(),
            time: 0,
            speed: 3,
            revision: 0,
        };
        // More than 128 ticking sections: prepare_masks must exercise its
        // helper path when HORIZON_PARALLEL_TICK is enabled, not just defer work.
        for x in 2..=65 {
            original.chunks.insert((x, 0), chunk(x));
        }
        Arc::make_mut(original.chunks.get_mut(&(1, 0)).unwrap()).lit = true; // precipitation-only chunk
        let mut cached = original.clone();
        // The masked path reads only the drawn blocks that can tick.
        let mut masked = original.clone();
        // The masked path again, its masks built ahead of each tick (`prepare_masks`).
        let mut prepared = original.clone();
        // The masked path once more, each chunk's sections read once (`ChunkTicking`).
        let mut snapped = original.clone();
        let mut snap_cache = RandomTickSections::default();
        let mut snap_ticking = ChunkTicking::default();
        let mut e = Ticks::new(12345);
        let mut prep_cache = RandomTickSections::default();
        let mut d = Ticks::new(12345);
        let mut cache = RandomTickSections::default();
        let mut mask_cache = RandomTickSections::default();
        let (mut a, mut b, mut c) = (Ticks::new(12345), Ticks::new(12345), Ticks::new(12345));
        for x in -1..=65 {
            a.add_empty_chunk(ChunkPos::new(x, 0)).unwrap();
            b.add_empty_chunk(ChunkPos::new(x, 0)).unwrap();
            c.add_empty_chunk(ChunkPos::new(x, 0)).unwrap();
            d.add_empty_chunk(ChunkPos::new(x, 0)).unwrap();
            e.add_empty_chunk(ChunkPos::new(x, 0)).unwrap();
        }
        for step in 0..128 {
            for level in [
                &mut original,
                &mut cached,
                &mut masked,
                &mut prepared,
                &mut snapped,
            ] {
                level.time = step;
                level.speed = [0, 1, 3, 9][step as usize % 4];
                level.reads.borrow_mut().clear();
                level.events.clear();
                match step {
                    4 => {
                        // The large fixture has exercised every tick speed and
                        // helper-mask invalidation; keep the remaining edit,
                        // unload and reload cases small as before.
                        level.chunks.retain(|&(x, _), _| x <= 1);
                    }
                    12 => {
                        level.chunks.remove(&(-1, 0));
                    }
                    13 => {
                        level.chunks.insert((-1, 0), chunk(-1));
                    }
                    30 => {
                        Arc::make_mut(level.chunks.get_mut(&(0, 0)).unwrap()).sections[7] =
                            Arc::new(crate::chunk::Section::default());
                    }
                    40 | 41 => {
                        Arc::make_mut(level.chunks.get_mut(&(0, 0)).unwrap()).lit = step == 41;
                    }
                    70 => {
                        // Every block of a ticking section replaced: its
                        // palette still lists grass, but nothing ticks there.
                        let c = Arc::make_mut(level.chunks.get_mut(&(-1, 0)).unwrap());
                        for i in 0..4096 {
                            c.set(i & 15, MIN_Y + 4 * 16 + (i >> 8), i >> 4 & 15, stone.0);
                        }
                        assert!(c.sections[4].blocks.maybe_has(|id| id == grass.0));
                    }
                    _ => {}
                }
            }
            if step == 12 {
                cache.remove(&(-1, 0));
                mask_cache.remove(&(-1, 0));
                prep_cache.remove(&(-1, 0));
                snap_cache.remove(&(-1, 0));
            }
            if step == 50 || step == 60 {
                facts.states[grass.0 as usize].random_ticks = step == 60;
                cache.clear();
                mask_cache.clear();
                prep_cache.clear();
                snap_cache.clear();
            }
            // A write between ticks.
            if step == 20 {
                for level in [
                    &mut original,
                    &mut cached,
                    &mut masked,
                    &mut prepared,
                    &mut snapped,
                ] {
                    level.set_raw(BlockPos::new(0, MIN_Y + 16, 0), grass);
                }
            }
            let lit = |level: &Level| {
                let mut keys: Vec<_> = level
                    .chunks
                    .iter()
                    .filter(|(_, c)| c.lit)
                    .map(|(&key, _)| key)
                    .collect();
                keys.sort();
                keys
            };
            let random = |id| facts.get(BlockStateId(id)).random_ticks;
            for (x, z) in lit(&original) {
                // Vanilla's rule, asked as each section is reached: any of its
                // 4,096 blocks ticks randomly.
                let reference = |level: &mut Level, from: i32| {
                    let chunk = &level.chunks[&(x, z)];
                    (from..MIN_Y + SECTIONS as i32 * 16)
                        .step_by(16)
                        .find(|&min_y| {
                            (0..16).any(|y| {
                                (0..16).any(|z| (0..16).any(|x| random(chunk.get(x, min_y + y, z))))
                            })
                        })
                };
                behaviours.tick_chunk_random(&mut original, &mut a, ChunkPos::new(x, z), reference);
            }
            for (x, z) in lit(&cached) {
                let memoized = |level: &mut Level, from| {
                    cache.get((x, z), &level.chunks[&(x, z)], &facts).from(from)
                };
                behaviours.tick_chunk_random(&mut cached, &mut b, ChunkPos::new(x, z), memoized);
            }
            for (x, z) in lit(&masked) {
                let next = |level: &mut Level, from| {
                    // Read at every ask, so an edit of the chunk reads its
                    // section again.
                    let now = level.chunks[&(x, z)].clone();
                    mask_cache
                        .next_masked(
                            (x, z),
                            &now,
                            &facts,
                            |id| behaviours.random_tick_candidate(BlockStateId(id)),
                            from,
                        )
                        .map(|(y, mask)| (y, RandomTickMaskUpdate::Replace(mask)))
                };
                behaviours.tick_chunk_random_masked(&mut masked, &mut c, ChunkPos::new(x, z), next);
            }
            for (x, z) in lit(&snapped) {
                snap_ticking.start((x, z));
                let next = |level: &mut Level, from| {
                    let stamp = (level.revision, 0);
                    let chunks = &level.chunks;
                    snap_ticking.next(
                        &mut snap_cache,
                        || chunks.get(&(x, z)),
                        &facts,
                        |id| behaviours.random_tick_candidate(BlockStateId(id)),
                        from,
                        stamp,
                    )
                };
                behaviours.tick_chunk_random_masked(
                    &mut snapped,
                    &mut e,
                    ChunkPos::new(x, z),
                    next,
                );
            }
            let keys = lit(&prepared);
            let candidate = |id| behaviours.random_tick_candidate(BlockStateId(id));
            prep_cache.prepare_masks(
                &keys,
                |key| prepared.chunks.get(&key).cloned(),
                &facts,
                &candidate,
            );
            for &(x, z) in &keys {
                // Every ticking section's mask is there before the tick asks.
                let now = prepared.chunks[&(x, z)].clone();
                let mut from = MIN_Y;
                while let Some((y, mask)) = prep_cache.next_masked(
                    (x, z),
                    &now,
                    &facts,
                    |_| unreachable!("built ahead"),
                    from,
                ) {
                    assert!(mask.is_some(), "a mask at {y}");
                    from = y + 16;
                }
            }
            for (x, z) in keys {
                let next = |level: &mut Level, from| {
                    let now = level.chunks[&(x, z)].clone();
                    prep_cache
                        .next_masked((x, z), &now, &facts, candidate, from)
                        .map(|(y, mask)| (y, RandomTickMaskUpdate::Replace(mask)))
                };
                behaviours.tick_chunk_random_masked(
                    &mut prepared,
                    &mut d,
                    ChunkPos::new(x, z),
                    next,
                );
            }
            assert_eq!(
                original.events, prepared.events,
                "prepared events at tick {step}"
            );
            assert_eq!(
                a.random_tick_value, d.random_tick_value,
                "prepared position RNG at tick {step}"
            );
            assert_eq!(
                original.random.clone().next_long(),
                prepared.random.next_long(),
                "prepared level RNG at tick {step}"
            );
            assert_eq!(original.reads, cached.reads, "ordered reads at tick {step}");
            assert_eq!(
                original.events, masked.events,
                "masked events at tick {step}"
            );
            assert!(masked.reads.borrow().len() <= original.reads.borrow().len());
            assert_eq!(
                original.events, snapped.events,
                "snapped events at tick {step}"
            );
            assert_eq!(
                *masked.reads.borrow(),
                *snapped.reads.borrow(),
                "snapped reads at tick {step}"
            );
            assert_eq!(
                a.random_tick_value, e.random_tick_value,
                "snapped position RNG at tick {step}"
            );
            assert_eq!(
                original.random.clone().next_long(),
                snapped.random.next_long(),
                "snapped level RNG at tick {step}"
            );
            assert_eq!(
                a.random_tick_value, c.random_tick_value,
                "masked position RNG at tick {step}"
            );
            assert_eq!(
                original.random.clone().next_long(),
                masked.random.next_long(),
                "masked level RNG at tick {step}"
            );
            assert_eq!(original.events, cached.events, "events at tick {step}");
            assert_eq!(
                a.random_tick_value, b.random_tick_value,
                "position RNG at tick {step}"
            );
            assert_eq!(
                original.random.next_long(),
                cached.random.next_long(),
                "level RNG at tick {step}"
            );
        }
        let dirt = registry.default_state("minecraft:dirt").unwrap();
        assert!(
            original
                .chunks
                .values()
                .any(|c| c.sections[4].blocks.maybe_has(|id| id == dirt.0))
        );
        for (key, c) in &original.chunks {
            for (s, t) in c.sections.iter().zip(&cached.chunks[key].sections) {
                assert_eq!(s.blocks, t.blocks);
            }
            for (s, t) in c.sections.iter().zip(&masked.chunks[key].sections) {
                assert_eq!(s.blocks, t.blocks);
            }
            for (s, t) in c.sections.iter().zip(&prepared.chunks[key].sections) {
                assert_eq!(s.blocks, t.blocks);
            }
            for (s, t) in c.sections.iter().zip(&snapped.chunks[key].sections) {
                assert_eq!(s.blocks, t.blocks);
            }
        }
    }
}
