//! `LevelChunk.markUnsaved` and `ChunkMap`'s insertion-ordered eager queue.
use crate::fast_hash::{FxHashMap, FxHashSet};
use std::{
    collections::BTreeMap,
    ops::Deref,
    time::{Duration, Instant},
};

type Key = (i32, i32);
const COOLDOWN: Duration = Duration::from_millis(10_000);

/// A chunk's place in `ModifiedChunks` before a tentative write marked it: what a transaction
/// keeps to put the mark back if it aborts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkMark {
    dirty: bool,
    queued: Option<u64>,
}

/// A chunk's entry in the eager queue, with what each visit asks of it: whether the chunk is
/// dirty and when its cooldown ends. They copy `dirty` and `next_save` for the key, and every
/// write to those writes them too, so a visit reads them in the entry instead of looking the key
/// up in a set and a map: the entries waiting out a cooldown or their chunk's light are visited
/// again every tick.
#[derive(Clone, Copy, Debug)]
struct Queued {
    key: Key,
    dirty: bool,
    next_save: Option<Instant>,
}

#[derive(Default)]
pub struct ModifiedChunks {
    dirty: FxHashSet<Key>,
    // Ordered independently of the dirty set: an autosave leaves queue entries
    // in place until the eager loop visits them, including cooled-down ones.
    order: BTreeMap<u64, Queued>,
    queued: FxHashMap<Key, u64>,
    sequence: u64,
    next_save: FxHashMap<Key, Instant>,
}
impl Deref for ModifiedChunks {
    type Target = FxHashSet<Key>;
    fn deref(&self) -> &Self::Target {
        &self.dirty
    }
}
impl IntoIterator for ModifiedChunks {
    type Item = Key;
    type IntoIter = std::collections::hash_set::IntoIter<Key>;
    fn into_iter(self) -> Self::IntoIter {
        self.dirty.into_iter()
    }
}
impl ModifiedChunks {
    pub fn insert(&mut self, key: Key) -> bool {
        let changed = self.dirty.insert(key);
        if changed {
            if let Some(entry) = self.queued_mut(&key) {
                // Saved or cleaned since it was queued, and dirty again: it keeps its place.
                entry.dirty = true;
            } else {
                self.sequence += 1;
                let next_save = self.next_save.get(&key).copied();
                let entry = Queued {
                    key,
                    dirty: true,
                    next_save,
                };
                self.order.insert(self.sequence, entry);
                self.queued.insert(key, self.sequence);
            }
        }
        changed
    }
    /// The queue entry of `key`, if it has one.
    fn queued_mut(&mut self, key: &Key) -> Option<&mut Queued> {
        let sequence = self.queued.get(key)?;
        self.order.get_mut(sequence)
    }
    pub fn extend(&mut self, keys: impl IntoIterator<Item = Key>) {
        for key in keys {
            self.insert(key);
        }
    }
    /// Where `key` stands now, dirty and queued or not, for `restore`.
    pub fn mark(&self, key: Key) -> ChunkMark {
        ChunkMark {
            dirty: self.dirty.contains(&key),
            queued: self.queued.get(&key).copied(),
        }
    }
    /// Puts back what `mark` kept of `key`: a chunk that was clean is clean again, and a queue
    /// entry made since goes. A chunk that was dirty stays as it is now, since only a save
    /// clears a mark, and a save then saw what it saved.
    pub fn restore(&mut self, key: Key, mark: ChunkMark) {
        if !mark.dirty {
            self.remove(&key);
        }
        if mark.queued.is_none()
            && let Some(sequence) = self.queued.remove(&key)
        {
            self.order.remove(&sequence);
        }
    }
    pub fn remove(&mut self, key: &Key) -> bool {
        let removed = self.dirty.remove(key);
        if removed && let Some(entry) = self.queued_mut(key) {
            entry.dirty = false;
        }
        removed
    }
    pub fn drain(&mut self) -> impl Iterator<Item = Key> + '_ {
        // The set is empty once the iterator is gone, taken to the end or not.
        self.order
            .values_mut()
            .for_each(|entry| entry.dirty = false);
        self.dirty.drain()
    }
    /// Like `drain`: queue entries of chunks no longer dirty go at the next eager pass.
    pub fn clear(&mut self) {
        self.dirty.clear();
        self.order
            .values_mut()
            .for_each(|entry| entry.dirty = false);
    }
    /// `ChunkMap.scheduleUnload`: forget the cooldown, including clean chunks.
    pub fn unloaded(&mut self, key: &Key) {
        if self.next_save.remove(key).is_some()
            && let Some(entry) = self.queued_mut(key)
        {
            entry.next_save = None;
        }
    }
    /// `ChunkMap.saveAllChunks(false)`: clear cooldowns, then save all dirty
    /// full chunks without a limit, starting a fresh cooldown for each write.
    pub fn autosaved(&mut self, keys: &[Key], now: Instant) {
        self.next_save.clear();
        self.next_save
            .extend(keys.iter().map(|&key| (key, now + COOLDOWN)));
        for entry in self.order.values_mut() {
            entry.next_save = self.next_save.get(&entry.key).copied();
        }
    }
    /// `ChunkMap.saveChunksEagerly` / `saveChunkIfNeeded`. `available` checks
    /// both haveTime and activeChunkWrites before EACH candidate; `save` takes
    /// an immutable snapshot and queues it. Failed submissions stay dirty.
    pub fn save_eagerly(
        &mut self,
        now: Instant,
        mut available: impl FnMut() -> bool,
        mut loaded: impl FnMut(Key) -> bool,
        mut save: impl FnMut(Key) -> bool,
    ) {
        let mut saved = 0;
        let mut remove = Vec::new();
        for (&sequence, entry) in &self.order {
            if saved == 20 || !available() {
                break;
            }
            // The entry's copies of `dirty` and `next_save` (see `Queued`).
            let key = entry.key;
            if !loaded(key) || !entry.dirty {
                remove.push((sequence, key));
            } else if entry.next_save.is_none_or(|at| now >= at) && save(key) {
                self.dirty.remove(&key);
                self.next_save.insert(key, now + COOLDOWN);
                remove.push((sequence, key));
                saved += 1;
            }
        }
        for (sequence, key) in remove {
            self.order.remove(&sequence);
            self.queued.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_restored_mark_takes_back_the_dirty_flag_and_queue_entry_a_write_made() {
        let mut modified = ModifiedChunks::default();
        modified.insert((1, 1));
        let clean = modified.mark((0, 0));
        let dirty = modified.mark((1, 1));
        modified.insert((0, 0));
        modified.insert((1, 1));
        modified.restore((0, 0), clean);
        modified.restore((1, 1), dirty);
        assert!(!modified.contains(&(0, 0)));
        assert!(modified.contains(&(1, 1)));
        assert_eq!(modified.mark((0, 0)), clean);
        assert_eq!(modified.mark((1, 1)), dirty);
        let mut saved = Vec::new();
        modified.save_eagerly(
            Instant::now(),
            || true,
            |_| true,
            |key| {
                saved.push(key);
                true
            },
        );
        assert_eq!(saved, [(1, 1)]);
    }

    /// The queue as it was before its entries copied the dirty flag and the cooldown: a set and
    /// a map looked up at every visit.
    #[derive(Default)]
    struct Lookups {
        dirty: FxHashSet<Key>,
        order: BTreeMap<u64, Key>,
        queued: FxHashMap<Key, u64>,
        sequence: u64,
        next_save: FxHashMap<Key, Instant>,
    }
    impl Lookups {
        fn insert(&mut self, key: Key) -> bool {
            let changed = self.dirty.insert(key);
            if changed && !self.queued.contains_key(&key) {
                self.sequence += 1;
                self.order.insert(self.sequence, key);
                self.queued.insert(key, self.sequence);
            }
            changed
        }
        fn mark(&self, key: Key) -> ChunkMark {
            ChunkMark {
                dirty: self.dirty.contains(&key),
                queued: self.queued.get(&key).copied(),
            }
        }
        fn restore(&mut self, key: Key, mark: ChunkMark) {
            if !mark.dirty {
                self.dirty.remove(&key);
            }
            if mark.queued.is_none()
                && let Some(sequence) = self.queued.remove(&key)
            {
                self.order.remove(&sequence);
            }
        }
        fn autosaved(&mut self, keys: &[Key], now: Instant) {
            self.next_save.clear();
            self.next_save
                .extend(keys.iter().map(|&key| (key, now + COOLDOWN)));
        }
        fn save_eagerly(
            &mut self,
            now: Instant,
            mut available: impl FnMut() -> bool,
            mut loaded: impl FnMut(Key) -> bool,
            mut save: impl FnMut(Key) -> bool,
        ) {
            let mut saved = 0;
            let mut remove = Vec::new();
            for (&sequence, &key) in &self.order {
                if saved == 20 || !available() {
                    break;
                }
                if !loaded(key) || !self.dirty.contains(&key) {
                    remove.push((sequence, key));
                } else if self.next_save.get(&key).is_none_or(|&at| now >= at) && save(key) {
                    self.dirty.remove(&key);
                    self.next_save.insert(key, now + COOLDOWN);
                    remove.push((sequence, key));
                    saved += 1;
                }
            }
            for (sequence, key) in remove {
                self.order.remove(&sequence);
                self.queued.remove(&key);
            }
        }
    }

    /// One eager pass, logging what it asks in turn (0: the time and the writers, 1: whether a
    /// chunk is loaded, 2: a save): the time or the writers run out at the visit `room` names,
    /// and the save of some chunks fails, as an unlit one's does.
    fn eager_pass(
        run: impl FnOnce(
            &mut dyn FnMut() -> bool,
            &mut dyn FnMut(Key) -> bool,
            &mut dyn FnMut(Key) -> bool,
        ),
        room: u64,
        loaded: &FxHashSet<Key>,
        fails: i32,
    ) -> Vec<(u8, Key)> {
        let log = std::cell::RefCell::new(Vec::new());
        let mut asked = 0;
        run(
            &mut || {
                log.borrow_mut().push((0, (0, 0)));
                asked += 1;
                asked <= room
            },
            &mut |key| {
                log.borrow_mut().push((1, key));
                loaded.contains(&key)
            },
            &mut |key| {
                log.borrow_mut().push((2, key));
                (key.0 * 7 + key.1).rem_euclid(5) != fails
            },
        );
        log.into_inner()
    }

    #[test]
    fn copied_flags_visit_as_the_lookups_did() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let (mut copied, mut looked) = (ModifiedChunks::default(), Lookups::default());
        let mut loaded: FxHashSet<Key> = FxHashSet::default();
        loaded.extend((0..12).flat_map(|x| (0..12).map(move |z| (x, z))));
        let mut now = Instant::now();
        let (mut asked, mut saves) = (0, 0);
        for step in 0..20_000 {
            let key = (rand(12) as i32, rand(12) as i32);
            let (what, rare) = (rand(32), rand(40) == 0);
            match what {
                0..=13 => assert_eq!(copied.insert(key), looked.insert(key), "step {step}"),
                14 => assert_eq!(
                    copied.remove(&key),
                    looked.dirty.remove(&key),
                    "step {step}"
                ),
                // A transaction marks a chunk, writes it and one more, and aborts.
                15 => {
                    let mark = copied.mark(key);
                    assert_eq!(mark, looked.mark(key), "step {step}");
                    let other = (key.1, key.0);
                    for write in [key, other] {
                        copied.insert(write);
                        looked.insert(write);
                    }
                    copied.restore(key, mark);
                    looked.restore(key, mark);
                }
                16 => {
                    copied.unloaded(&key);
                    looked.next_save.remove(&key);
                }
                // Drained to the end or not, the set is empty after.
                17 if rare => {
                    let taken = rand(4) as usize;
                    let _ = copied.drain().take(taken).count();
                    let _ = looked.dirty.drain().take(taken).count();
                }
                18 if rare => {
                    copied.clear();
                    looked.dirty.clear();
                }
                19 if rare => {
                    let mut keys: Vec<Key> = copied.dirty.iter().copied().collect();
                    keys.sort_unstable();
                    keys.truncate(rand(keys.len() as u64 + 1) as usize);
                    copied.autosaved(&keys, now);
                    looked.autosaved(&keys, now);
                }
                20 => {
                    if !loaded.remove(&key) {
                        loaded.insert(key);
                    }
                }
                // An eager pass, with time going on a twentieth of a cooldown on average: the
                // chunks saved stay cooling down for a score of passes.
                29.. => {
                    now += Duration::from_millis(rand(1000));
                    let (room, fails) = (rand(40), rand(5) as i32);
                    let a = eager_pass(
                        |t, l, s| copied.save_eagerly(now, t, l, s),
                        room,
                        &loaded,
                        fails,
                    );
                    let b = eager_pass(
                        |t, l, s| looked.save_eagerly(now, t, l, s),
                        room,
                        &loaded,
                        fails,
                    );
                    assert_eq!(a, b, "step {step}");
                    asked += a.iter().filter(|(what, _)| *what == 1).count();
                    saves += a.iter().filter(|(what, _)| *what == 2).count();
                }
                _ => {}
            }
            assert_eq!(copied.dirty, looked.dirty, "step {step}");
            assert_eq!(copied.queued, looked.queued, "step {step}");
            assert_eq!(copied.next_save, looked.next_save, "step {step}");
            let order: Vec<(u64, Key)> = copied.order.iter().map(|(&s, e)| (s, e.key)).collect();
            let expected: Vec<(u64, Key)> = looked.order.iter().map(|(&s, &k)| (s, k)).collect();
            assert_eq!(order, expected, "step {step}");
            for entry in copied.order.values() {
                assert_eq!(
                    entry.dirty,
                    copied.dirty.contains(&entry.key),
                    "step {step}"
                );
                assert_eq!(entry.next_save, copied.next_save.get(&entry.key).copied());
            }
        }
        assert!(
            asked > 10_000 && saves > 2_000,
            "{asked} visits, {saves} saves"
        );
    }
}
