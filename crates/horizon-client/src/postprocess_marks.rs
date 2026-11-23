//! Which resident chunks still hold generation's postprocessing marks
//! (`ProtoChunk.postProcessing`), kept from the chunk map's touched columns.
//!
//! `LevelChunk.postProcessGeneration` runs for a chunk once it is lit and its eight neighbours are
//! resident (`Game::postprocess_chunks`). Asking that of every resident chunk read each chunk's
//! own struct every tick, a cache miss for each of the thousands a busy server holds, to find the
//! few with marks.
use crate::{chunk_map::ChunkMap, fast_hash::FxHashSet};

/// The resident chunks with postprocessing marks, read again only for the columns the chunk map
/// says were touched since the last ask (`ChunkMap::touched_since`). A chunk's marks are written
/// before it joins the map and taken through `ChunkMap::get_mut`, and a chunk comes and goes
/// through `insert` and `remove`: all of them touch its column. Keep one per level, like the map
/// it follows; a cursor of another map reads everything again.
#[derive(Default)]
pub struct PostprocessMarks {
    /// The resident chunks whose `postprocess` was not empty when they were last read.
    holding: FxHashSet<(i32, i32)>,
    /// The `ChunkMap::touch_cursor` `holding` was read at.
    synced: Option<(u64, u64)>,
}

impl PostprocessMarks {
    /// The chunks of `chunks` that hold marks, are lit and have their eight neighbours resident
    /// (`ChunkMap::surrounded`), in position order: what filtering every resident chunk gives,
    /// sorted.
    pub fn ready(&mut self, chunks: &ChunkMap) -> Vec<(i32, i32)> {
        self.sync(chunks);
        let mut ready: Vec<(i32, i32)> = (self.holding.iter().copied())
            .filter(|key| {
                chunks
                    .get(key)
                    .is_some_and(|c| c.lit && !c.postprocess.is_empty())
                    && chunks.surrounded(*key)
            })
            .collect();
        ready.sort_unstable();
        ready
    }
    /// `holding` for `chunks` as they are now.
    fn sync(&mut self, chunks: &ChunkMap) {
        let now = chunks.touch_cursor();
        if self.synced == Some(now) {
            return;
        }
        match self.synced.and_then(|at| chunks.touched_since(at)) {
            Some(touched) => {
                for key in touched {
                    if chunks.get(key).is_some_and(|c| !c.postprocess.is_empty()) {
                        self.holding.insert(*key);
                    } else {
                        self.holding.remove(key);
                    }
                }
            }
            None => {
                self.holding.clear();
                let holding = (chunks.iter())
                    .filter(|(_, chunk)| !chunk.postprocess.is_empty())
                    .map(|(&key, _)| key);
                self.holding.extend(holding);
            }
        }
        self.synced = Some(now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use std::sync::Arc;

    /// The chunks `Game::postprocess_chunks` found by reading every resident chunk: lit, with
    /// marks, and the nine columns around each resident, sorted.
    fn scan(chunks: &ChunkMap) -> Vec<(i32, i32)> {
        let mut ready: Vec<(i32, i32)> = chunks
            .iter()
            .filter(|(_, c)| c.lit && !c.postprocess.is_empty())
            .map(|(k, _)| *k)
            .filter(|&(x, z)| {
                (-1..=1).all(|dz| (-1..=1).all(|dx| chunks.contains_key(&(x + dx, z + dz))))
            })
            .collect();
        ready.sort_unstable();
        ready
    }

    /// Asked at random pauses through chunks coming lit or not, with marks or not, going,
    /// replaced, lit later, their marks taken, whole-map writes, a clear, a swapped map and more
    /// touches than the map keeps between two asks, the ready chunks are those of the scan.
    #[test]
    fn ready_chunks_are_the_ones_a_scan_of_every_chunk_finds() {
        let mut seed = 7u64;
        let mut next = move |n: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % n
        };
        let chunk = |key: (i32, i32), lit: bool, marked: bool| {
            let mut chunk = Chunk::empty(key.0, key.1);
            chunk.lit = lit;
            if marked {
                chunk.postprocess = vec![[key.0 * 16 + 3, 64, key.1 * 16 + 5]];
            }
            Arc::new(chunk)
        };
        let mut chunks = ChunkMap::default();
        let mut other = ChunkMap::default();
        let mut marks = PostprocessMarks::default();
        let (mut asks, mut found) = (0, 0);
        for step in 0..30_000 {
            let key = (next(9) as i32 - 4, next(9) as i32 - 4);
            match next(12) {
                0..=4 => {
                    chunks.insert(key, chunk(key, next(3) != 0, next(2) == 0));
                }
                5 => {
                    chunks.remove(&key);
                }
                6 => {
                    if let Some(c) = chunks.get_mut(&key) {
                        Arc::make_mut(c).lit = true;
                    }
                }
                7 => {
                    // `Game::postprocess_chunks` takes a ready chunk's marks.
                    if let Some(c) = chunks.get_mut(&key) {
                        Arc::make_mut(c).postprocess.clear();
                    }
                }
                8 if next(50) == 0 => {
                    for c in chunks.values_mut() {
                        Arc::make_mut(c).lit = true;
                    }
                }
                9 if next(500) == 0 => chunks.clear(),
                10 if next(300) == 0 => std::mem::swap(&mut chunks, &mut other),
                _ => {}
            }
            // Not asked at every step: touches pile up between asks.
            if next(4) == 0 {
                let expected = scan(&chunks);
                asks += 1;
                found += expected.len();
                assert_eq!(marks.ready(&chunks), expected, "step {step}");
                // Asked again with nothing touched in between: the same.
                assert_eq!(marks.ready(&chunks), expected, "step {step}, asked again");
            }
        }
        assert!(
            found > asks,
            "the run must find ready chunks: {found} in {asks} asks"
        );
        // More touches than the map keeps between two asks: everything is read again.
        let _ = marks.ready(&chunks);
        for i in 0..20_000 {
            let key = (i % 9 - 4, i % 7 - 4);
            if let Some(c) = chunks.get_mut(&key) {
                let marks = if i % 3 == 0 {
                    Vec::new()
                } else {
                    vec![[0, 0, 0]]
                };
                Arc::make_mut(c).postprocess = marks;
            }
        }
        assert_eq!(marks.ready(&chunks), scan(&chunks));
        // A map it never read: everything is read again.
        assert_eq!(marks.ready(&other), scan(&other));
    }
}
