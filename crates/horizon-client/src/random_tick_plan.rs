//! The random ticks of the parallel tick, planned on the tick's helpers
//! ([`crate::parallel::random_ticks_own_random`]).
//!
//! `ServerChunkCache.tickChunk` draws a chunk's precipitation chances from the level's random and its
//! positions from the level's one position stream (`ServerLevel.getBlockRandomPos`), chunk after
//! chunk, so the loop can only run in order. With the parallel tick on, each chunk draws both from a
//! stream of its own ([`crate::parallel::chunk_tick_random`]): the same odds per block, not the values
//! a vanilla level gives. What a pass then reads to find its ticks, the chunks' sections and the
//! drawn blocks' states, no longer depends on the ticks before it, so [`plan`] reads it for every
//! chunk on the helpers first; the ticks themselves still run on the server thread, in the chunks'
//! order, each from the block as it is then (`Behaviours::run_planned_random_tick`).
use crate::chunk::{Chunk, index};
use horizon_core::position::BlockPos;
use horizon_core::random::RandomSource;
use horizon_world::level::PlannedTick;
use std::sync::Arc;

/// Below this many chunks a pass is planned on the server thread alone.
const MIN_CHUNKS: usize = 64;

/// The random ticks of `chunks`, in their order: for each chunk its precipitation draws that hit,
/// then, section by section up from `low` and below `high`, the drawn blocks whose state
/// `candidate` accepts, in the sections whose palette may hold a state `ticking` accepts (as
/// `LevelChunkSection.isRandomlyTicking`, which a palette can only overstate). `speed` is the
/// `randomTickSpeed` rule. The same plan for any number of the tick's helpers.
pub fn plan(
    chunks: &[((i32, i32), Option<Arc<Chunk>>)],
    game_time: i64,
    speed: i32,
    (low, high): (i32, i32),
    ticking: &(impl Fn(u32) -> bool + Sync),
    candidate: &(impl Fn(u32) -> bool + Sync),
) -> Vec<((i32, i32), PlannedTick)> {
    plan_on(
        crate::parallel::lanes(),
        chunks,
        game_time,
        speed,
        (low, high),
        ticking,
        candidate,
    )
}

/// [`plan`] on `lanes` threads.
fn plan_on(
    lanes: usize,
    chunks: &[((i32, i32), Option<Arc<Chunk>>)],
    game_time: i64,
    speed: i32,
    range: (i32, i32),
    ticking: &(impl Fn(u32) -> bool + Sync),
    candidate: &(impl Fn(u32) -> bool + Sync),
) -> Vec<((i32, i32), PlannedTick)> {
    crate::parallel::par_map_on(chunks, MIN_CHUNKS, lanes, |part, out| {
        for &(key, ref chunk) in part {
            if let Some(chunk) = chunk {
                plan_chunk(chunk, key, game_time, speed, range, ticking, candidate, out);
            }
        }
    })
}

/// One chunk's part of [`plan`], in the order `Behaviours::tick_chunk_random_masked` runs a chunk:
/// the precipitation draws, then the sections from the bottom.
#[allow(clippy::too_many_arguments)]
fn plan_chunk(
    chunk: &Chunk,
    (x, z): (i32, i32),
    game_time: i64,
    speed: i32,
    (low, high): (i32, i32),
    ticking: &impl Fn(u32) -> bool,
    candidate: &impl Fn(u32) -> bool,
    out: &mut Vec<((i32, i32), PlannedTick)>,
) {
    let mut random = crate::parallel::chunk_tick_random(x, z, game_time);
    // `ServerLevel.getBlockRandomPos`, on the chunk's own position stream.
    let mut value = random.next_int();
    let (min_x, min_z) = (x * 16, z * 16);
    let mut draw = |min_y: i32| {
        value = value.wrapping_mul(3).wrapping_add(1_013_904_223);
        let v = value >> 2;
        BlockPos::new(
            min_x + (v & 15),
            min_y + ((v >> 16) & 15),
            min_z + ((v >> 8) & 15),
        )
    };
    for _ in 0..speed {
        if random.next_int_bound(48).unwrap_or(1) == 0 {
            out.push(((x, z), PlannedTick::Precipitation(draw(0))));
        }
    }
    let base = chunk.min_y();
    for (i, section) in chunk.sections.iter().enumerate() {
        let min_y = base + 16 * i as i32;
        if min_y + 16 <= low {
            continue;
        }
        if min_y >= high {
            break;
        }
        let blocks = &section.blocks;
        if !blocks.is_global() && !blocks.maybe_has(ticking) {
            continue;
        }
        for _ in 0..speed {
            let pos = draw(min_y);
            if candidate(blocks.get(index(pos.x, pos.y, pos.z))) {
                out.push(((x, z), PlannedTick::Block(pos)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chunks of stone (state 1) with a ticking state (5) scattered in two sections.
    fn chunks() -> Vec<((i32, i32), Option<Arc<Chunk>>)> {
        (0..120)
            .map(|n| {
                let (x, z) = (n % 12 - 6, n / 12 - 5);
                let mut chunk = Chunk::empty(x, z);
                for y in 0..32 {
                    for lz in 0..16 {
                        for lx in 0..16 {
                            let ticks = (y + lx + lz + n) % 3 == 0 && y > 0;
                            chunk.set(lx, y, lz, if ticks { 5 } else { 1 });
                        }
                    }
                }
                ((x, z), (n % 13 != 0).then(|| Arc::new(chunk)))
            })
            .collect()
    }

    /// The plan is the same on any number of threads, holds only blocks of a ticking state inside
    /// their chunk and the level's range, and keeps the chunks' order.
    #[test]
    fn the_plan_is_the_same_on_any_number_of_threads() {
        let chunks = chunks();
        let (ticking, candidate) = (|v: u32| v == 5, |v: u32| v == 5);
        let one = plan_on(1, &chunks, 1234, 3, (-64, 320), &ticking, &candidate);
        for lanes in [2, 4, 8] {
            let many = plan_on(lanes, &chunks, 1234, 3, (-64, 320), &ticking, &candidate);
            assert_eq!(one, many, "{lanes} lanes");
        }
        let blocks: Vec<_> = (one.iter())
            .filter_map(|&(key, t)| match t {
                PlannedTick::Block(pos) => Some((key, pos)),
                PlannedTick::Precipitation(_) => None,
            })
            .collect();
        assert!(blocks.len() > 100, "{} blocks", blocks.len());
        assert!(
            one.iter()
                .any(|(_, t)| matches!(t, PlannedTick::Precipitation(_)))
        );
        for &((x, z), pos) in &blocks {
            assert_eq!((pos.x >> 4, pos.z >> 4), (x, z));
            assert!(pos.y > 0 && pos.y < 32, "{pos:?}");
            let chunk = chunks
                .iter()
                .find(|(k, _)| *k == (x, z))
                .unwrap()
                .1
                .as_ref()
                .unwrap();
            assert_eq!(chunk.get(pos.x & 15, pos.y, pos.z & 15), 5);
        }
        let order: Vec<_> = chunks.iter().map(|(k, _)| *k).collect();
        let mut last = 0;
        for (key, _) in &one {
            let at = order.iter().position(|k| k == key).unwrap();
            assert!(at >= last);
            last = at;
        }
        // Another tick draws other positions.
        assert_ne!(
            one,
            plan_on(1, &chunks, 1235, 3, (-64, 320), &ticking, &candidate)
        );
    }
}
