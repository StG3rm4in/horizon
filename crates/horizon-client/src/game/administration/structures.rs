//! Which structures contain a position (`StructureManager.getStructureAt`
//! and `getStructureWithPieceAt`), from the chunk's structure references and
//! the generator's cached starts.
use super::Game;
use horizon_core::bounding_box::BoundingBox;
use horizon_world::worldgen::structure::start::{Piece, StructureStart};
use std::sync::Arc;

/// A structure a position is in, by either of vanilla's two reads: `getStructureAt` tests its
/// start's box, `getStructureWithPieceAt` its pieces as placed, with no start box gate. A
/// piece that settled below the box it was made in (a shipwreck) holds blocks outside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Reach {
    pub name: String,
    /// The start's box holds the position.
    pub in_box: bool,
    /// One of its pieces, as placed, holds the position.
    pub in_piece: bool,
}

impl Game<'_> {
    /// Structures reaching `pos` by either read ([`Reach`]): (id, inside a piece as placed).
    pub(super) fn structures_reaching(&self, pos: [i32; 3]) -> Vec<(String, bool)> {
        self.structures_reaching_where(pos, &|_| true)
    }
    /// The same, for the structures whose id `wanted` takes: the others'
    /// starts aren't looked up.
    pub(super) fn structures_reaching_where(
        &self,
        pos: [i32; 3],
        wanted: &dyn Fn(&str) -> bool,
    ) -> Vec<(String, bool)> {
        structures_reaching_in(&self.world, &|| self.end.locator(), pos, wanted)
    }
    /// The structures with a piece containing `pos` (`getStructureWithPieceAt`).
    pub(super) fn structures_at(&self, pos: [i32; 3]) -> Vec<String> {
        self.structures_reaching(pos)
            .into_iter()
            .filter(|(_, piece)| *piece)
            .map(|(n, _)| n)
            .collect()
    }
}

type Locator<'a> = &'a dyn Fn() -> Option<Arc<horizon_client::vanillagen::VanillaTerrain>>;

/// [`Game::structures_reaching_where`] over what it reads: [`reaches_in`] as (id, inside a
/// piece as placed).
pub(super) fn structures_reaching_in(
    world: &horizon_client::world::World,
    locator: Locator<'_>,
    pos: [i32; 3],
    wanted: &dyn Fn(&str) -> bool,
) -> Vec<(String, bool)> {
    reaches_in(world, locator, pos, wanted)
        .into_iter()
        .map(|reach| (reach.name, reach.in_piece))
        .collect()
}

/// The structures reaching `pos` by either read, from the world's chunks and, asked only when a
/// chunk refers to a structure, the terrain that locates the starts. A start the generator no
/// longer holds is built again, as vanilla's `getChunk(STRUCTURE_STARTS)` loads it.
pub(super) fn reaches_in(
    world: &horizon_client::world::World,
    locator: Locator<'_>,
    pos: [i32; 3],
    wanted: &dyn Fn(&str) -> bool,
) -> Vec<Reach> {
    let Some(chunk) = world.chunks.get(&(pos[0] >> 4, pos[2] >> 4)) else {
        return Vec::new();
    };
    if chunk.structure_refs.is_empty() {
        return Vec::new();
    }
    let Some(locator) = locator() else {
        return Vec::new();
    };
    reaching(chunk, pos, wanted, &|x, z| Some(locator.starts_at(x, z))).unwrap_or_default()
}

/// The starts that a chunk's references name and that can hold one of its positions, in the order
/// [`reaches_in`] reads them, for the references `wanted` takes: [`reaches_in`] for every position
/// of the chunk, the starts looked up once. Natural spawning asks for the structures at each of its
/// attempts, and the attempts in a chunk ask about the same references.
///
/// A start keeps the pieces that can hold one of the chunk's positions: every piece but the jigsaw
/// ones, whose box can move (a template piece settles on the terrain when it is placed) and is read
/// when a position asks, as [`reaches_in`] reads it, and the jigsaw pieces whose box, which never
/// changes, meets the chunk's column. A start with none of those whose own box (fixed when it was
/// made) misses the column holds none of the chunk's positions and is not kept.
#[derive(Default)]
pub(super) struct ChunkStarts {
    starts: Vec<(Arc<StructureStart>, Vec<u32>)>,
}
impl ChunkStarts {
    /// The starts of the chunk `at` of `world`, as [`reaches_in`] looks them up for a position in
    /// it.
    pub(super) fn in_world(
        world: &horizon_client::world::World,
        locator: Locator<'_>,
        at: (i32, i32),
        wanted: &dyn Fn(&str) -> bool,
    ) -> Self {
        let Some(chunk) = world.chunks.get(&at) else {
            return Self::default();
        };
        if chunk.structure_refs.is_empty() {
            return Self::default();
        }
        let Some(locator) = locator() else {
            return Self::default();
        };
        Self::of(at, chunk, wanted, &|x, z| locator.starts_at(x, z))
    }
    /// The starts of `chunk`, the chunk at `at`, `starts` giving a chunk's starts.
    fn of(
        at: (i32, i32),
        chunk: &horizon_client::chunk::Chunk,
        wanted: &dyn Fn(&str) -> bool,
        starts: &dyn Fn(i32, i32) -> Vec<Arc<StructureStart>>,
    ) -> Self {
        let (x0, z0) = (at.0 << 4, at.1 << 4);
        let column = BoundingBox::new([x0, i32::MIN, z0], [x0 + 15, i32::MAX, z0 + 15]);
        let mut kept = Vec::new();
        for (id, (sx, sz)) in &chunk.structure_refs {
            if !wanted(id) {
                continue;
            }
            for start in starts(*sx, *sz) {
                if *start.name != **id {
                    continue;
                }
                let pieces: Vec<u32> = start
                    .pieces
                    .iter()
                    .enumerate()
                    .filter(|(_, piece)| {
                        !matches!(piece, Piece::Pool(_))
                            || piece.placed_bounding_box().intersects(&column)
                    })
                    .map(|(i, _)| i as u32)
                    .collect();
                if pieces.is_empty() && !start.bounding_box.intersects(&column) {
                    continue;
                }
                kept.push((start, pieces));
            }
        }
        Self { starts: kept }
    }
    /// [`reaches_in`] at `pos`, a position of the chunk.
    pub(super) fn reaching(&self, pos: [i32; 3]) -> Vec<Reach> {
        let mut out = Vec::new();
        for (start, pieces) in &self.starts {
            let in_box = start.bounding_box.is_inside(pos);
            let in_piece = pieces
                .iter()
                .any(|&i| start.pieces[i as usize].placed_bounding_box().is_inside(pos));
            add_reach(&mut out, start, in_box, in_piece);
        }
        out
    }
}

/// [`structures_reaching_in`] for a read that loads and builds nothing: from the starts the
/// generator holds already. None for an unloaded chunk, a level without the terrain that
/// locates its references, and a reference whose starts aren't held.
pub(super) fn cached_structures_reaching_in(
    world: &horizon_client::world::World,
    locator: Locator<'_>,
    pos: [i32; 3],
) -> Option<Vec<Reach>> {
    let chunk = world.chunks.get(&(pos[0] >> 4, pos[2] >> 4))?;
    if chunk.structure_refs.is_empty() {
        return Some(Vec::new());
    }
    let locator = locator()?;
    reaching(chunk, pos, &|_| true, &|x, z| locator.cached_starts_at(x, z))
}

/// The structures `chunk` refers to that hold `pos` by either read ([`Reach`]); `starts`
/// gives a chunk's starts. None when it can't give those of a chunk a wanted reference names.
fn reaching(
    chunk: &horizon_client::chunk::Chunk,
    pos: [i32; 3],
    wanted: &dyn Fn(&str) -> bool,
    starts: &dyn Fn(i32, i32) -> Option<Vec<Arc<StructureStart>>>,
) -> Option<Vec<Reach>> {
    let mut out: Vec<Reach> = Vec::new();
    for (id, (sx, sz)) in &chunk.structure_refs {
        if !wanted(id) {
            continue;
        }
        for start in starts(*sx, *sz)? {
            if *start.name != **id {
                continue;
            }
            let in_box = start.bounding_box.is_inside(pos);
            // `structureHasPieceAt`: the placed pieces, whatever the start's box says.
            let in_piece = start
                .pieces
                .iter()
                .any(|p| p.placed_bounding_box().is_inside(pos));
            add_reach(&mut out, &start, in_box, in_piece);
        }
    }
    Some(out)
}

/// Adds `start` to `out`, the structures that hold a position by either read, when it holds it:
/// the starts of a structure make one entry, which holds the position by a read when one does.
fn add_reach(out: &mut Vec<Reach>, start: &StructureStart, in_box: bool, in_piece: bool) {
    if !in_box && !in_piece {
        return;
    }
    match out.iter_mut().find(|r| r.name == start.name) {
        Some(r) => {
            r.in_box |= in_box;
            r.in_piece |= in_piece;
        }
        None => out.push(Reach {
            name: start.name.clone(),
            in_box,
            in_piece,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_client::chunk::Chunk;
    use horizon_core::{density::TerrainAdjustment, position::ChunkPos};
    use horizon_world::worldgen::structure::{
        Mirror, Rotation, StructureTemplate,
        legacy::{TemplateHandler, TemplatePiece},
        processor::ProcessorList,
        start::{Piece, StructureDef, StructureKind},
    };

    /// `getStructureWithPieceAt` reads a piece where it was placed, with no start box gate: a
    /// shipwreck that settled below the box it was made in holds the blocks of its hull there,
    /// and none of the empty box above, which only `getStructureAt` still reads.
    #[test]
    fn a_settled_shipwreck_is_found_in_its_hull_and_not_in_the_box_it_left() {
        let mut template = StructureTemplate::empty();
        template.size = [9, 7, 13];
        let wreck = TemplatePiece::new(
            "minecraft:shipwreck",
            "minecraft:shipwreck/with_mast",
            Arc::new(template),
            [100, 90, 200],
            Rotation::None,
            Mirror::None,
            [4, 0, 15],
            Arc::new(ProcessorList::default()),
            true,
            Vec::new(),
            TemplateHandler::Shipwreck { beached: false },
        );
        *wreck.state.lock().unwrap() = Some(60);
        let def = StructureDef {
            name: "minecraft:shipwreck".into(),
            kind: StructureKind::Unsupported("test".into()),
            biomes: Default::default(),
            step: 0,
            terrain: TerrainAdjustment::None,
            enhanced_terrain: None,
        };
        let home = ChunkPos::new(100 >> 4, 200 >> 4);
        let start = Arc::new(StructureStart::new(
            0,
            &def,
            home,
            vec![Piece::Legacy(Box::new(wreck))],
        ));
        let mut chunk = Chunk::empty(home.x, home.z);
        chunk.structure_refs = vec![(Arc::from("minecraft:shipwreck"), (home.x, home.z))];
        let at = |pos| reaching(&chunk, pos, &|_| true, &|_, _| Some(vec![start.clone()])).unwrap();
        let reach = |in_box, in_piece| {
            vec![Reach {
                name: "minecraft:shipwreck".into(),
                in_box,
                in_piece,
            }]
        };
        assert_eq!(at([100, 60, 200]), reach(false, true), "in the settled hull");
        assert_eq!(at([100, 90, 200]), reach(true, false), "in the box it left, in no piece");
        assert_eq!(at([100, 80, 200]), Vec::new(), "between the two");
    }

    /// A piece whose box moves by the cell's height, as a settling template piece's does when it
    /// is placed.
    #[derive(Debug)]
    struct Settling(Arc<std::sync::atomic::AtomicI32>, BoundingBox);
    impl horizon_world::worldgen::structure::legacy::LegacyPiece for Settling {
        fn bounding_box(&self) -> BoundingBox {
            let dy = self.0.load(std::sync::atomic::Ordering::Relaxed);
            self.1.moved([0, dy, 0])
        }
        fn type_name(&self) -> &'static str {
            "test:settling"
        }
        fn post_process(
            &self,
            _: &mut horizon_world::worldgen::structure::legacy::PieceContext<'_>,
        ) -> horizon_world::worldgen::structure::legacy::Result<()> {
            Ok(())
        }
    }

    /// The starts a chunk keeps answer each of its positions as its references do: the jigsaw
    /// pieces that miss the chunk's column and a start that misses it whole are dropped, a piece
    /// that moves is read when a position asks (here after the starts were kept), the references
    /// `wanted` drops are dropped, and two starts of one structure make one entry.
    #[test]
    fn a_chunks_kept_starts_answer_as_its_references() {
        use horizon_world::worldgen::structure::jigsaw::{PoolElement, PoolPiece};
        use std::sync::atomic::{AtomicI32, Ordering};
        let def = |name: &str| StructureDef {
            name: name.into(),
            kind: StructureKind::Unsupported("test".into()),
            biomes: Default::default(),
            step: 0,
            terrain: TerrainAdjustment::None,
            enhanced_terrain: None,
        };
        let pool = |min: [i32; 3], max: [i32; 3]| {
            Piece::Pool(PoolPiece {
                element: Arc::new(PoolElement::Empty),
                position: min,
                ground_level_delta: 0,
                rotation: Rotation::None,
                bounding_box: BoundingBox::new(min, max),
                junctions: Vec::new(),
                waterlogging: false,
            })
        };
        let start = |name: &str, x: i32, z: i32, pieces: Vec<Piece>| {
            Arc::new(StructureStart::new(0, &def(name), ChunkPos::new(x, z), pieces))
        };
        // Chunk (2, -3) holds x 32..=47 and z -48..=-33.
        let (cx, cz) = (2, -3);
        let village = start(
            "test:village",
            1,
            -3,
            vec![
                pool([20, 60, -60], [40, 70, -40]),
                // One block west of the chunk, and one block east.
                pool([0, 60, -60], [31, 70, -40]),
                pool([48, 60, -50], [60, 70, -40]),
                pool([33, 40, -34], [35, 90, -20]),
            ],
        );
        let lift = Arc::new(AtomicI32::new(0));
        let hull = BoundingBox::new([34, 80, -40], [40, 90, -36]);
        let wreck = start(
            "test:wreck",
            cx,
            cz,
            vec![Piece::Legacy(Box::new(Settling(lift.clone(), hull)))],
        );
        let hamlet = start("test:village", 2, -4, vec![pool([40, 64, -45], [44, 64, -45])]);
        let far = start("test:far", 5, -3, vec![pool([100, 60, -40], [110, 70, -36])]);
        // A start whose box reaches over the chunk, though none of its pieces does (a grown box).
        let mut grown = StructureStart::new(
            0,
            &def("test:grown"),
            ChunkPos::new(3, -3),
            vec![pool([50, 60, -40], [55, 70, -36])],
        );
        grown.bounding_box = BoundingBox::new([30, 0, -50], [60, 100, -30]);
        let grown = Arc::new(grown);
        let other = start("test:other", cx, cz, vec![pool([32, 64, -48], [47, 64, -33])]);
        let starts = |x: i32, z: i32| match (x, z) {
            (1, -3) => vec![village.clone()],
            (2, -3) => vec![wreck.clone(), other.clone()],
            (2, -4) => vec![hamlet.clone()],
            (5, -3) => vec![far.clone()],
            (3, -3) => vec![grown.clone()],
            _ => Vec::new(),
        };
        let mut chunk = Chunk::empty(cx, cz);
        chunk.structure_refs = [
            ("test:village", (1, -3)),
            ("test:wreck", (2, -3)),
            ("test:village", (2, -4)),
            ("test:far", (5, -3)),
            ("test:grown", (3, -3)),
            ("test:other", (2, -3)),
        ]
        .into_iter()
        .map(|(id, at)| (Arc::from(id), at))
        .collect();
        let all = |_: &str| true;
        let not_other = |id: &str| id != "test:other";
        for wanted in [&all as &dyn Fn(&str) -> bool, &not_other] {
            lift.store(0, Ordering::Relaxed);
            let kept = ChunkStarts::of((cx, cz), &chunk, wanted, &starts);
            let names: Vec<&str> = kept.starts.iter().map(|(s, _)| s.name.as_str()).collect();
            let mut expected = vec![
                "test:village",
                "test:wreck",
                "test:village",
                "test:grown",
                "test:other",
            ];
            expected.retain(|name| wanted(name));
            assert_eq!(names, expected, "the far start is dropped");
            assert_eq!(kept.starts[0].1, [0, 3], "the village's pieces over the chunk");
            assert!(kept.starts[3].1.is_empty(), "the grown start keeps its box alone");
            // The hull settles after the starts were kept.
            for dy in [0, -30] {
                lift.store(dy, Ordering::Relaxed);
                for x in 32..48 {
                    for z in -48..-32 {
                        for y in [39, 40, 55, 60, 64, 65, 70, 80, 85, 90, 91] {
                            let pos = [x, y, z];
                            let full = reaching(&chunk, pos, wanted, &|x, z| Some(starts(x, z)));
                            assert_eq!(kept.reaching(pos), full.unwrap(), "{pos:?} at {dy}");
                        }
                    }
                }
            }
        }
        let kept = ChunkStarts::of((cx, cz), &chunk, &all, &starts);
        let reach = |name: &str, in_box, in_piece| Reach {
            name: name.into(),
            in_box,
            in_piece,
        };
        lift.store(-30, Ordering::Relaxed);
        let settled = kept.reaching([36, 55, -38]);
        assert!(settled.contains(&reach("test:wreck", false, true)), "{settled:?}");
        lift.store(0, Ordering::Relaxed);
        let merged = kept.reaching([42, 64, -45]);
        assert_eq!(merged[0], reach("test:village", true, true), "{merged:?}");
    }
}
