//! `BigDripleafBlock.entityInside` fires for every entity whose box is inside
//! the leaf, and `BlockBehaviour.getEntityInsideCollisionShape` is the whole
//! block, so a motionless mob, a dropped item, a boat, a minecart and a trident
//! standing on one tilt it exactly as a player does: UNSTABLE, PARTIAL ten
//! ticks later, FULL ten ticks after that, and back to NONE 100 ticks after
//! FULL.

use super::Game;
use super::block_ticks::standing_feet;
use super::boats::Boat;
use super::game_tests::tests::server_game;
use super::minecarts::{CartKind, Minecart};
use super::trident::ThrownTrident;
use horizon_client::chunk::Chunk;
use horizon_client::entities::ItemEntity;
use horizon_client::items::ItemStack;
use horizon_client::mobs::Mob;
use horizon_core::geometry::Vec3;
use std::sync::Arc;

#[test]
fn feet_exclude_a_leaf_only_touched_by_the_undeflated_box() {
    let mut standing = Vec::new();
    standing_feet(&mut standing, [1.3 - 0.000005, 64.9, 0.5], true, 0.3);
    assert_eq!(standing.len(), 1);
    assert_eq!(standing[0].0, ([1, 64, 0], [1, 64, 0]));
}

/// The `tilt` of the block at `pos`.
fn tilt(game: &Game<'_>, pos: [i32; 3]) -> String {
    let state = game.world.block(pos[0], pos[1], pos[2]);
    game.pack
        .assets
        .registry
        .state(state)
        .and_then(|s| s.properties.get("tilt").cloned())
        .unwrap_or_default()
}

/// A server's game whose chunk (0, 0) and its neighbours are lit, so its
/// scheduled ticks run (`ChunkMap.prepareTickingChunk`), with a player in it.
fn test_game<'a>(resources: &'a mut Option<Arc<crate::game::Resources>>) -> Game<'a> {
    let mut game = server_game(resources);
    game.settings.simulation_distance = 4;
    game.settings.render_distance = 8;
    game.player = Some(
        horizon_client::motion::PlayerMotion::new(game.motion, Vec3::new(8.5, 65.0, 8.5)).unwrap(),
    );
    for x in -1..=1 {
        for z in -1..=1 {
            let mut chunk = Chunk::empty(x, z);
            chunk.lit = true;
            game.world.chunks.insert((x, z), Arc::new(chunk));
        }
    }
    game.update_chunk_tickets();
    game
}

/// `n` level ticks of the block phases (`ServerLevel.tick`'s, without the
/// entities' own ticks: the entity sits still on the leaf).
fn tick_blocks(game: &mut Game<'_>, n: u32) {
    for _ in 0..n {
        game.world.game_time += 1;
        game.tick_blocks().unwrap();
    }
}

#[test]
#[ignore = "daily"]
fn a_leaf_tilts_under_a_mob_an_item_a_boat_a_minecart_and_a_trident() {
    let mut resources = None;
    let mut game = test_game(&mut resources);
    let registry = game.pack.assets.registry;
    let clay = registry.default_state("minecraft:clay").unwrap();
    let leaf = registry.default_state("minecraft:big_dripleaf").unwrap();
    // Six leaves on a clay floor at y 63; the fifth has nothing on it.
    let xs = [0, 2, 4, 6, 8, 10];
    for x in xs {
        game.world.set(x, 63, 8, clay).unwrap();
        game.world.set(x, 64, 8, leaf).unwrap();
    }
    // Each entity rests on the leaf's top (15/16, 64.9375) and stands on the
    // ground: what its own tick sets, and what canEntityTilt reads.
    let top = 64.9375;
    let mut zombie = Mob::new(
        1,
        game.res
            .mob_types()
            .get("minecraft:zombie")
            .unwrap()
            .clone(),
        Vec3::new(0.5, top, 8.5),
        0.0,
    );
    zombie.on_ground = true;
    game.mobs.push(zombie);
    let mut item = ItemEntity::new(
        2,
        ItemStack::new("minecraft:stone", 1),
        Vec3::new(2.5, top, 8.5),
        Vec3::ZERO,
        0.0,
    );
    item.on_ground = true;
    game.dropped.push(item);
    let mut boat = Boat::new(3, "oak".into(), Vec3::new(4.5, top, 8.5), 0.0);
    boat.on_ground = true;
    game.boats.push(boat);
    let mut cart = Minecart::new(4, CartKind::Rideable, Vec3::new(6.5, top, 8.5));
    cart.on_ground = true;
    game.carts.push(cart);
    // `AbstractArrow.applyEffectsFromBlocks` reaches a stuck trident too.
    let mut trident = ThrownTrident::remote(5, Vec3::new(10.5, top, 8.5), 0.0, 0.0);
    trident.on_ground = true;
    game.tridents.push(trident);

    let tilted = |game: &Game<'_>| -> Vec<String> { xs.map(|x| tilt(game, [x, 64, 8])).to_vec() };
    let expect = |state: &str| -> Vec<String> { vec![state.to_owned(); 6] };
    assert_eq!(tilted(&game), expect("none"), "flat to start with");

    // One tick inside the leaf: UNSTABLE, with no sound and no vibration
    // (`setTiltAndScheduleTick` with no sound).
    game.audio.capture = Some(Vec::new());
    tick_blocks(&mut game, 1);
    let mut want = expect("unstable");
    want[4] = "none".to_owned();
    assert_eq!(tilted(&game), want);
    assert!(
        game.audio.capture.as_ref().unwrap().is_empty(),
        "going UNSTABLE plays nothing"
    );

    // PARTIAL ten ticks later, with the leaf's tilt-down sound under it.
    tick_blocks(&mut game, 10);
    let mut want = expect("partial");
    want[4] = "none".to_owned();
    assert_eq!(tilted(&game), want);
    let played: Vec<_> = game
        .audio
        .capture
        .take()
        .unwrap()
        .into_iter()
        .map(|s| (s.event, s.position, s.volume, s.pitch))
        .collect();
    let at = |x: i32| [f64::from(x) + 0.5, 64.5, 8.5];
    assert_eq!(played.len(), 5, "one sound per tilted leaf: {played:?}");
    // The leaves tick in the order their ticks were scheduled, which is the
    // order the entities were fed: check each leaf's sound, not that order.
    for x in [0, 2, 4, 6, 10] {
        let here: Vec<_> = played
            .iter()
            .filter(|(_, position, _, _)| *position == at(x))
            .collect();
        assert_eq!(here.len(), 1, "one sound at {x}: {played:?}");
        let (event, _, volume, pitch) = here[0];
        assert_eq!(event, "minecraft:block.big_dripleaf.tilt_down");
        assert_eq!(*volume, 1.0);
        // Mth.randomBetween(random, 0.8F, 1.2F) from the level random.
        assert!((0.8..1.2).contains(pitch), "pitch {pitch} outside 0.8..1.2");
    }

    // FULL ten ticks after PARTIAL, then back to NONE 100 ticks after FULL.
    tick_blocks(&mut game, 10);
    let mut want = expect("full");
    want[4] = "none".to_owned();
    assert_eq!(tilted(&game), want);
    tick_blocks(&mut game, 100);
    assert_eq!(tilted(&game), expect("none"), "the leaf comes back up");
}

#[test]
#[ignore = "daily"]
fn a_leaf_stays_flat_under_an_entity_that_cannot_tilt_it() {
    let mut resources = None;
    let mut game = test_game(&mut resources);
    let registry = game.pack.assets.registry;
    let clay = registry.default_state("minecraft:clay").unwrap();
    let leaf = registry.default_state("minecraft:big_dripleaf").unwrap();
    let xs = [0, 2, 4];
    for x in xs {
        game.world.set(x, 63, 8, clay).unwrap();
        game.world.set(x, 64, 8, leaf).unwrap();
    }
    // canEntityTilt: `entity.onGround() && entity.position().y > pos.getY() +
    // 0.6875F`, and `checkInsideBlocks` skips a dead entity. One falls (not on
    // the ground), one floats at 0.6875 above the leaf's floor, one is dead:
    // all three are inside a leaf's box.
    let mut falling = Mob::new(
        1,
        game.res
            .mob_types()
            .get("minecraft:zombie")
            .unwrap()
            .clone(),
        Vec3::new(0.5, 64.9375, 8.5),
        0.0,
    );
    falling.on_ground = false;
    game.mobs.push(falling);
    let mut low = Mob::new(
        2,
        game.res
            .mob_types()
            .get("minecraft:zombie")
            .unwrap()
            .clone(),
        Vec3::new(2.5, 64.6875, 8.5),
        0.0,
    );
    low.on_ground = true;
    game.mobs.push(low);
    let mut dead = Mob::new(
        3,
        game.res
            .mob_types()
            .get("minecraft:zombie")
            .unwrap()
            .clone(),
        Vec3::new(4.5, 64.9375, 8.5),
        0.0,
    );
    dead.on_ground = true;
    dead.health = 0.0;
    game.mobs.push(dead);
    tick_blocks(&mut game, 1);
    assert_eq!(
        xs.map(|x| tilt(&game, [x, 64, 8])).to_vec(),
        ["none", "none", "none"]
    );
}

/// The value of property `name` of the block at `pos`.
fn property(game: &Game<'_>, pos: [i32; 3], name: &str) -> String {
    let state = game.world.block(pos[0], pos[1], pos[2]);
    game.pack
        .assets
        .registry
        .state(state)
        .and_then(|s| s.properties.get(name).cloned())
        .unwrap_or_default()
}

/// The entity sweep passes over a box, or an entity's feet, whose sections
/// hold no entity-sensing block, or no leaf, in their palettes
/// (`SweepSections::may_hold`). A scene of two plates under a zombie and an
/// item, leaves under zombies, and entities where no section holds either
/// ticks to the same blocks, the same block changes and the same sounds
/// (whose pitches draw the level's random) as when every cell is read. The
/// first plate's press sends every later cell through (`now != looked`).
#[test]
#[ignore = "daily"]
fn the_sweep_passes_over_quiet_sections_with_the_same_effects() {
    let run = |read_every_cell: bool| {
        let mut resources = None;
        let mut game = test_game(&mut resources);
        let registry = game.pack.assets.registry;
        let state = |name: &str| registry.default_state(name).unwrap();
        let (stone, clay) = (state("minecraft:stone"), state("minecraft:clay"));
        let leaf = state("minecraft:big_dripleaf");
        // A stone plate and an oak plate in chunk (0, 0)'s section 64..80,
        // the stone one first in (x, y, z) order, with a leaf beside them;
        // another leaf alone in section 80..96; stone in chunk (-1, -1).
        let placed = [
            (2, 63, 2, stone),
            (2, 64, 2, state("minecraft:stone_pressure_plate")),
            (12, 63, 12, stone),
            (12, 64, 12, state("minecraft:oak_pressure_plate")),
            (6, 63, 12, clay),
            (6, 64, 12, leaf),
            (4, 89, 4, clay),
            (4, 90, 4, leaf),
            (-9, 63, -9, stone),
        ];
        for (x, y, z, block) in placed {
            game.world.set(x, y, z, block).unwrap();
        }
        let zombie = game
            .res
            .mob_types()
            .get("minecraft:zombie")
            .unwrap()
            .clone();
        // On the stone plate, beside it, across a section edge, on each
        // leaf, and on the stone in chunk (-1, -1).
        let mobs = [
            (2.5, 64.0, 2.5),
            (4.5, 64.0, 2.5),
            (9.5, 79.0, 3.5),
            (6.5, 64.9375, 12.5),
            (4.5, 90.9375, 4.5),
            (-8.5, 64.0, -8.5),
        ];
        for (id, (x, y, z)) in (1..).zip(mobs) {
            let mut mob = Mob::new(id, zombie.clone(), Vec3::new(x, y, z), 0.0);
            mob.on_ground = true;
            game.mobs.push(mob);
        }
        // An item on the oak plate, and one on nothing in chunk (1, 1).
        for (id, x, z) in [(20, 12.5, 12.5), (21, 24.5, 24.5)] {
            let stack = ItemStack::new("minecraft:stone", 1);
            let mut item = ItemEntity::new(id, stack, Vec3::new(x, 64.0, z), Vec3::ZERO, 0.0);
            item.on_ground = true;
            game.dropped.push(item);
        }
        // The player high over chunk (0, 0): its box and its feet reach air.
        game.player = Some(
            horizon_client::motion::PlayerMotion::new(game.motion, Vec3::new(8.5, 100.0, 8.5))
                .unwrap(),
        );
        game.update_chunk_tickets();
        game.tick_scratch.sweep.read_every_cell = read_every_cell;
        game.audio.capture = Some(Vec::new());
        tick_blocks(&mut game, 25);
        let mut blocks = Vec::new();
        for x in -16..32 {
            for y in 60..110 {
                for z in -16..32 {
                    blocks.push(game.world.block(x, y, z).0);
                }
            }
        }
        let sounds: Vec<_> = game
            .audio
            .capture
            .take()
            .unwrap()
            .into_iter()
            .map(|s| (s.event, s.position, s.volume, s.pitch))
            .collect();
        let ends = [
            property(&game, [2, 64, 2], "powered"),
            tilt(&game, [6, 64, 12]),
            tilt(&game, [4, 90, 4]),
        ];
        let passed_over = game.tick_scratch.sweep.passed_over;
        (blocks, game.world.revision, sounds, ends, passed_over)
    };
    let shortcut = run(false);
    let every_cell = run(true);
    assert!(shortcut.4 > 0, "no box was passed over");
    assert_eq!(every_cell.4, 0);
    assert_eq!(shortcut.3, ["true", "full", "full"].map(String::from));
    assert_eq!(shortcut.0, every_cell.0, "the blocks differ");
    assert_eq!(shortcut.1, every_cell.1, "the block changes differ");
    assert_eq!(shortcut.2, every_cell.2, "the sounds differ");
}
