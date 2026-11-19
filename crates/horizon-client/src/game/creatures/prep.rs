//! What the mob loop in islands (`mob_islands`) prepares before it runs: each mob's nearest player,
//! found on the tick's helper threads while the parallel tick is on (`horizon_client::parallel`),
//! each answer reading only that mob's position and the players, and the mobs' world without its
//! hooks into the game, for the helpers.
#[cfg(feature = "upstream-parallel-mob-islands")]
use super::super::natural_spawning::NearestPlayers;
#[cfg(feature = "upstream-parallel-mob-islands")]
use horizon_client::mobs::Mob;
use horizon_client::mobs::MobWorld;
#[cfg(feature = "upstream-parallel-mob-islands")]
use horizon_client::parallel::par_map;
#[cfg(feature = "upstream-parallel-mob-islands")]
use horizon_core::geometry::Vec3;

/// From this many mobs a preparation goes to the helper threads.
#[cfg(feature = "upstream-parallel-mob-islands")]
const MIN_MOBS: usize = 256;

/// The nearest player each mob's despawn check reads (`Mob.checkDespawn`), found for every mob before
/// the loop. A mob that moved before its turn is asked again where it stands.
#[cfg(feature = "upstream-parallel-mob-islands")]
pub(super) struct MobsNearest<'p> {
    players: &'p NearestPlayers<'p>,
    /// Each mob's position as the loop started and its nearest player there; empty while the
    /// parallel tick is off (every mob asks at its turn).
    found: Vec<(Vec3, Option<Vec3>)>,
}

#[cfg(feature = "upstream-parallel-mob-islands")]
impl<'p> MobsNearest<'p> {
    /// `wanted`: whether the loop asks (not with one player, whose position answers).
    pub(super) fn new(players: &'p NearestPlayers<'p>, at: &[Vec3], wanted: bool) -> Self {
        let found = if wanted && horizon_client::parallel::lanes() > 1 {
            par_map(at, MIN_MOBS, |part, out| {
                out.extend(part.iter().map(|&p| (p, Self::ask(players, p))));
            })
        } else {
            Vec::new()
        };
        Self { players, found }
    }
    fn ask(players: &NearestPlayers, at: Vec3) -> Option<Vec3> {
        players.nearest(at.x, at.y, at.z).map(|p| p.position)
    }
    /// The nearest player to mob `i` of the loop, standing at `at`.
    pub(super) fn at(&self, i: usize, at: Vec3) -> Option<Vec3> {
        match self.found.get(i) {
            Some(&(was, nearest)) if was == at => nearest,
            _ => Self::ask(self.players, at),
        }
    }
}

/// Where each of `mobs` stands, for [`MobsNearest::new`].
#[cfg(feature = "upstream-parallel-mob-islands")]
pub(super) fn positions(mobs: &[Mob]) -> Vec<Vec3> {
    mobs.iter().map(|m| m.position).collect()
}

/// `base` without its hooks into the game, which can't leave the server thread, as a recipe a
/// helper thread builds its own world from: native mod effects (the phases on helpers don't run
/// them) and the UUID lookup.
pub(crate) fn hookless<'w>(base: &MobWorld<'w>) -> impl Fn() -> MobWorld<'w> + Sync + use<'w> {
    // Every field named, so that a field added to `MobWorld` is decided here: the hooks into the
    // game and the per-mob views (`player_selection`, `live_mobs`, `section_writer`) stay behind.
    let MobWorld {
        mob_maps,
        world,
        registry,
        facts,
        fluid_tables,
        movement,
        collision,
        suffocation,
        data,
        player,
        controlling_player_movement,
        nearest_player,
        others,
        dragon,
        leash_boats,
        others_grid,
        others_index,
        others_villagers,
        species_slots: _,
        others_solid,
        daylight,
        monsters_burn,
        game_time,
        day_time,
        raining,
        thundering,
        level,
        block_tags,
        landing,
        biomes,
        mob_griefing,
        universal_anger,
        mob_drops,
        max_entity_cramming,
        difficulty,
        dimension_type,
        world_seed,
        sea_level,
        teams,
        items,
        lid_openers,
        mod_effects: _,
        who_lookup: _,
        foreign: _,
        border,
        professions,
        loop_context,
        attributes,
        player_selection: _,
        player_identity,
        passenger_objects,
        live_mobs: _,
        mob_hooks: _,
        goal_host: _,
        asks_animal_tame,
        asks_living_destroy_block,
        asks_baby_spawn,
        section_writer: _,
    } = base.clone();
    move || MobWorld {
        mob_maps,
        world,
        registry,
        facts,
        fluid_tables,
        movement,
        collision,
        suffocation,
        data,
        player: player.clone(),
        controlling_player_movement,
        nearest_player,
        others,
        dragon,
        leash_boats,
        others_grid,
        others_index,
        others_villagers,
        species_slots: None,
        others_solid,
        daylight,
        monsters_burn,
        game_time,
        day_time,
        raining,
        thundering,
        level,
        block_tags,
        landing,
        biomes,
        mob_griefing,
        universal_anger,
        mob_drops,
        max_entity_cramming,
        difficulty,
        dimension_type,
        world_seed,
        sea_level,
        teams: std::sync::Arc::clone(&teams),
        items,
        lid_openers,
        mod_effects: None,
        who_lookup: None,
        foreign: None,
        border,
        professions,
        loop_context,
        attributes,
        player_selection: None,
        player_identity,
        passenger_objects,
        live_mobs: None,
        mob_hooks: None,
        goal_host: None,
        asks_animal_tame,
        asks_living_destroy_block,
        asks_baby_spawn,
        section_writer: None,
    }
}

#[cfg(all(test, feature = "upstream-parallel-mob-islands"))]
mod tests {
    use super::super::super::natural_spawning::LevelPlayer;
    use super::*;
    use horizon_core::geometry::Aabb;

    #[test]
    #[ignore = "needs the upstream-parallel-mobs feature and prepared resources"]
    fn the_nearest_player_is_asked_again_after_a_move() {
        let mut game = super::super::super::native_machines::tests::server_game();
        game.build_registries();
        game.set_allow_commands(true);
        for command in [
            "summon pig 8.5 80 8.5 {NoAI:1b}",
            "summon pig 9.5 80 8.5 {NoAI:1b}",
        ] {
            assert!(game.run_command(command), "{command}");
        }
        assert_eq!(game.mobs.len(), 2);
        let examples = game.mobs.clone();
        game.mobs = (0..MIN_MOBS + 37)
            .map(|i| {
                let mut mob = examples[i % 2].clone();
                mob.id = 10_000 + i as u32;
                mob
            })
            .collect();
        let player = |x, spectator| {
            let position = Vec3::new(x, 80.0, 8.5);
            LevelPlayer {
                uuid: 0,
                position,
                bounds: Aabb::new(position, position),
                alive: false, // dead players still count for checkDespawn
                creative: false,
                spectator,
            }
        };
        let players = [player(8.5, true), player(7.5, false), player(9.5, false)];
        let index = NearestPlayers::new(&players);
        let nearest = MobsNearest::new(&index, &positions(&game.mobs), true);
        for (i, mob) in game.mobs.iter().enumerate() {
            let expected = if i % 2 == 0 {
                players[1].position
            } else {
                players[2].position
            };
            assert_eq!(nearest.at(i, mob.position), Some(expected));
            // A previous mob's push moved this mob across the nearest-player tie.
            assert_eq!(
                nearest.at(i, Vec3::new(10.0, 80.0, 8.5)),
                Some(players[2].position)
            );
        }
        let no_cache = MobsNearest::new(&index, &positions(&game.mobs), false);
        assert_eq!(
            no_cache.at(0, game.mobs[0].position),
            Some(players[1].position)
        );
        let spectators = [player(8.5, true)];
        let index = NearestPlayers::new(&spectators);
        let nearest = MobsNearest::new(&index, &positions(&game.mobs), true);
        assert_eq!(nearest.at(0, game.mobs[0].position), None);
    }
}
