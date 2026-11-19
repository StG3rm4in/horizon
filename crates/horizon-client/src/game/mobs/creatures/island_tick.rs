//! The mob loop of `tick_creature_entities` in islands on several threads (`mob_islands`), under
//! the parallel tick, for a tick in which the borrowed loop (`borrowed_tick.rs`) would not call
//! back into the game: the same turns (`mob_group`), with their results left where the borrowed
//! loop leaves its own. Behind the `upstream-parallel-mob-islands` feature.
use super::mob_group::{Shared, Start, Turns};
use super::mob_islands::{HostSerial, Islands};
use super::*;
use horizon_core::random::{LegacyRandom, LevelRandom};

/// What the borrowed loop hands back: the level's random and attachments, and a panic for the
/// caller to resume once it has put everything back.
type Finished = (
    horizon_client::health::LentLevel,
    Option<Box<dyn std::any::Any + Send>>,
);

impl Game<'_> {
    /// The mob loop in islands, when this tick can run so; otherwise `state` comes back as it was
    /// (boxed: it is large), for the borrowed loop.
    pub(super) fn tick_mob_islands<'a>(
        &mut self,
        mut state: borrowed_tick::BorrowedMobTick<'a>,
    ) -> std::result::Result<Finished, Box<borrowed_tick::BorrowedMobTick<'a>>> {
        if !super::mob_islands::may_group() {
            return Err(Box::new(state));
        }
        // As the borrowed loop does before each of its passes.
        self.leash_boats = self
            .boats
            .iter()
            .map(super::super::boat_leads::boat_state)
            .collect();
        let round = state.round;
        let (rows, outcome) = {
            let mut base = self.mob_world_base(&[]);
            base.section_writer = None;
            base.passenger_objects = state.passenger_objects;
            let host = HostSerial {
                hooks: !mod_api::__private::mods().is_empty()
                    || round.sections()
                    || mod_api::__private::listened(mod_api::HookId::MobDespawn)
                    || mod_api::__private::listened(mod_api::HookId::LivingChangeTarget)
                    || base.mob_hooks.is_some()
                    || base.goal_host.is_some()
                    || base.asks_animal_tame
                    || base.asks_living_destroy_block
                    || base.asks_baby_spawn,
                off_level: !state.blocked.is_empty()
                    || !self.persist.off_level.is_empty()
                    || !self.persist.removed_vehicles.is_empty(),
                pending_pushes: !state.pushes.is_empty(),
                meal: state
                    .mobs
                    .iter()
                    .any(|m| horizon_client::mobs::fox::may_eat_next_tick(m, &base)),
            };
            let islands = Islands {
                uuid: self
                    .player
                    .is_some()
                    .then_some(self.level.player_uuid)
                    .flatten(),
                ticking: &|| Box::new(self.entity_ticking_area()),
                host,
            };
            if islands.serial_now(&state.mobs[..]) {
                return Err(Box::new(state));
            }
            let players = self.level_players();
            let nearest_players = super::super::natural_spawning::NearestPlayers::new(&players);
            let single_player = horizon_client::mobs::single_player_pos(
                players.iter().map(|p| (p.position, p.spectator)),
            );
            let nearest = super::prep::MobsNearest::new(
                &nearest_players,
                &super::prep::positions(&state.mobs[..]),
                single_player.is_none(),
            );
            let path_changes = horizon_client::mobs::collision_changes(&base, state.path_writes);
            let guest_identities: Vec<(u32, ([i32; 4], u32))> = state
                .guests
                .iter()
                .filter_map(|(conn, _)| Some((*conn, self.guest_identity(*conn)?)))
                .collect();
            let shared = Shared {
                uuid: state.uuid,
                level_id: state.level_id,
                path_changes: &path_changes,
                guest_ids: &state.guest_ids[..],
                guest_identities: &guest_identities,
                items: world_items!(self),
                registry: self.pack.assets.registry,
                block_registries: &self.pack.block_registries,
                damage_hooks: &self.damage_hooks,
            };
            let ticking = self.entity_ticking_area();
            let world = &self.world;
            let difficulty = move |at: [f64; 3]| local_difficulty_in(world, at);
            let canceled = |id: u32| round.canceled(id);
            // The level random goes to the groups' lanes and comes back after them; as it was,
            // should a turn panic.
            let random = std::mem::replace(
                state.lent.random_mut(),
                LevelRandom::for_level(LegacyRandom::new(0)),
            );
            let before = random.clone();
            let start = Start {
                events: std::mem::take(state.events),
                random,
                // Copies: a turn's panic leaves the caller's as they were.
                guests: state.guests.clone(),
                player: base.player.take(),
                bindings: state.bindings.clone(),
                died: state.died.clone(),
                ticked: std::mem::take(state.ticked),
            };
            let mut turns = Turns {
                base: &base,
                ticking: &ticking,
                nearest: &nearest,
                single_player,
                canceled: &canceled,
                difficulty: &difficulty,
                cached_difficulty: None,
            };
            let Some(plan) = islands.plan(&state.mobs[..], &state.snapshot[..], &turns, &start)
            else {
                // Serial after all: what the start took goes back.
                *state.events = start.events;
                *state.ticked = start.ticked;
                *state.lent.random_mut() = start.random;
                return Err(Box::new(state));
            };
            state.laps.lap("creature_prepare");
            let mut rows = state.mobs.take_rows();
            let snapshot = std::mem::take(state.snapshot);
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                islands.run(plan, &mut rows, snapshot, start, &mut turns, &shared)
            }));
            state.laps.lap("creature_loop");
            (rows, ran.map_err(|payload| (payload, before)))
        };
        let mut group = match outcome {
            Ok(group) => group,
            Err((payload, before)) => {
                // Every mob as far as its turns got, and the level random as the tick found it.
                state.mobs.put_rows(rows);
                *state.lent.random_mut() = before;
                return Ok((state.lent, Some(payload)));
            }
        };
        // As the borrowed loop ends its pass: the mobs that go leave the list, then every mob
        // left takes the pushes given to it after its turn.
        state.mobs.put_rows(group.mobs.take_rows());
        let mut keep = std::mem::take(&mut group.keep).into_iter();
        state.mobs.retain(|_| keep.next().unwrap_or(true));
        for m in state.mobs.iter_mut() {
            horizon_client::mobs::apply_mob_pushes(m, &mut group.pushes);
        }
        *state.pushes = group.pushes;
        *state.snapshot = group.snapshot;
        *state.events = group.events;
        *state.dead = group.dead;
        *state.corpses = group.corpses;
        *state.gone = group.gone;
        *state.died = group.died;
        *state.ticked = group.ticked;
        *state.bindings = group.bindings;
        state.guest_events.extend(group.guest_events);
        for id in group.credit_expired {
            state.player_credit.remove(&id);
        }
        *state.lent.random_mut() = group.random;
        // The borrowed loop leaves no section write behind either.
        self.native_lifecycle.entity_sections.ticked.clear();
        Ok((state.lent, None))
    }
}
