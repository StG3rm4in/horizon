//! The mob loop of `tick_creature_entities` over a group of mobs of the parallel mob tick (#14457),
//! in their list order, with everything the loop keeps for them: their snapshot and its indexes,
//! the pushes and events their ticks give, the level random they draw from, the players they see
//! and what dies. A group holds the mobs no mob outside it can reach this tick. Each turn is the
//! turn of the borrowed loop (`borrowed_tick.rs`) for a level without native mods, the only
//! levels whose ticks run in groups (`mob_islands`): the steps that only a mod can reach (despawn
//! events, target listeners, section observers), a fox about to eat and removed vehicles keep the
//! tick serial instead.
use super::*;
use horizon_client::{
    fast_hash::{FxHashMap, FxHashSet},
    join_seq::JoinList,
    mobs::{PendingPushes, seen_grid::SeenGrid, spawning::LocalDifficulty},
};
use horizon_core::random::LevelRandom;

/// What every mob of the loop reads and none writes.
pub(super) struct Shared<'a> {
    pub uuid: Option<[i32; 4]>,
    pub level_id: &'static str,
    /// The blocks whose collision changed since the last tick: mobs re-path around them.
    pub path_changes: &'a [[i32; 3]],
    /// The guests' entity ids by connection, for the events a mob sends the guest it sees.
    pub guest_ids: &'a [(u32, u32)],
    /// The guests' identities by connection (`Game::guest_identity`), read on the server thread.
    pub guest_identities: &'a [(u32, ([i32; 4], u32))],
    pub items: &'a horizon_client::items::Items,
    pub registry: &'a horizon_world::registry::BlockRegistry,
    pub block_registries: &'a horizon_client::command::args::Registries,
    pub damage_hooks: &'a horizon_client::health::DamageHooks,
}

/// A mob whose body goes this tick: (id, feet, width, height, the grown cube it splits into).
/// The last field is the grown slime, magma cube or sulfur cube the body leaves behind
/// (`AbstractCubeMob.remove`), captured as the body goes because it has left the list by the
/// time it splits.
pub(super) type Gone = (u32, Vec3, f64, f64, Option<Mob>);

/// A group's mobs and everything the loop keeps for them, in their list order.
pub(super) struct Group {
    pub mobs: JoinList<Mob>,
    /// Each mob's place in the whole list; empty when the group is the whole list.
    pub global: Vec<usize>,
    /// After each turn, where the turn's entries end: in `events`, `dead`, `corpses`, `gone` and
    /// `guest_events`. A turn's entries stay where it left them.
    pub ends: Vec<[usize; 5]>,
    /// Each mob's snapshot entry: as it was before the tick until its turn, as its turn left it
    /// after.
    pub snapshot: Vec<Seen>,
    grid: SeenGrid,
    /// The first slot of each id, as a scan would find it.
    index: FxHashMap<u32, u32>,
    villager_slots: Vec<u32>,
    species_slots: horizon_client::mobs::species_slots::SpeciesSlots,
    solid_slots: Vec<u32>,
    pub pushes: PendingPushes,
    pub events: Vec<MobEvent>,
    /// The level random, lent to the mobs' level at each turn (brain behaviours draw from it).
    pub random: LevelRandom,
    pub guests: super::super::lan_mobs::GuestViews,
    /// What stands in a guest's slot while a mob holds that guest's view for its tick.
    spare: Option<PlayerView>,
    /// The host's own player, lent to each mob that sees it for its tick.
    pub player: Option<PlayerView>,
    /// The guest each mob sees, by mob id.
    pub bindings: FxHashMap<u32, u32>,
    pub died: FxHashSet<u32>,
    pub ticked: FxHashSet<u32>,
    /// Whether each mob stays in the list, pushed at its turn.
    pub keep: Vec<bool>,
    pub dead: Vec<Mob>,
    pub corpses: Vec<Mob>,
    pub gone: Vec<Gone>,
    pub guest_events: Vec<(u32, u32, MobEvent)>,
    /// The mobs whose last-hurt-by-player credit ran out in their turn: the game forgets who it
    /// credits for them at the join, as the borrowed loop does at the turn (no turn reads it).
    pub credit_expired: Vec<u32>,
    /// Whether each turn draws its level-random rolls from a stream of its own, so that it waits
    /// for no other lane (`parallel::mobs_own_random`, read once where the group is made).
    own_random: bool,
}

#[cfg(test)]
thread_local! {
    /// A test's switch for the groups its thread makes, over the parallel tick's.
    pub(super) static OWN_RANDOM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn own_random() -> bool {
    #[cfg(test)]
    if OWN_RANDOM.with(std::cell::Cell::get) {
        return true;
    }
    horizon_client::parallel::mobs_own_random()
}

/// What decides each turn besides the group, read at the turn as the serial loop reads it.
pub(super) struct Turns<'t, 'w> {
    pub base: &'t MobWorld<'w>,
    /// Whether a block column is in an entity-ticking chunk.
    pub ticking: &'t dyn Fn(i32, i32) -> bool,
    pub nearest: &'t super::prep::MobsNearest<'t>,
    /// The one player's position, when the level has one: every mob's nearest.
    pub single_player: Option<Option<Vec3>>,
    /// Whether a mod canceled a mob's tick.
    pub canceled: &'t (dyn Fn(u32) -> bool + Sync),
    /// `getCurrentDifficultyAt` a position.
    pub difficulty: &'t (dyn Fn([f64; 3]) -> LocalDifficulty + Sync),
    /// The last chunk's difficulty: mobs come in runs of one chunk.
    pub cached_difficulty: Option<((i32, i32), LocalDifficulty)>,
}

/// What a group starts from besides its mobs and their snapshot.
#[derive(Clone)]
pub(super) struct Start {
    pub events: Vec<MobEvent>,
    pub random: LevelRandom,
    pub guests: super::super::lan_mobs::GuestViews,
    pub player: Option<PlayerView>,
    pub bindings: FxHashMap<u32, u32>,
    pub died: FxHashSet<u32>,
    pub ticked: FxHashSet<u32>,
}

impl Group {
    /// A group of `mobs`, whose snapshot entries are `snapshot`.
    pub(super) fn new(mobs: Vec<Mob>, snapshot: Vec<Seen>, start: Start) -> Self {
        let grid = SeenGrid::new(snapshot.iter().map(|s| s.bounds));
        let mut index = FxHashMap::with_capacity_and_hasher(snapshot.len(), Default::default());
        let (mut villager_slots, mut solid_slots) = (Vec::new(), Vec::new());
        for (i, s) in snapshot.iter().enumerate() {
            index.entry(s.id).or_insert(i as u32);
            if s.species == Species::Villager {
                villager_slots.push(i as u32);
            }
            if s.may_block_mobs() {
                solid_slots.push(i as u32);
            }
        }
        let n = mobs.len();
        let species_slots = horizon_client::mobs::species_slots::SpeciesSlots::new(&snapshot);
        Self {
            spare: start.guests.first().map(|g| g.1.clone()),
            mobs: mobs.into_iter().collect(),
            global: Vec::new(),
            ends: Vec::with_capacity(n),
            snapshot,
            grid,
            index,
            villager_slots,
            species_slots,
            solid_slots,
            pushes: PendingPushes::default(),
            events: start.events,
            random: start.random,
            guests: start.guests,
            player: start.player,
            bindings: start.bindings,
            died: start.died,
            ticked: start.ticked,
            keep: Vec::with_capacity(n),
            dead: Vec::new(),
            corpses: Vec::new(),
            gone: Vec::new(),
            guest_events: Vec::new(),
            credit_expired: Vec::new(),
            own_random: own_random(),
        }
    }

    /// The group's spatial query extent and scan threshold match the whole list's `whole`.
    pub(super) fn answering_as(self, whole: (f64, usize)) -> Self {
        Self {
            grid: self.grid.answering_as(whole),
            ..self
        }
    }

    /// Mob `i`'s place in the whole list.
    pub(super) fn at(&self, i: usize) -> usize {
        self.global.get(i).copied().unwrap_or(i)
    }

    /// Every mob's turn, in order. A group of the parallel tick draws the level random in list
    /// order with the other groups (`OrderedDraws`), and shows each mob after its turn to the
    /// mobs of the other groups that look it up by id (`foreign`).
    pub(super) fn run(
        &mut self,
        turns: &mut Turns,
        shared: &Shared,
        foreign: Option<&super::mob_islands::Foreign>,
    ) {
        // A panic in a turn still lets the other groups finish (the threads' join passes it on).
        let _done = self.random.lane_done();
        for i in 0..self.mobs.len() {
            let at = self.at(i);
            self.random.begin_item(at);
            #[cfg(test)]
            super::mob_islands::inject_turn_failure(at);
            if let Some(foreign) = foreign {
                foreign.reader.set(at);
                foreign.in_seed_order(at, || self.turn(i, turns, shared));
            } else {
                self.turn(i, turns, shared);
            }
            let ends = [
                self.events.len(),
                self.dead.len(),
                self.corpses.len(),
                self.gone.len(),
                self.guest_events.len(),
            ];
            self.ends.push(ends);
            if let Some(foreign) = foreign {
                foreign.publish(at, &self.snapshot[i]);
            }
            self.random.end_item();
        }
        self.random.finish_lane();
    }

    /// Mob `i`'s turn, decided as the borrowed loop decides it: a removed mob leaves the list, a
    /// mob in an unloaded chunk is kept or despawns, one outside the entity-ticking chunks stays
    /// as it is, the others tick.
    fn turn(&mut self, i: usize, turns: &mut Turns, shared: &Shared) {
        if self.mobs[i].removed {
            self.keep.push(false);
            self.snapshot[i].alive = false;
            // `removed` is the view's authoritative gone (arismoko/horizon#795).
            self.snapshot[i].removed = true;
            return;
        }
        let position = self.mobs[i].position;
        let (bx, bz) = (position.x.floor() as i32, position.z.floor() as i32);
        let seen_guest = self.bind_guest(i);
        if !turns.base.world.loaded(bx, bz) {
            self.unloaded(i);
            return;
        }
        // ServerLevel.tick: entities outside entity-ticking chunks do not tick
        // and do not check despawn (#16749).
        if !(turns.ticking)(bx, bz) {
            self.keep.push(true);
            return;
        }
        let nearest_player = turns
            .single_player
            .unwrap_or_else(|| turns.nearest.at(self.at(i), position));
        let controlling = self.controlling(i);
        // A mod canceled its tick: it stays as it is.
        if (turns.canceled)(self.mobs[i].id) {
            self.mobs[i].despawn = mod_api::Despawn::default();
            self.keep.push(true);
            return;
        }
        // getCurrentDifficultyAt(blockPosition()): the mob's own chunk,
        // not the local player's (a server thread has no player).
        let chunk_key = (bx >> 4, bz >> 4);
        let difficulty = match turns.cached_difficulty {
            Some((key, difficulty)) if key == chunk_key => difficulty,
            _ => {
                let difficulty = (turns.difficulty)([position.x, position.y, position.z]);
                turns.cached_difficulty = Some((chunk_key, difficulty));
                difficulty
            }
        };
        let player = (seen_guest, nearest_player, controlling);
        self.tick(i, player, difficulty, turns.base, shared);
    }

    /// The guest mob `i` sees now, kept in the bindings: none while no guest is on.
    pub(super) fn bind_guest(&mut self, i: usize) -> Option<u32> {
        if self.guests.is_empty() {
            return None;
        }
        let m = &self.mobs[i];
        let bound = self.bindings.get(&m.id).copied();
        let conn = super::super::lan_mobs::player_for(m, self.player.as_ref(), &self.guests, bound);
        match conn {
            Some(conn) => self.bindings.insert(m.id, conn),
            None => self.bindings.remove(&m.id),
        };
        conn
    }

    /// Mob `i`'s turn when its chunk isn't loaded: it is kept as it is, unless it would despawn.
    fn unloaded(&mut self, i: usize) {
        let m = &self.mobs[i];
        self.keep.push(!m.hostile() || m.persistent);
    }

    /// Who controls mob `i`'s moves this tick: the velocity of the player riding it, when it is
    /// a controlled mount.
    fn controlling(&self, i: usize) -> Option<Vec3> {
        let m = &self.mobs[i];
        horizon_client::mobs::equine::controlled(m)
            .then(|| {
                self.player
                    .as_ref()
                    .filter(|p| p.riding_vehicle == Some(m.id))
                    .map(|p| p.velocity)
                    .or_else(|| {
                        self.guests
                            .iter()
                            .find(|(_, p)| p.riding_vehicle == Some(m.id))
                            .map(|(_, p)| p.velocity)
                    })
            })
            .flatten()
    }

    /// Mob `i`'s own tick, then what the loop does with what it did, in the borrowed loop's
    /// order. `players`: the guest it sees (none: the host's player), its nearest player, and the
    /// velocity of the player controlling it as a mount ([`Self::controlling`]).
    fn tick(
        &mut self,
        i: usize,
        (seen_guest, nearest_player, controlling_player_movement): (
            Option<u32>,
            Option<Vec3>,
            Option<Vec3>,
        ),
        difficulty: LocalDifficulty,
        base: &MobWorld,
        shared: &Shared,
    ) {
        let m = &mut self.mobs[i];
        horizon_client::mobs::apply_mob_pushes(m, &mut self.pushes);
        horizon_client::mobs::blocks_changed(m, shared.path_changes);
        // LivingEntity.aiStep -> EntityEquipment.tick, before movement.
        super::super::inventory_tick::equipment_inventory_tick(
            base.world,
            base.facts,
            shared.level_id,
            &mut m.extra.equipment,
        );
        let from = self.events.len();
        // The guest's view is lent to the mob for its tick, not copied: the tick only reads it
        // (`&MobWorld`), and it goes back to the guest's slot straight after. Its identity is
        // captured with it (arismoko/horizon#1077).
        let lent = seen_guest.and_then(|conn| self.guests.iter().position(|g| g.0 == conn));
        let active_player = match (seen_guest, lent) {
            (None, _) => self.player.take(),
            (Some(_), Some(k)) => self
                .spare
                .take()
                .map(|stand_in| std::mem::replace(&mut self.guests[k].1, stand_in)),
            (Some(_), None) => None,
        };
        let active_identity = match seen_guest {
            None => base.player_identity,
            Some(conn) => shared
                .guest_identities
                .iter()
                .find(|g| g.0 == conn)
                .map(|g| g.1),
        };
        let mut w = MobWorld {
            others: &self.snapshot,
            others_grid: Some(&self.grid),
            others_index: Some(&self.index),
            others_villagers: Some(&self.villager_slots),
            species_slots: Some(&self.species_slots),
            others_solid: Some(&self.solid_slots),
            difficulty,
            nearest_player,
            player: active_player,
            player_identity: active_identity,
            player_selection: None,
            teams: std::sync::Arc::clone(&base.teams),
            section_writer: None,
            ..*base
        };
        w.controlling_player_movement = controlling_player_movement;
        let id = self.mobs[i].id;
        let credit_expiring = self.mobs[i].ai.last_hurt_by_player == 1;
        self.ticked.insert(id);
        let sampled = super::super::tick_profile::start_mob(&self.mobs[i].kind);
        // The level's stream is one for every lane: a turn that draws from it waits for the
        // turns before it. On its own stream it waits for none.
        let mut own = self.own_random.then(|| {
            LevelRandom::for_level(horizon_client::parallel::mob_turn_random(id, w.game_time))
        });
        // The group's random is its mobs' level random for the turn, and comes back after it.
        let mut lent_level = horizon_client::health::LentLevel::lane(std::mem::replace(
            &mut self.random,
            LevelRandom::for_level(horizon_core::random::LegacyRandom::new(0)),
        ));
        let kept = {
            let mut level = horizon_client::health::DamageLevel::lent(
                base.world,
                shared.registry,
                &mut lent_level,
                Some(shared.items),
                Some(shared.block_registries),
                shared.damage_hooks,
            );
            let mut level = match own.as_mut() {
                Some(own) => level.reborrow_with_random(own),
                None => level.reborrow(),
            };
            match horizon_client::mobs::riders::tick_until_consumption(
                &mut self.mobs,
                i,
                &w,
                &mut level,
                &mut self.events,
            ) {
                horizon_client::mobs::ai::TickProgress::Complete(kept) => kept,
                // A meal pauses the turn for the game to serve it, which a group can't call, so a
                // tick with a fox about to eat stays serial (`fox::may_eat_next_tick`). One that
                // came to it all the same (an earlier turn woke it, or its target went) goes on
                // without the meal, and eats next tick.
                horizon_client::mobs::ai::TickProgress::Consumption { was_on_fire, .. } => {
                    horizon_client::mobs::riders::resume_after_consumption(
                        &mut self.mobs,
                        i,
                        &w,
                        &mut level,
                        was_on_fire,
                        &mut self.events,
                    )
                }
            }
        };
        super::super::tick_profile::end_mob(shared.level_id, sampled);
        let m = &mut self.mobs[i];
        m.despawn = mod_api::Despawn::default();
        if credit_expiring && m.ai.last_hurt_by_player == 0 {
            self.credit_expired.push(m.id);
        }
        if seen_guest.is_none() {
            self.player = w.player;
        } else if let Some(k) = lent {
            self.spare = w
                .player
                .map(|v| std::mem::replace(&mut self.guests[k].1, v));
        }
        // A killed mob's body is removed: its effects that act on
        // death run (`LivingEntity.remove` with KILLED).
        if !kept && !m.alive() && !m.removed && !m.extra.effects.is_empty() {
            self.corpses.push(m.clone());
        }
        horizon_client::mobs::take_mob_pushes(&mut self.events, from, &mut self.pushes);
        // LivingEntity.tickDeath: after twenty ticks the body is removed.
        if !kept && !m.removed && !m.alive() && m.death_time >= 20 {
            let (width, height) = (f64::from(m.width()), f64::from(m.height()));
            // `AbstractCubeMob.remove` splits a grown cube into its smaller halves when it is
            // removed, twenty ticks after the death. The parent leaves the list before the split
            // runs, so its size, position, kind and persistence are captured here, as they are
            // at the removal.
            let split = if matches!(
                m.species(),
                Species::Slime | Species::MagmaCube | Species::SulfurCube
            ) && m.extra.size > 1
            {
                Some(m.clone())
            } else {
                None
            };
            self.gone.push((m.id, m.position, width, height, split));
        }
        super::borrowed_tick::route_player_events(
            &mut self.events,
            from,
            seen_guest,
            shared.guest_ids,
            &mut self.guest_events,
            m.id,
        );
        // Killed during its own tick (drowning, burning, falling...).
        if !m.alive() && !m.removed && self.died.insert(m.id) {
            self.dead.push(m.clone());
        }
        if !kept {
            self.died.remove(&m.id);
        }
        self.keep.push(kept);
        // ServerLevel ticks entities one after another: the ones
        // after this mob see it as it is now.
        let fresh = seen(m, shared.uuid, shared.items);
        let reclassified = self.snapshot[i].species != fresh.species;
        self.snapshot[i] = fresh;
        if reclassified {
            self.species_slots =
                horizon_client::mobs::species_slots::SpeciesSlots::new(&self.snapshot);
        }
        self.snapshot[i].alive &= kept;
        self.snapshot[i].removed |= !kept;
        self.grid.update(i, self.snapshot[i].bounds);
        {
            let mut level = horizon_client::health::DamageLevel::lent(
                base.world,
                shared.registry,
                &mut lent_level,
                Some(shared.items),
                Some(shared.block_registries),
                shared.damage_hooks,
            );
            horizon_client::mobs::apply_brain_events(
                &mut self.mobs,
                &mut level,
                &mut self.events,
                from,
            );
        }
        self.random = lent_level.into_random();
    }
}
