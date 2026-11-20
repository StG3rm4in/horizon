//! The block phases of a level's tick (`ServerLevel.tick`): scheduled ticks, random ticks in the
//! chunks the tickets simulate, block events and moving pistons (`tick_blocks`).
use super::*;

/// How many chunks the random tick loop reads ahead of itself.
const READ_AHEAD: usize = 8;

impl Game<'_> {
    /// ServerLevel.tick block phases: scheduled ticks, random ticks in the
    /// chunks the tickets simulate, block events and moving pistons.
    /// The length the block log had after the changed blocks went out
    /// (`ServerChunkCache.tick`'s broadcast), before the block events and the
    /// block entities ran.
    pub(super) fn tick_blocks(&mut self) -> Result<Option<usize>> {
        let mut scratch = std::mem::take(&mut self.tick_scratch);
        let result = self.tick_blocks_with_scratch(&mut scratch);
        self.tick_scratch = scratch;
        result
    }
    pub(super) fn tick_blocks_with_scratch(
        &mut self,
        scratch: &mut TickScratch,
    ) -> Result<Option<usize>> {
        let mut laps = horizon_client::spans::laps();
        // ServerChunkCache.tick: the tickets that time out.
        let world = &self.world;
        self.tickets
            .tick_portal(|ticket| portal_ticket_can_expire(world, ticket));
        self.sync_tick_chunks();
        self.start_ticking_chunks();
        laps.lap("tb_tickets");
        self.random_tick_chunks(&mut scratch.ticking);
        laps.lap("tb_chunks");
        // Entity.checkInsideBlocks for the plates, tripwires, detector rails and
        // buttons under the players (the host and the LAN guests), the minecarts
        // and the arrows (the mobs and other entities are added in `with_level`
        // below).
        scratch.inside_boxes.clear();
        scratch.inside.clear();
        let guests = self.lan_block_trigger_boxes();
        let boxes = self
            .player
            .iter()
            .filter(|_| !self.spectator())
            .map(|p| {
                let w = f64::from(p.dimensions.width) / 2.0;
                let h = f64::from(p.dimensions.height);
                (
                    [p.position.x - w, p.position.y, p.position.z - w],
                    [p.position.x + w, p.position.y + h, p.position.z + w],
                )
            })
            .chain(self.carts.iter().map(|c| {
                let p = c.position;
                (
                    [p.x - 0.49, p.y, p.z - 0.49],
                    [p.x + 0.49, p.y + 0.7, p.z + 0.49],
                )
            }))
            .chain(guests);
        // Their cells are walked in `with_level` below, box by box.
        for (lo, hi) in boxes {
            scratch.inside_boxes.push(box_cells(lo, hi));
        }
        // AbstractArrow.applyEffectsFromBlocks: the blocks every arrow passed
        // through this tick (a wooden button is pressed by one inside it), and
        // the ones a thrown trident is in.
        for p in self
            .creatures
            .projectiles
            .iter()
            .filter(|p| p.arrow.is_some())
        {
            scratch.inside.extend(horizon_client::arrows::blocks_inside(
                p.previous, p.position,
            ));
        }
        for t in &self.tridents {
            scratch.inside.extend(horizon_client::arrows::blocks_inside(
                t.position, t.position,
            ));
        }
        // ServerLevel.tick ticks only the entities in entity-ticking chunks, and
        // applyEffectsFromBlocks runs in their tick: a mob or item past the
        // simulation distance presses, burns and slows on nothing.
        let entity_ticking = self.entity_ticking_area();
        // BigDripleafBlock.entityInside also asks whether the entity stands on
        // the ground and how high its feet are: the leaves its box's feet
        // layer is inside. The host, LAN guests, mobs, dropped items, boats,
        // minecarts and other entities feed it; fishing hooks are still separate.
        scratch.standing.clear();
        for p in self.player.iter().filter(|_| !self.spectator()) {
            standing_feet(
                &mut scratch.standing,
                [p.position.x, p.position.y, p.position.z],
                p.ground.on_ground,
                f64::from(p.dimensions.width) / 2.0,
            );
        }
        for (at, on_ground) in self.lan_guest_feet() {
            standing_feet(&mut scratch.standing, at, on_ground, 0.3);
        }
        {
            use entity::{Entity as _, entity_lists_of, for_each_entity};
            let lists = entity_lists_of!(self);
            for_each_entity!(lists, e => {
                // A kind that `applyEffectsFromBlocks` never reaches says so
                // (`sweeps_blocks_inside`); the carts, swept with the players
                // below, say so too.
                let at = e.position();
                if !e.is_removed()
                    && e.is_alive()
                    && e.sweeps_blocks_inside()
                    && entity_ticking(at.x.floor() as i32, at.z.floor() as i32)
                {
                    let b = e.bounds();
                    standing_feet(
                        &mut scratch.standing,
                        [at.x, at.y, at.z],
                        e.on_ground(),
                        (b.max_position().x - b.min_position().x) / 2.0,
                    );
                }
            });
            // AbstractArrow.applyEffectsFromBlocks reaches a trident in the
            // ground too: it is off the box sweep only because `tick_blocks`
            // takes a trident's blocks from its path (`arrows::blocks_inside`).
            for t in lists.tridents {
                let at = t.position;
                if !t.is_removed()
                    && t.is_alive()
                    && entity_ticking(at.x.floor() as i32, at.z.floor() as i32)
                {
                    let b = t.bounds();
                    standing_feet(
                        &mut scratch.standing,
                        [at.x, at.y, at.z],
                        t.on_ground,
                        (b.max_position().x - b.min_position().x) / 2.0,
                    );
                }
            }
        }
        for c in &self.carts {
            let at = c.position;
            if entity_ticking(at.x.floor() as i32, at.z.floor() as i32) {
                standing_feet(&mut scratch.standing, [at.x, at.y, at.z], c.on_ground, 0.49);
            }
        }
        let first_contact_drop = self.next_entity + 1;
        self.with_level(|b, level, ticks| {
            // The palettes' answers kept from the last tick may be for other
            // behaviours (a reload) or another level.
            scratch.sweep.clear();
            // LilyPadBlock.entityInside: a boat inside a lily pad breaks it.
            scratch.boat_cells.clear();
            for bounds in level.boat_trigger_boxes() {
                let (lo, hi) = (bounds.min_position(), bounds.max_position());
                for x in (lo.x + 1e-7).floor() as i32..=(hi.x - 1e-7).floor() as i32 {
                    for y in (lo.y + 1e-7).floor() as i32..=(hi.y - 1e-7).floor() as i32 {
                        for z in (lo.z + 1e-7).floor() as i32..=(hi.z - 1e-7).floor() as i32 {
                            scratch.boat_cells.push(BlockPos::new(x, y, z));
                        }
                    }
                }
            }
            for pos in scratch.boat_cells.drain(..) {
                b.boat_inside(level, ticks, pos);
            }
            // The mobs and the other entities (items, orbs, projectiles...) too.
            scratch.touched.clear();
            use entity::{Entity as _, for_each_entity};
            let mut touch = |bounds: Aabb| {
                let (lo, hi) = (bounds.min_position(), bounds.max_position());
                let cells = box_cells([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]);
                scratch.touched.push(cells);
            };
            for_each_entity!(level.entities(), entity => {
                if entity.triggers_blocks() && entity.sweeps_blocks_inside() {
                    let at = entity.position();
                    if entity_ticking(at.x.floor() as i32, at.z.floor() as i32) {
                        touch(entity.bounds());
                    }
                }
            });
            for crystal in level.crystals {
                touch(crystal.bounds());
            }
            for &(bounds, applies) in &level.trigger_hooks {
                if applies {
                    touch(bounds);
                }
            }
            for &bounds in &level.trigger_extras {
                touch(bounds);
            }
            // The host's, the minecarts' and the LAN guests' boxes, cell by
            // cell. At a block that senses no entity, `entity_inside` only
            // reads the state: a box whose sections hold no sensing state in
            // their palettes when its turn comes changes nothing at any of its
            // cells, so it is passed over.
            for &cells in &scratch.inside_boxes {
                if !scratch.sweep.may_hold(level.world, b, cells, SENSING) {
                    continue;
                }
                let (lo, hi) = cells;
                for x in lo[0]..=hi[0] {
                    for y in lo[1]..=hi[1] {
                        for z in lo[2]..=hi[2] {
                            b.entity_inside(level, ticks, BlockPos::new(x, y, z));
                        }
                    }
                }
            }
            for pos in scratch.inside.drain(..) {
                b.entity_inside(level, ticks, pos);
            }
            // The touched blocks go once each, in (x, y, z) order. Only an
            // entity-sensing block does anything there, so the others are
            // passed over while no block has changed since they were looked
            // at (they are still not sensing), and only the few sensing ones
            // are sorted. Once a block changes, every later one goes, as the
            // whole sorted list would. Looking for the sensing ones changes
            // nothing, so a box whose sections hold no sensing state in their
            // palettes has none to find: its cells are not read.
            let key = |p: &BlockPos| (p.x, p.y, p.z);
            scratch.sensing.clear();
            for &cells in &scratch.touched {
                if !scratch.sweep.may_hold(level.world, b, cells, SENSING) {
                    continue;
                }
                let (lo, hi) = cells;
                for x in lo[0]..=hi[0] {
                    for y in lo[1]..=hi[1] {
                        for z in lo[2]..=hi[2] {
                            if b.senses_entities(level.world.block(x, y, z)) {
                                scratch.sensing.push(BlockPos::new(x, y, z));
                            }
                        }
                    }
                }
            }
            scratch.sensing.sort_unstable_by_key(key);
            scratch.sensing.dedup();
            let looked = (
                level.world.revision,
                level.world.chunks.membership_revision(),
            );
            let mut i = 0;
            while i < scratch.sensing.len() {
                let pos = scratch.sensing[i];
                b.entity_inside(level, ticks, pos);
                i += 1;
                let now = (
                    level.world.revision,
                    level.world.chunks.membership_revision(),
                );
                if now != looked {
                    // Every touched cell after `pos`, sensing or not, as the
                    // sorted list of them all has it: the sensing ones up to
                    // `pos` went already.
                    scratch.later.clear();
                    for &(lo, hi) in &scratch.touched {
                        for x in lo[0]..=hi[0] {
                            for y in lo[1]..=hi[1] {
                                for z in lo[2]..=hi[2] {
                                    let cell = BlockPos::new(x, y, z);
                                    if key(&cell) > key(&pos) {
                                        scratch.later.push(cell);
                                    }
                                }
                            }
                        }
                    }
                    scratch.later.sort_unstable_by_key(key);
                    scratch.later.dedup();
                    for pos in scratch.later.drain(..) {
                        b.entity_inside(level, ticks, pos);
                    }
                    break;
                }
            }
            // BigDripleafBlock.entityInside at the feet: anywhere but at a
            // leaf it only reads the state, so a feet layer whose sections
            // hold no leaf in their palettes when its turn comes (after the
            // leaves the ones before tilted) is passed over.
            for ((lo, hi), on_ground, feet_y) in scratch.standing.drain(..) {
                if !scratch.sweep.may_hold(level.world, b, (lo, hi), LEAF) {
                    continue;
                }
                for x in lo[0]..=hi[0] {
                    for z in lo[2]..=hi[2] {
                        let pos = BlockPos::new(x, lo[1], z);
                        b.entity_on_big_dripleaf(level, ticks, pos, on_ground, feet_y);
                    }
                }
            }
        });
        self.process_level_events()?;
        self.contact_drops = first_contact_drop..self.next_entity + 1;
        laps.lap("tb_entity_inside");
        // `ServerLevel.tick` sets `handlingTick` before its block ticks and
        // clears it after `runBlockEvents` (ServerLevel.java:360, :412), so
        // every phase below runs inside it: the scheduled ticks (:394-396),
        // the random ticks of `ServerChunkCache.tick` (:406, whose broadcast is
        // `lan_broadcast_changes`) and the block events (:409). A piston that
        // loses its signal in any of them answers `checkIfExtend`'s
        // `isHandlingTick()` (PistonBaseBlock.java:108) with event 2
        // (`TRIGGER_DROP`) when the block it pushed is still in flight, as
        // vanilla does. One flag for the whole tick, not one per phase: phases
        // between them (the broadcast, an eager save, the level events of the
        // ticks already run) are inside the same span in vanilla too.
        //
        // The phases vanilla runs above :394 but before the bracket here (the
        // world border, the weather and the clock, `tickTime` with its
        // scheduled data pack functions, :365-388 and :485; the raids, :402)
        // run with the flag down: a function that sets a block there still
        // reaches a piston with event 1 where vanilla answers 2 (see
        // `Ticks::begin_handling_tick`, which names the gap).
        //
        // The entity sweep above is the tick's entity pass in vanilla
        // (:423-453), moved here to run before the blocks it presses; it
        // stays outside the flag, as do the entities and the block entities
        // (`tick_moving_pistons`, `tick_block_entities`, vanilla's :455) that
        // tick below :412.
        self.block_ticks.begin_handling_tick();
        // ServerLevel.tick: a Debug world runs no scheduled block or fluid ticks, and
        // ServerChunkCache.tickChunks no random ticks.
        let debug = self.world_debug;
        if !debug {
            // A failed tick still leaves the flag down behind it.
            if let Err(e) = self.tick_scheduled() {
                self.block_ticks.end_handling_tick();
                return Err(e);
            }
        }
        horizon_threads::work::clear_work();
        self.with_level(|b, level, ticks| {
            // The masks the random ticks will read, built on helper threads first while the
            // parallel tick is on (`HORIZON_PARALLEL_TICK`): the draws and ticks don't depend on them.
            // With the parallel tick on, the chunks draw on streams of their own and the pass is
            // planned on the helpers instead (`horizon_client::random_tick_plan`).
            let planned = !debug && horizon_client::parallel::random_ticks_own_random();
            if !debug && !planned && horizon_client::parallel::lanes() > 1 {
                horizon_client::spans::time("random_tick_masks", || {
                    let chunks = &level.world.chunks;
                    level.random_tick_sections.prepare_masks(
                        &scratch.ticking,
                        |key| chunks.get(&key).cloned(),
                        level.facts,
                        &|id| b.random_tick_candidate(BlockStateId(id)),
                    );
                });
            }
            horizon_client::spans::time("random_ticks", || {
                // Each chunk's ticking sections, read once and answered from while no block
                // changes and no chunk comes or goes; the mask itself is retained by
                // tick_chunk_random_masked between callback asks.
                if planned {
                    let speed = level.game_rules().random_tick_speed;
                    if speed > 0 {
                        let chunks = &level.world.chunks;
                        let columns: Vec<_> = (scratch.ticking.iter())
                            .map(|&key| (key, chunks.get(&key).cloned()))
                            .collect();
                        let facts = level.facts;
                        let range = (level.min_y(), level.max_y());
                        let plan = horizon_client::random_tick_plan::plan(
                            &columns,
                            level.game_time(),
                            speed,
                            range,
                            &|id| facts.get(BlockStateId(id)).random_ticks,
                            &|id| b.random_tick_candidate(BlockStateId(id)),
                        );
                        drop(columns);
                        let mut chunk = None;
                        for (key, planned) in plan {
                            if chunk != Some(key) {
                                horizon_threads::work::set_chunk(key.0, key.1);
                                chunk = Some(key);
                            }
                            b.run_planned_random_tick(level, ticks, planned);
                        }
                    }
                    scratch.ticking.clear();
                    return;
                }
                let mut sections = horizon_client::random_tick_sections::ChunkTicking::default();
                for (n, &(x, z)) in scratch.ticking.iter().enumerate().filter(|_| !debug) {
                    // The next chunks' first reads, read together (`RandomTickSections::read_ahead`):
                    // their entries here and in the chunk map.
                    if n % READ_AHEAD == 0 {
                        let ahead = &scratch.ticking[n..(n + READ_AHEAD).min(scratch.ticking.len())];
                        let chunks = &level.world.chunks;
                        let read = level.random_tick_sections.read_ahead(ahead);
                        let found = ahead
                            .iter()
                            .filter_map(|key| chunks.get(key))
                            .fold(0usize, |sum, chunk| sum ^ std::sync::Arc::as_ptr(chunk) as usize);
                        std::hint::black_box(read ^ found);
                    }
                    let pos = horizon_core::position::ChunkPos::new(x, z);
                    sections.start((x, z));
                    b.tick_chunk_random_masked(level, ticks, pos, |level, from| {
                        let stamp = (
                            level.world.revision,
                            level.world.chunks.membership_revision(),
                        );
                        let chunks = &level.world.chunks;
                        sections.next(
                            level.random_tick_sections,
                            || chunks.get(&(x, z)),
                            level.facts,
                            |id| b.random_tick_candidate(BlockStateId(id)),
                            from,
                            stamp,
                        )
                    });
                }
                scratch.ticking.clear();
            });
        });
        horizon_threads::work::clear_work();
        laps = horizon_client::spans::laps();
        let sent = self.lan_broadcast_changes();
        laps.lap("tb_broadcast");
        self.save_chunks_eagerly();
        self.with_level(|b, level, ticks| {
            b.run_block_events(level, ticks);
        });
        laps.lap("tb_block_events");
        // `ServerLevel`: `handlingTick = false` right after `runBlockEvents`
        // (:412), before the entities and the block entities.
        self.block_ticks.end_handling_tick();
        self.process_level_events()?;
        laps.lap("tb_level_events");
        self.sync_moving_pistons();
        laps.lap("tb_pistons");
        Ok(sent)
    }

    /// The ticks shown by this frame; still mode follows world time.
    /// `ChunkMap.forEachBlockTickingChunk`: the chunks that random tick this
    /// tick, empty ones included (`tickChunk` still draws for precipitation in
    /// those).
    ///
    /// Vanilla walks the simulation tracker's own table, whose order doesn't
    /// follow chunk loading. The world map's order does: it is filled as the
    /// streaming workers finish, so the level random's draws landed on other
    /// blocks from one run of a world to the next (#15164). Chunk position
    /// order is the same every run; vanilla's exact order is its fastutil
    /// table's slot order, which follows that table's whole history.
    pub(super) fn random_tick_chunks(&self, out: &mut Vec<(i32, i32)>) {
        let levels = self.tickets.levels();
        let bounds = levels.random_tick_bounds();
        out.clear();
        // The levels give the chunks that random tick already in order: the resident ones among
        // them are the same list the scan of every resident chunk and its sort make.
        if levels.random_ticking_sorted(out) {
            out.retain(|&(x, z)| ticking_chunk(&self.world, x, z));
            return;
        }
        out.extend(self.world.chunks.keys().copied().filter(|&(x, z)| {
            if let Some((min_x, min_z, max_x, max_z)) = bounds
                && (x < min_x || x > max_x || z < min_z || z > max_z)
            {
                return false;
            }
            levels.random_ticking(x, z) && ticking_chunk(&self.world, x, z)
        }));
        out.sort_unstable();
    }
    /// Track which chunks are ticking (Ticks needs chunk load/unload).
    pub(super) fn sync_tick_chunks(&mut self) {
        if !self.tick_chunks.synchronize(&self.world.chunks) {
            return;
        }
        // Move only the added keys out so registration can borrow all of Game.
        let added = self.tick_chunks.take_added();
        for &key in &added {
            self.register_chunk_ticks(key);
        }
        for &key in self.tick_chunks.removed() {
            self.random_tick_sections.remove(&key);
            self.portals.sections.remove(&key);
            self.block_ticks
                .remove_chunk(horizon_core::position::ChunkPos::new(key.0, key.1));
            self.block_ticks
                .comparator_outputs
                .retain(|p, _| (p.x >> 4, p.z >> 4) != key);
            self.hoppers.retain(|p, _| (p.0 >> 4, p.2 >> 4) != key);
            // Its moving pistons were saved with it and load back with it.
            self.block_ticks
                .moving_pistons
                .retain(|(p, _)| (p.x >> 4, p.z >> 4) != key);
        }
        self.tick_chunks.recycle_added(added);
    }

    /// `ServerLevel.tick`: moving block entities push after entity travel.
    pub(super) fn tick_moving_pistons(&mut self) -> Result<()> {
        self.with_level(|b, level, ticks| b.tick_block_entities(level, ticks));
        let pushes = std::mem::take(&mut self.block_ticks.piston_pushes);
        self.push_piston_entities(pushes);
        self.process_level_events()?;
        self.sync_moving_pistons();
        Ok(())
    }

    /// `ServerLevel.tick`'s scheduled block then fluid ticks, one at a time:
    /// the side effects of each (a shrieker summoning a warden, drops, game
    /// events) happen before the next tick runs, as they do inline in vanilla.
    pub(super) fn tick_scheduled(&mut self) -> Result<()> {
        for fluids in [false, true] {
            let mut laps = horizon_client::spans::laps();
            let name = if fluids { "tb_fluids" } else { "tb_scheduled" };
            if !self.with_level(|b, level, ticks| b.scheduled_begin(level, ticks, fluids)) {
                laps.lap(name);
                continue;
            }
            loop {
                // Due ticks run against one level while each leaves nothing for
                // the game to apply (level events, spilled items or orbs): the
                // next one would see the same level built again.
                let more = self.with_level(|b, level, ticks| {
                    loop {
                        let more = b.scheduled_step(level, ticks, fluids);
                        if !more || !level.events.is_empty() || !level.spill.is_empty() {
                            break more;
                        }
                    }
                });
                if !self.level_events.is_empty() {
                    self.process_level_events()?;
                }
                if !more {
                    break;
                }
            }
            self.with_level(|b, _, ticks| b.scheduled_finish(ticks, fluids));
            laps.lap(name);
        }
        Ok(())
    }
}

/// The blocks the feet layer of an entity centred at `at` and `w` wide is
/// inside, as one layer of cells (walked x first, then z), with whether it
/// stands on the ground and how high its feet are: what
/// `BigDripleafBlock.canEntityTilt` reads. Vanilla's `checkInsideBlocks`
/// sweeps the whole deflated box, but `canEntityTilt` needs the feet above the
/// leaf's 0.6875 and a box starts at its feet, so only a leaf in the feet
/// layer can pass it.
pub(in crate::game) fn standing_feet(
    out: &mut Vec<(CellBox, bool, f64)>,
    at: [f64; 3],
    on_ground: bool,
    w: f64,
) {
    let epsilon = f64::from(1.0e-5_f32);
    let y = (at[1] + epsilon).floor() as i32;
    let cell = |v: f64| v.floor() as i32;
    let (x0, x1) = (cell(at[0] - w + epsilon), cell(at[0] + w - epsilon));
    let (z0, z1) = (cell(at[2] - w + epsilon), cell(at[2] + w - epsilon));
    out.push((([x0, y, z0], [x1, y, z1]), on_ground, at[1]));
}

/// The cells `lo..=hi` of a box, both corners included (none along an axis
/// where `hi` is below `lo`), walked x first, then y, then z.
pub(in crate::game) type CellBox = ([i32; 3], [i32; 3]);

/// The cells the sweep of `tick_blocks` walks for a box from `lo` to `hi`:
/// each corner moved 1e-7 inwards, then floored.
pub(in crate::game) fn box_cells(lo: [f64; 3], hi: [f64; 3]) -> CellBox {
    (
        lo.map(|v| (v + 1e-7).floor() as i32),
        hi.map(|v| (v - 1e-7).floor() as i32),
    )
}

/// [`SweepSections`] bit: a state [`Behaviours::senses_entities`] says yes to,
/// where `entity_inside` may do anything.
const SENSING: u8 = 1;
/// [`SweepSections`] bit: a big dripleaf, which `entity_on_big_dripleaf` may tilt.
const LEAF: u8 = 2;
/// [`SweepSections::states`]: the state's bits are known.
const KNOWN: u8 = 4;

/// `World::may_hold` for the entity sweep of `tick_blocks`, for both of its
/// questions at once (a block sensing entities, a big dripleaf): each
/// section's palette is read once, and its answer kept while no block changes
/// and no chunk comes or goes (the stamp the sweep's sensing blocks are looked
/// at with), for the boxes of the other entities in that section and for each
/// entity's feet after its box. Cleared every tick: the behaviours may change.
#[derive(Default)]
pub(in crate::game) struct SweepSections {
    /// Per section, `(chunk x, section index, chunk z)`: the bits a state of
    /// its palette may answer to.
    sections: FxHashMap<(i32, i32, i32), u8>,
    /// Per block state: its bits with [`KNOWN`], or 0 before it is asked.
    states: Vec<u8>,
    /// The world's `(revision, membership_revision)` the sections were read at.
    stamp: Option<(u64, u64)>,
    /// Every box may hold anything: every cell is read, as before the memo.
    #[cfg(test)]
    pub(in crate::game) read_every_cell: bool,
    /// How many boxes were passed over.
    #[cfg(test)]
    pub(in crate::game) passed_over: usize,
}

impl SweepSections {
    /// Forgets every answer, for a new tick.
    fn clear(&mut self) {
        self.sections.clear();
        self.states.clear();
        self.stamp = None;
    }
    /// The bits of `state`.
    fn state(&mut self, b: &Behaviours, state: BlockStateId) -> u8 {
        let i = state.0 as usize;
        if i >= self.states.len() {
            self.states.resize(i + 1, 0);
        }
        if self.states[i] == 0 {
            let sensing = if b.senses_entities(state) { SENSING } else { 0 };
            let leaf = if b.is_big_dripleaf(state) { LEAF } else { 0 };
            self.states[i] = KNOWN | sensing | leaf;
        }
        self.states[i] & !KNOWN
    }
    /// The bits of section `s` of the chunk at `(cx, cz)`, as `World::may_hold`
    /// reads it: a missing chunk is air, a palette of the whole registry may
    /// hold anything, and an entry no block uses any more still counts.
    fn section(&mut self, world: &World, b: &Behaviours, (cx, s, cz): (i32, i32, i32)) -> u8 {
        if let Some(&bits) = self.sections.get(&(cx, s, cz)) {
            return bits;
        }
        let bits = match world.chunks.get(&(cx, cz)) {
            None => self.state(b, world.air),
            // The level's chunks hold its column (`World::shape`); one that
            // did not would have its cells read.
            Some(chunk) if chunk.layout() != world.layout() => SENSING | LEAF,
            Some(chunk) => match chunk.sections.get(s as usize) {
                Some(section) if !section.blocks.is_global() => {
                    let mut bits = 0;
                    section.blocks.maybe_has(|v| {
                        bits |= self.state(b, BlockStateId(v));
                        bits == SENSING | LEAF
                    });
                    bits
                }
                _ => SENSING | LEAF,
            },
        };
        self.sections.insert((cx, s, cz), bits);
        bits
    }
    /// Whether a block of the cells `lo..=hi` may answer to `bits`, read now:
    /// `World::may_hold` of [`Self::state`]'s questions. `false` means that no
    /// cell holds such a block at this moment.
    fn may_hold(&mut self, world: &World, b: &Behaviours, (lo, hi): CellBox, bits: u8) -> bool {
        #[cfg(test)]
        {
            if self.read_every_cell {
                return true;
            }
        }
        // The answers kept are for the world as it was read: once a block is
        // set (by the sweep itself) or a chunk comes or goes, all are read
        // again.
        let now = (world.revision, world.chunks.membership_revision());
        if self.stamp != Some(now) {
            self.sections.clear();
            self.stamp = Some(now);
        }
        let layout = world.layout();
        let (bottom, top) = (layout.min_y(), layout.max_y() - 1);
        // Heights outside the sections read as state 0, or as air without a chunk.
        if (lo[1] < bottom || hi[1] > top)
            && (self.state(b, BlockStateId(0)) | self.state(b, world.air)) & bits != 0
        {
            return true;
        }
        let sections = (lo[1].max(bottom) - bottom) >> 4..=(hi[1].min(top) - bottom) >> 4;
        for cx in lo[0] >> 4..=hi[0] >> 4 {
            for cz in lo[2] >> 4..=hi[2] >> 4 {
                for s in sections.clone() {
                    if self.section(world, b, (cx, s, cz)) & bits != 0 {
                        return true;
                    }
                }
            }
        }
        #[cfg(test)]
        {
            self.passed_over += 1;
        }
        false
    }
}

#[cfg(test)]
mod sweep_sections_tests {
    use super::*;

    /// [`SweepSections::may_hold`] answers as `World::may_hold` does for its
    /// two questions, over section and chunk edges, above and below the world
    /// and where no chunk is, and reads a section again once a block is set:
    /// an answer given before an effect of the sweep is not kept after it.
    #[test]
    #[ignore = "daily"]
    fn sweep_sections_answer_as_world_may_hold() {
        let mut resources = None;
        let mut game = crate::game::game_tests::tests::server_game(&mut resources);
        let registry = game.pack.assets.registry;
        let plate = registry
            .default_state("minecraft:stone_pressure_plate")
            .unwrap();
        let leaf = registry.default_state("minecraft:big_dripleaf").unwrap();
        for (x, z) in [(0, 0), (1, 0)] {
            let chunk = std::sync::Arc::new(horizon_client::chunk::Chunk::empty(x, z));
            game.world.chunks.insert((x, z), chunk);
        }
        // A plate in chunk (0, 0)'s section 64..80, a leaf in chunk (1, 0)'s.
        game.world.set(3, 64, 3, plate).unwrap();
        game.world.set(20, 70, 5, leaf).unwrap();
        let b = game.res.behaviours();
        let mut sweep = SweepSections::default();
        let boxes: [CellBox; 10] = [
            ([2, 63, 2], [3, 65, 3]),
            ([5, 64, 5], [6, 65, 6]),
            ([5, 79, 5], [5, 80, 5]),
            ([2, 40, 2], [3, 41, 3]),
            ([14, 64, 2], [17, 65, 3]),
            ([20, 70, 5], [20, 70, 5]),
            ([-1, 64, -1], [0, 65, 0]),
            ([40, 64, 0], [41, 65, 1]),
            ([0, 318, 0], [1, 330, 1]),
            ([0, -80, 0], [1, -70, 1]),
        ];
        for cells in boxes {
            let (lo, hi) = cells;
            let sensing = game.world.may_hold(lo, hi, |s| b.senses_entities(s));
            let leaves = game.world.may_hold(lo, hi, |s| b.is_big_dripleaf(s));
            assert_eq!(
                sweep.may_hold(&game.world, b, cells, SENSING),
                sensing,
                "{cells:?}"
            );
            assert_eq!(
                sweep.may_hold(&game.world, b, cells, LEAF),
                leaves,
                "{cells:?}"
            );
        }
        // Not every answer is yes: the plate's section and the leaf's only.
        assert!(sweep.may_hold(&game.world, b, boxes[0], SENSING));
        assert!(!sweep.may_hold(&game.world, b, boxes[0], LEAF));
        assert!(sweep.may_hold(&game.world, b, boxes[5], LEAF));
        assert!(!sweep.may_hold(&game.world, b, boxes[3], SENSING));
        assert!(!sweep.may_hold(&game.world, b, boxes[7], SENSING));
        // A section answered for is read again once a block is set in it.
        let high = ([4, 100, 4], [5, 101, 5]);
        assert!(!sweep.may_hold(&game.world, b, high, SENSING));
        game.world.set(4, 100, 4, plate).unwrap();
        assert!(sweep.may_hold(&game.world, b, high, SENSING));
    }
}
