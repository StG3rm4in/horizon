//! The survival scene of the exactness harness (`tools/tick_digest.py --scene survival`): how a
//! fixture player plays out of creative mode (`Survivor`). It walks on the floors the scene built,
//! keeping their height and claiming the ground, as a client standing on a block does: the arena
//! by the world spawn, the Nether's roof and a room under it, a platform of the End. It hits the
//! nearest monster in reach and respawns when it dies. What it knows of the entities around it and
//! of its death it reads from what it is sent (`Survivor::observe`); what the server waits for (the
//! level's load, the teleport, the chunk batches) it takes from the server's state, as a creative
//! one does.

use horizon_client::lan::guest::Guest;
use horizon_protocol::{
    buf::Reader,
    ids::play::clientbound as ids,
    packets::{
        Empty,
        play::{self, Clientbound as C, Serverbound as S},
    },
};
use std::collections::BTreeMap;

/// What a survivor takes for a monster, by entity type: those it may meet by the arena, in the
/// Nether and in the End. Not the villager or the iron golem: an angry golem ends every fight.
pub(super) const HOSTILE: &[&str] = &[
    "minecraft:blaze",
    "minecraft:bogged",
    "minecraft:cave_spider",
    "minecraft:creeper",
    "minecraft:drowned",
    "minecraft:enderman",
    "minecraft:endermite",
    "minecraft:hoglin",
    "minecraft:husk",
    "minecraft:magma_cube",
    "minecraft:phantom",
    "minecraft:piglin",
    "minecraft:piglin_brute",
    "minecraft:silverfish",
    "minecraft:skeleton",
    "minecraft:slime",
    "minecraft:spider",
    "minecraft:stray",
    "minecraft:witch",
    "minecraft:wither_skeleton",
    "minecraft:zoglin",
    "minecraft:zombie",
    "minecraft:zombie_villager",
    "minecraft:zombified_piglin",
];

/// The arena (`SURVIVAL_COMMANDS` in `tools/tick_digest.py`), in blocks from the middle of the
/// world spawn's block, where survivors join and respawn: nothing that collides stands within
/// `RING` of it along either axis, so any straight walk inside the ring is clear. The ring's
/// spokes run east, south, west and north; the corner after each, clockwise, is `corner`'s.
const RING: f64 = 5.0;
const SPOKES: [[f64; 2]; 4] = [[RING, 0.0], [0.0, RING], [-RING, 0.0], [0.0, -RING]];
/// Where the guards stand: by the spawner, the trial spawner, the vault and the furnace.
const POSTS: [[f64; 2]; 4] = [[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0], [5.0, 5.0]];
/// Where the standers stand, in the ring's hazards: a cobweb, a sweet berry bush, powder snow,
/// magma, a wither rose, an open eyeblossom, fire, the stone pressure plate.
const HAZARDS: [[f64; 2]; 8] = [
    [-3.0, -5.0],
    [2.0, -5.0],
    [5.0, -2.0],
    [5.0, 2.0],
    [2.0, 5.0],
    [-2.0, 5.0],
    [-5.0, -2.0],
    [0.0, -5.0],
];
/// The corners of the square walked in the Nether's room and on the End's platform.
const SQUARE: [[f64; 2]; 4] = [[3.0, 3.0], [-3.0, 3.0], [-3.0, -3.0], [3.0, -3.0]];
/// Blocks a tick: a walk (a client walks 0.216), a chase, a roam over the Nether's roof.
const WALK: f64 = 0.2;
const CHASE: f64 = 0.25;
const ROAM: f64 = 0.25;
/// A blow on one tick in `STRIKE_EVERY`, each survivor on its own (a fist recharges in 5), at a
/// monster `REACH` blocks across and 2 up or down at most: within the server's 3 from the eyes to
/// its box.
const STRIKE_EVERY: u64 = 10;
const REACH: f64 = 2.5;
/// The death screen enables its respawn button 20 ticks after the death.
const RESPAWN_AFTER: u64 = 20;

/// A survivor's walk, at the height of the floor it was placed on.
#[derive(Clone, Copy)]
enum Walk {
    /// Round the arena's patrol ring, in by one of its spokes.
    Ring { spoke: usize, clockwise: bool },
    /// After the nearest monster inside the ring, else back to a spot of its own.
    Chase([f64; 2]),
    /// To a spot, then it stands there.
    Stand([f64; 2]),
    /// Straight on: the Nether's roof is one flat layer of bedrock.
    Roam,
    /// Round a square of 3 blocks about where it was placed (the Nether's room, the End's
    /// platform).
    Square,
}

/// A fixture player out of creative mode (see the module docs).
pub(super) struct Survivor {
    /// Its place in the joining order: its walk in the Overworld, the tick of its blows.
    index: usize,
    /// `observe` keeps what it reads: not in creative mode, where it plays no scene.
    pub(super) watching: bool,
    loaded_sent: bool,
    accepted: Option<i32>,
    position: Option<[f64; 3]>,
    /// The middle of the arena (x, z): where it first stood in the Overworld.
    arena: Option<[f64; 2]>,
    /// Where its walk is laid out from (x, z), in which level, and the leg it is on.
    anchor: [f64; 2],
    level: String,
    walk: Walk,
    leg: usize,
    /// A `RESPAWN` came (a respawn, another level): the next teleport places it anew.
    relocated: bool,
    /// The entities it was told of, by network id: their type and position.
    seen: BTreeMap<i32, (i32, [f64; 3])>,
    /// The tick it died on (health 0, `PLAYER_COMBAT_KILL`) until a `RESPAWN`, and whether it has
    /// asked to respawn.
    dead_since: Option<u64>,
    respawn_asked: bool,
    /// Its ticks played, and where it looks.
    clock: u64,
    yaw: f32,
}

impl Survivor {
    pub(super) fn new(index: usize) -> Self {
        Self {
            index,
            watching: true,
            loaded_sent: false,
            accepted: None,
            position: None,
            arena: None,
            anchor: [0.0; 2],
            level: String::new(),
            walk: Walk::Stand([0.0; 2]),
            leg: 0,
            relocated: false,
            seen: BTreeMap::new(),
            dead_since: None,
            respawn_asked: false,
            clock: 0,
            yaw: 0.0,
        }
    }

    /// What a client learns from one body it was sent (VarInt id and payload): the entities
    /// around it (`ADD_ENTITY`, their moves and teleports, `REMOVE_ENTITIES`, all gone at a
    /// `RESPAWN`) and its own death (`SET_HEALTH` at 0, `PLAYER_COMBAT_KILL`).
    pub(super) fn observe(&mut self, body: &[u8]) {
        if !self.watching {
            return;
        }
        let mut reader = Reader::new(body);
        let Ok(id) = reader.varint() else {
            return;
        };
        // Only these are decoded: most bodies say nothing of this, and a chunk's is large.
        if !matches!(
            id,
            ids::ADD_ENTITY
                | ids::MOVE_ENTITY_POS
                | ids::MOVE_ENTITY_POS_ROT
                | ids::ENTITY_POSITION_SYNC
                | ids::TELEPORT_ENTITY
                | ids::REMOVE_ENTITIES
                | ids::RESPAWN
                | ids::SET_HEALTH
                | ids::PLAYER_COMBAT_KILL
        ) {
            return;
        }
        let Ok(packet) = C::decode(id, reader.rest()) else {
            return;
        };
        match packet {
            C::AddEntity(p) => {
                self.seen.insert(p.id, (p.kind, p.position));
            }
            C::MoveEntityPos(play::MoveEntityPos(m))
            | C::MoveEntityPosRot(play::MoveEntityPosRot(m)) => {
                if let (Some(delta), Some(seen)) = (&m.delta, self.seen.get_mut(&m.id)) {
                    seen.1 = delta.end(seen.1);
                }
            }
            C::EntityPositionSync(p) => {
                if let Some(seen) = self.seen.get_mut(&p.id) {
                    seen.1 = p.path.end();
                }
            }
            C::TeleportEntity(p) => {
                if let Some(seen) = self.seen.get_mut(&p.id) {
                    let relative = [play::relative::X, play::relative::Y, play::relative::Z];
                    for (k, flag) in relative.into_iter().enumerate() {
                        seen.1[k] = if p.relatives & flag != 0 {
                            seen.1[k] + p.position[k]
                        } else {
                            p.position[k]
                        };
                    }
                }
            }
            C::RemoveEntities(p) => {
                for id in p.ids {
                    self.seen.remove(&id);
                }
            }
            C::Respawn(_) => {
                self.seen.clear();
                self.relocated = true;
                self.dead_since = None;
                self.respawn_asked = false;
            }
            C::SetHealth(p) if p.health > 0.0 => {}
            C::SetHealth(_) | C::PlayerCombatKill(_) => {
                self.dead_since.get_or_insert(self.clock);
            }
            _ => {}
        }
    }

    /// One tick, before the server's: what the server waits for, then, alive, a blow, a step of
    /// its walk and its move.
    pub(super) fn play(&mut self, guest: &Guest, hostile: &[i32], packets: &mut Vec<S>) {
        self.clock += 1;
        // The level's load, at the join and after each respawn (`Guest::loading`).
        if !self.loaded_sent || guest.loading {
            packets.push(S::PlayerLoaded(Empty));
            self.loaded_sent = true;
        }
        if let Some((id, at)) = guest.awaiting_teleport
            && self.accepted != Some(id)
        {
            packets.push(S::AcceptTeleportation(play::AcceptTeleportation {
                id,
                x: at[0],
                y: at[1],
                z: at[2],
                yaw: 0.0,
                pitch: 0.0,
            }));
            self.accepted = Some(id);
            // The join, a respawn or another level places it; any other teleport (a move the
            // server refused) only puts it back on its way.
            if self.position.is_none() || self.relocated || guest.dimension != self.level {
                self.place(at, &guest.dimension);
            }
            self.position = Some(at);
        }
        for _ in 0..guest.unacknowledged {
            packets.push(S::ChunkBatchReceived(play::ChunkBatchReceived {
                chunks_per_tick: 64.0,
            }));
        }
        if let Some(died) = self.dead_since {
            // The death screen: its respawn button, once enabled.
            if !self.respawn_asked && self.clock >= died + RESPAWN_AFTER {
                packets.push(S::ClientCommand(play::ClientCommand { action: 0 }));
                self.respawn_asked = true;
            }
            packets.push(S::ClientTickEnd(Empty));
            return;
        }
        let Some(mut pos) = self.position else {
            return;
        };
        self.strike(pos, hostile, packets);
        let here = ((pos[0].floor() as i32) >> 4, (pos[2].floor() as i32) >> 4);
        if guest.sent_chunks.contains(&here) {
            self.walk(&mut pos, hostile);
        }
        self.position = Some(pos);
        packets.push(S::MovePlayerPosRot(play::MovePlayerPosRot {
            x: pos[0],
            y: pos[1],
            z: pos[2],
            yaw: self.yaw,
            pitch: 0.0,
            // On the ground: the scene laid a floor under every walk.
            flags: 1,
        }));
        packets.push(S::ClientTickEnd(Empty));
    }

    /// Lays out its walk from where a teleport put it, by its level and its index.
    fn place(&mut self, at: [f64; 3], level: &str) {
        let i = self.index;
        self.level = level.to_owned();
        self.relocated = false;
        self.anchor = [at[0], at[2]];
        self.walk = match level {
            "minecraft:overworld" => {
                // Every walk there is laid out from the arena's middle, the respawn point.
                self.anchor = *self.arena.get_or_insert(self.anchor);
                match i % 5 {
                    0 | 1 => Walk::Ring {
                        spoke: (i / 5) % 4,
                        clockwise: i.is_multiple_of(5),
                    },
                    // Its own spot when no monster is near, spread over the ring's inside.
                    2 => {
                        let spot = [(i / 5) % 5, (i / 25) % 4].map(|k| k as f64 * 2.0);
                        Walk::Chase([spot[0] - 4.0, spot[1] - 3.0])
                    }
                    3 => Walk::Stand(POSTS[(i / 5) % 4]),
                    _ => Walk::Stand(HAZARDS[(i / 5) % 8]),
                }
            }
            // The roof: bedrock at y 127 all over.
            "minecraft:the_nether" if at[1] >= 128.0 => Walk::Roam,
            _ => Walk::Square,
        };
        self.leg = match self.walk {
            Walk::Square => i % 4,
            _ => 0,
        };
    }

    /// On its tick in `STRIKE_EVERY`: a blow at the nearest monster in reach (the attack, then the
    /// arm's swing, as a client sends them), or one time in four a swing at the air.
    fn strike(&mut self, pos: [f64; 3], hostile: &[i32], packets: &mut Vec<S>) {
        let tick = self.clock + self.index as u64;
        if !tick.is_multiple_of(STRIKE_EVERY) {
            return;
        }
        match self.nearest(pos, hostile, REACH, |at| (at[1] - pos[1]).abs() <= 2.0) {
            Some((id, at)) => {
                self.yaw = yaw_toward(at[0] - pos[0], at[2] - pos[2]);
                packets.push(S::Attack(play::Attack { id }));
                packets.push(S::Punch(Empty));
            }
            None if tick.is_multiple_of(4 * STRIKE_EVERY) => packets.push(S::Punch(Empty)),
            None => {}
        }
    }

    /// The monster it knows of nearest across, within `range` and taken by `keep`; the lowest id
    /// among equals.
    fn nearest(
        &self,
        pos: [f64; 3],
        hostile: &[i32],
        range: f64,
        keep: impl Fn([f64; 3]) -> bool,
    ) -> Option<(i32, [f64; 3])> {
        let mut best: Option<(f64, i32, [f64; 3])> = None;
        for (&id, &(kind, at)) in &self.seen {
            let across = (at[0] - pos[0]).hypot(at[2] - pos[2]);
            if across <= range
                && hostile.contains(&kind)
                && keep(at)
                && best.is_none_or(|(nearest, ..)| across < nearest)
            {
                best = Some((across, id, at));
            }
        }
        best.map(|(_, id, at)| (id, at))
    }

    /// One tick of its walk, at the height it was placed at.
    fn walk(&mut self, pos: &mut [f64; 3], hostile: &[i32]) {
        let [ax, az] = self.anchor;
        match self.walk {
            Walk::Ring { spoke, clockwise } => {
                // Out along its spoke, then from corner to corner.
                let goal = match (self.leg, clockwise) {
                    (0, _) => SPOKES[spoke],
                    (leg, true) => corner(spoke + leg - 1),
                    (leg, false) => corner(spoke + 4 - leg % 4),
                };
                if self.head_to(pos, [ax + goal[0], az + goal[1]], WALK) {
                    self.leg = self.leg % 4 + 1;
                }
            }
            Walk::Chase(spot) => {
                let anchor = self.anchor;
                let in_ring = |at: [f64; 3]| chebyshev(anchor, at) <= RING + 1.0;
                let goal = match self.nearest(*pos, hostile, 8.0, in_ring) {
                    // In reach: it stays where its blows land.
                    Some((_, at)) if (at[0] - pos[0]).hypot(at[2] - pos[2]) <= 1.2 => return,
                    // Never past the ring, where blocks collide.
                    Some((_, at)) => within(anchor, at, RING),
                    None => [ax + spot[0], az + spot[1]],
                };
                self.head_to(pos, goal, CHASE);
            }
            Walk::Stand(spot) => {
                self.head_to(pos, [ax + spot[0], az + spot[1]], WALK);
            }
            Walk::Roam => {
                // The golden angle apart: the roamers fan out.
                let heading = self.index as f64 * std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
                let (dz, dx) = heading.sin_cos();
                pos[0] += dx * ROAM;
                pos[2] += dz * ROAM;
                self.yaw = yaw_toward(dx, dz);
            }
            Walk::Square => {
                let goal = SQUARE[self.leg % 4];
                if self.head_to(pos, [ax + goal[0], az + goal[1]], WALK) {
                    self.leg += 1;
                }
            }
        }
    }

    /// A step of `speed` at most toward `goal` (x, z), facing its way; true once there.
    fn head_to(&mut self, pos: &mut [f64; 3], goal: [f64; 2], speed: f64) -> bool {
        let (dx, dz) = (goal[0] - pos[0], goal[1] - pos[2]);
        let distance = dx.hypot(dz);
        if distance <= speed {
            pos[0] = goal[0];
            pos[2] = goal[1];
            return true;
        }
        self.yaw = yaw_toward(dx, dz);
        pos[0] += dx / distance * speed;
        pos[2] += dz / distance * speed;
        false
    }
}

/// The ring's corner after spoke `k`, clockwise.
fn corner(k: usize) -> [f64; 2] {
    let (a, b) = (SPOKES[k % 4], SPOKES[(k + 1) % 4]);
    [a[0] + b[0], a[1] + b[1]]
}

/// How far `at` is from `anchor` (x, z) along x or along z, whichever is more.
fn chebyshev(anchor: [f64; 2], at: [f64; 3]) -> f64 {
    (at[0] - anchor[0]).abs().max((at[2] - anchor[1]).abs())
}

/// `at` (x and z), brought back into the square of half side `half` about `anchor`.
fn within(anchor: [f64; 2], at: [f64; 3], half: f64) -> [f64; 2] {
    [
        anchor[0] + (at[0] - anchor[0]).clamp(-half, half),
        anchor[1] + (at[2] - anchor[1]).clamp(-half, half),
    ]
}

/// The yaw of a look along (dx, dz), as the client's `Mth.atan2(dz, dx) * 180 / PI - 90`.
fn yaw_toward(dx: f64, dz: f64) -> f32 {
    (dz.atan2(dx).to_degrees() - 90.0) as f32
}
