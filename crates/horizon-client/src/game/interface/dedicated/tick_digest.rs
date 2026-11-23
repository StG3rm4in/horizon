//! The exactness harness of the server tick (`tools/tick_digest.py`): a digest of every tick, and a
//! fixture mode that takes worker timing out of the tick.
//!
//! The digest (`HORIZON_TICK_DIGEST=FILE`): one line per server tick with hashes of the state once
//! the tick ends (`tick_digest/parts.rs`; docs/tick-digest.md lists the parts): the level in place's
//! (the Overworld while it ticks) under plain names, the server's own (the id and join counters, the
//! players and their files, what the fixture's players were sent, the chunk results applied), then
//! each other level's that ticked, under its name. A level's: its random, its block writes, scheduled
//! ticks, entities with their fields and own randoms, ticking block entities' NBT, its chunks and,
//! every `HORIZON_TICK_DIGEST_FULL` ticks (default 20, 0 for never), the blocks and light of every
//! section and every block entity's NBT. Two runs of one fixture must write the same lines: the first
//! line that differs names the tick and the part. `HORIZON_TICK_DIGEST_DETAIL=FROM-TO` also writes
//! the rows behind the hashes of those ticks to `FILE.detail`, to find what differs (on full ticks, a
//! row per section with the hash of its blocks and of its light).
//!
//! The fixture (`HORIZON_TICK_FIXTURE_TICKS=N`): the server runs N ticks back to back, then stops.
//! Before each tick it runs the console commands of `HORIZON_TICK_FIXTURE_COMMANDS` (lines
//! `TICK COMMAND`, ticks counted from 1) due then, waits for every chunk job in flight and applies
//! the results in a fixed order (`streaming::set_deterministic`), as many rounds as the work needs.
//! The tick itself is unchanged.

use super::super::Game;
use std::{
    hash::{DefaultHasher, Hash, Hasher},
    io::{BufWriter, Write},
};

mod scene;

/// The VarInt at `at` in a packet body, and where the next field starts.
fn varint(body: &[u8], at: usize) -> (u32, usize) {
    let mut value = 0u32;
    for (i, byte) in body.iter().skip(at).take(5).enumerate() {
        value |= u32::from(byte & 0x7F) << (7 * i);
        if byte & 0x80 == 0 {
            return (value, at + i + 1);
        }
    }
    (value, body.len())
}

/// The VarInt id at the start of a packet body (`Out::Packet`: VarInt id and payload).
fn packet_id(body: &[u8]) -> u32 {
    varint(body, 0).0
}

/// What the `packets` part hashes of a body. `None` for the packets whose body holds the clock rather
/// than the game: `UPDATE_ADVANCEMENTS`, whose criteria carry the date they were met. An `ADD_ENTITY`
/// with its UUID zeroed: entity UUIDs are random (`save::random_uuid`, seeded from the clock, one
/// counter for every thread), and the `entities` part leaves them out too. Two runs of one build give
/// the same `packets` once these two are set aside (checked over 600 ticks with 100 players).
fn canonical(body: &[u8]) -> Option<std::borrow::Cow<'_, [u8]>> {
    use horizon_protocol::ids::play::clientbound as ids;
    let (id, after_id) = varint(body, 0);
    if id == ids::UPDATE_ADVANCEMENTS as u32 {
        return None;
    }
    if id == ids::ADD_ENTITY as u32 {
        let (_, uuid) = varint(body, after_id);
        if body.len() >= uuid + 16 {
            let mut masked = body.to_vec();
            masked[uuid..uuid + 16].fill(0);
            return Some(std::borrow::Cow::Owned(masked));
        }
    }
    Some(std::borrow::Cow::Borrowed(body))
}

/// `HORIZON_TICK_DIGEST_SKIP_PACKETS`: more packet ids (hex, comma-separated) for the `packets` part to
/// leave out, to find which packet differs between two runs (the detail rows name it).
fn skipped_packets() -> Vec<u32> {
    std::env::var("HORIZON_TICK_DIGEST_SKIP_PACKETS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|id| u32::from_str_radix(id.trim().trim_start_matches("0x"), 16).ok())
        .collect()
}

#[path = "tick_digest/parts.rs"]
mod parts;

/// A count and a hash of what was added: `DefaultHasher::new` is SipHash with fixed keys, the same in
/// every run of one toolchain.
struct Part {
    n: u64,
    h: DefaultHasher,
}
impl Part {
    fn new() -> Self {
        Self {
            n: 0,
            h: DefaultHasher::new(),
        }
    }
    fn add(&mut self, value: impl Hash) {
        self.n += 1;
        value.hash(&mut self.h);
    }
    fn show(&self) -> String {
        format!("{}:{:016x}", self.n, self.h.finish())
    }
}

pub(super) struct TickDigest {
    out: BufWriter<std::fs::File>,
    detail: Option<(u64, u64, BufWriter<std::fs::File>)>,
    full_every: u64,
    tick: u64,
    /// The line of the state before the first tick (`0 ...`) is written.
    pub(super) started: bool,
}

impl TickDigest {
    pub(super) fn from_env() -> horizon_client::Result<Option<Self>> {
        let Some(path) = std::env::var_os("HORIZON_TICK_DIGEST") else {
            return Ok(None);
        };
        let out = BufWriter::new(std::fs::File::create(&path)?);
        let detail = match std::env::var("HORIZON_TICK_DIGEST_DETAIL") {
            Ok(range) => {
                let (from, to) = range
                    .split_once('-')
                    .ok_or("HORIZON_TICK_DIGEST_DETAIL: FROM-TO")?;
                let mut name = path.clone();
                name.push(".detail");
                Some((
                    from.trim().parse()?,
                    to.trim().parse()?,
                    BufWriter::new(std::fs::File::create(name)?),
                ))
            }
            Err(_) => None,
        };
        let full_every = match std::env::var("HORIZON_TICK_DIGEST_FULL") {
            Ok(every) => every.trim().parse()?,
            Err(_) => 20,
        };
        let range = detail.as_ref().map(|(from, to, _)| (*from, *to));
        parts::start(full_every, range);
        Ok(Some(Self {
            out,
            detail,
            full_every,
            tick: 0,
            started: false,
        }))
    }
}

/// Whether this run is a fixture run; the chunk results are then applied in a fixed order from the
/// start, the spawn search included.
pub(super) fn fixture_requested() -> bool {
    let on = std::env::var_os("HORIZON_TICK_FIXTURE_TICKS").is_some();
    if on {
        horizon_client::streaming::set_deterministic(true);
    }
    on
}

pub(super) struct Fixture {
    pub(super) ticks: u64,
    pub(super) ran: u64,
    commands: Vec<(u64, String)>,
    /// `HORIZON_TICK_FIXTURE_PLAYERS`: simulated players, one joining every
    /// `HORIZON_TICK_FIXTURE_JOIN_EVERY` ticks (default 10) from the first tick on.
    player_count: usize,
    join_every: u64,
    players: Vec<Phantom>,
    /// The entity types the phantoms out of creative mode take for monsters (`scene::HOSTILE`),
    /// looked up once.
    hostile: Vec<i32>,
}

/// One simulated player of a fixture, as `horizon-bots --behaviour mixed` plays: it enters through the
/// same events a network connection posts (`NetEvent::Joined`, then `NetEvent::Packet`), before the
/// tick that takes them, and answers what the server waits for from the server's own state (the
/// teleport to accept, the chunk batches to acknowledge). Three of five wander within 48 blocks of
/// their spawn, one flies away at 22 blocks a second, one stands; all 24 blocks above their spawn,
/// moving only while the server has sent the chunk they are in. What it is sent is read once a tick
/// (`parts::receive`): in creative mode it makes nothing of it, out of it its play does (`observe`).
struct Phantom {
    conn: u32,
    rx: std::sync::mpsc::Receiver<horizon_client::lan::Out>,
    /// Its sender's budget of bytes not read yet (`MAX_PENDING`), which `parts::receive` gives back:
    /// a test channel has no writer to do it, and past the budget nothing more is sent.
    pending: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// splitmix64 state, from the player's index.
    random: u64,
    /// 0 wander, 1 explore, 2 stand.
    behaviour: u8,
    loaded_sent: bool,
    accepted: Option<i32>,
    position: Option<[f64; 3]>,
    spawn: [f64; 3],
    target: [f64; 3],
    cruise: f64,
    heading: f64,
    /// Its play out of creative mode: the survival scene (`scene.rs`).
    survivor: scene::Survivor,
}

impl Phantom {
    fn unit(&mut self) -> f64 {
        self.random = self.random.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.random;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// One tick of movement (`horizon-bots`: climb to cruising height, then the behaviour).
    fn step(&mut self) {
        let Some(mut pos) = self.position else {
            return;
        };
        if pos[1] < self.cruise {
            pos[1] = (pos[1] + 0.5).min(self.cruise);
        } else if self.behaviour == 0 {
            let (dx, dz) = (self.target[0] - pos[0], self.target[2] - pos[2]);
            let distance = dx.hypot(dz);
            if distance < 1.0 {
                let angle = self.unit() * std::f64::consts::TAU;
                let reach = 8.0 + self.unit() * 40.0;
                self.target = [
                    self.spawn[0] + angle.cos() * reach,
                    self.cruise,
                    self.spawn[2] + angle.sin() * reach,
                ];
            } else {
                let step = 0.28_f64.min(distance);
                pos[0] += dx / distance * step;
                pos[2] += dz / distance * step;
            }
        } else if self.behaviour == 1 {
            pos[0] += self.heading.cos() * 1.1;
            pos[2] += self.heading.sin() * 1.1;
        }
        self.position = Some(pos);
    }

    /// What a client learns from one body it was sent (`Out::Packet`: VarInt id and payload),
    /// kept for its play out of creative mode (`scene::Survivor::observe`); the body is left as it
    /// is. Called on every body taken from `rx`, in order, before the phantom's next turn.
    fn observe(&mut self, body: &[u8]) {
        self.survivor.observe(body);
    }
}

impl Fixture {
    pub(super) fn from_env() -> horizon_client::Result<Option<Self>> {
        let Ok(ticks) = std::env::var("HORIZON_TICK_FIXTURE_TICKS") else {
            return Ok(None);
        };
        let mut commands = Vec::new();
        if let Some(path) = std::env::var_os("HORIZON_TICK_FIXTURE_COMMANDS") {
            for line in std::fs::read_to_string(path)?.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let (at, command) = line
                    .split_once(' ')
                    .ok_or("HORIZON_TICK_FIXTURE_COMMANDS: lines TICK COMMAND")?;
                commands.push((
                    at.parse()?,
                    command.trim().trim_start_matches('/').to_owned(),
                ));
            }
        }
        let number = |name: &str, default: u64| -> horizon_client::Result<u64> {
            Ok(match std::env::var(name) {
                Ok(value) => value.trim().parse()?,
                Err(_) => default,
            })
        };
        horizon_client::streaming::set_deterministic(true);
        Ok(Some(Self {
            ticks: ticks.trim().parse()?,
            ran: 0,
            commands,
            player_count: number("HORIZON_TICK_FIXTURE_PLAYERS", 0)? as usize,
            join_every: number("HORIZON_TICK_FIXTURE_JOIN_EVERY", 10)?.max(1),
            players: Vec::new(),
            hostile: Vec::new(),
        }))
    }

    /// The commands to run before the next tick.
    pub(super) fn due(&self) -> Vec<String> {
        let next = self.ran + 1;
        self.commands
            .iter()
            .filter(|(at, _)| *at == next)
            .map(|(_, command)| command.clone())
            .collect()
    }
}

impl Game<'_> {
    /// Before a fixture tick: the simulated players due join, and every player's packets of this tick
    /// are posted for the tick's `Hub::poll` (see `Phantom`).
    pub(super) fn fixture_players(&mut self, fixture: &mut Fixture) {
        use horizon_protocol::packets::{
            Empty,
            play::{self, Serverbound as S},
        };
        let Some(host) = self.lan.as_mut() else {
            return;
        };
        let shared = host.hub.lan.shared.clone();
        let next = fixture.ran + 1;
        while fixture.players.len() < fixture.player_count
            && next > fixture.players.len() as u64 * fixture.join_every
        {
            let index = fixture.players.len();
            let name = format!("fixture{index}");
            let Some(conn) = shared.next_id() else {
                break;
            };
            let (tx, rx) = std::sync::mpsc::channel();
            let pending = std::sync::Arc::<std::sync::atomic::AtomicUsize>::default();
            let joined = horizon_client::lan::Joined {
                conn,
                uuid: horizon_protocol::crypto::offline_uuid(&name),
                name,
                properties: Vec::new(),
                info: horizon_protocol::packets::configuration::ClientInformation::new(10),
                brand: "fixture".into(),
                sender: horizon_client::lan::PlayerSender::new(tx, pending.clone()),
                latency: std::sync::Arc::new(std::sync::atomic::AtomicI32::new(0)),
                address: ([127, 0, 0, 1], 0).into(),
                expected: false,
            };
            if shared
                .events
                .send(horizon_client::lan::NetEvent::Joined(Box::new(joined)))
                .is_err()
            {
                super::log("WARN", "tick fixture: a simulated player could not join");
                break;
            }
            let mut phantom = Phantom {
                conn,
                rx,
                pending,
                random: 0x5EED_0000 ^ index as u64,
                behaviour: match index % 5 {
                    0 | 2 | 4 => 0,
                    1 => 1,
                    _ => 2,
                },
                loaded_sent: false,
                accepted: None,
                position: None,
                spawn: [0.0; 3],
                target: [0.0; 3],
                cruise: 0.0,
                heading: 0.0,
                survivor: scene::Survivor::new(index),
            };
            phantom.heading = phantom.unit() * std::f64::consts::TAU;
            fixture.players.push(phantom);
        }
        // The monsters' entity types, once, for the phantoms out of creative mode (`scene.rs`).
        if fixture.hostile.is_empty() {
            let sync = shared.sync();
            fixture.hostile = scene::HOSTILE
                .iter()
                .filter_map(|kind| sync.id("minecraft:entity_type", kind))
                .collect();
        }
        // What a phantom sends this tick, for the tick's `Hub::poll` to take.
        let post = |conn: u32, packets: Vec<S>| {
            for packet in packets {
                let _ = shared.events.send(horizon_client::lan::NetEvent::Packet(
                    conn,
                    Box::new(packet),
                    16,
                ));
            }
        };
        for phantom in &mut fixture.players {
            let Some(guest) = host.hub.guests.iter().find(|g| g.conn == phantom.conn) else {
                continue;
            };
            // Out of creative mode a phantom plays the survival scene, from what it reads of what
            // it is sent (`Phantom::observe`); in creative mode it reads nothing.
            phantom.survivor.watching = !guest.creative();
            parts::receive(phantom);
            if !guest.creative() {
                let mut packets = Vec::new();
                phantom.survivor.play(guest, &fixture.hostile, &mut packets);
                post(phantom.conn, packets);
                continue;
            }
            let mut packets = Vec::new();
            if !phantom.loaded_sent {
                packets.push(S::PlayerLoaded(Empty));
                phantom.loaded_sent = true;
            }
            if let Some((id, at)) = guest.awaiting_teleport
                && phantom.accepted != Some(id)
            {
                packets.push(S::AcceptTeleportation(play::AcceptTeleportation {
                    id,
                    x: at[0],
                    y: at[1],
                    z: at[2],
                    yaw: 0.0,
                    pitch: 0.0,
                }));
                phantom.accepted = Some(id);
                if phantom.position.is_none() {
                    phantom.spawn = at;
                    phantom.target = at;
                    phantom.cruise = at[1] + 24.0;
                }
                phantom.position = Some(at);
            }
            for _ in 0..guest.unacknowledged {
                packets.push(S::ChunkBatchReceived(play::ChunkBatchReceived {
                    chunks_per_tick: 64.0,
                }));
            }
            if let Some(pos) = phantom.position {
                let here = ((pos[0].floor() as i32) >> 4, (pos[2].floor() as i32) >> 4);
                if guest.sent_chunks.contains(&here) {
                    phantom.step();
                }
                let pos = phantom.position.unwrap_or(pos);
                packets.push(S::MovePlayerPos(play::MovePlayerPos {
                    x: pos[0],
                    y: pos[1],
                    z: pos[2],
                    flags: 0,
                }));
                packets.push(S::ClientTickEnd(Empty));
            }
            post(phantom.conn, packets);
        }
    }

    /// Before a fixture tick: the chunk jobs in flight finish and their results are applied in a fixed
    /// order, one `update_world` per round, until no job is left. Jobs the tick itself queues run
    /// during it and are applied before the next tick.
    pub(super) fn fixture_settle(&mut self) {
        let mut empty_rounds = 0;
        loop {
            self.streaming.wait_quiescent();
            let waiting = self.streaming.results_waiting();
            if !waiting && !self.streaming.in_flight() {
                return;
            }
            if waiting {
                empty_rounds = 0;
            } else {
                // Work counted in flight with no job running and no result: one more update hands
                // it to the workers; past a few, something is stuck.
                empty_rounds += 1;
                if empty_rounds > 3 {
                    super::log(
                        "WARN",
                        "tick fixture: chunk work counted in flight never finished",
                    );
                    return;
                }
            }
            self.update_world();
        }
    }

    /// Writes the digest line of the tick that just ended (see the module docs), or with `start` the
    /// line `0` of the state before the first tick.
    pub(super) fn write_tick_digest(
        &mut self,
        digest: &mut TickDigest,
        start: bool,
        fixture: Option<&mut Fixture>,
    ) -> horizon_client::Result<()> {
        if start {
            digest.started = true;
        } else {
            digest.tick += 1;
        }
        let tick = if start { 0 } else { digest.tick };
        let detailed = digest
            .detail
            .as_ref()
            .is_some_and(|(from, to, _)| (*from..=*to).contains(&tick));
        let full = parts::full_tick(digest.full_every, tick);
        let mut rows: Vec<String> = Vec::new();
        let mut line = format!(
            "{tick} time={} dim={:?} random={:016x} rtv={}",
            self.world.game_time,
            self.dim,
            self.world.random.state(),
            self.block_ticks.random_tick_value,
        );
        // The level in place (the Overworld while it ticks) under plain names, then the server's
        // own parts (`tick_digest/parts.rs`).
        line += &self.level_parts("", "", full, detailed, &mut rows);
        line += &self.server_parts(fixture, detailed, &mut rows);
        // Then each other level that ticked, under its name (`tick_digest_level`).
        line += &parts::other_levels(tick, &mut rows);

        writeln!(digest.out, "{line}")?;
        digest.out.flush()?;
        if detailed && let Some((_, _, file)) = &mut digest.detail {
            writeln!(file, "# tick {tick}: {line}")?;
            for row in &rows {
                writeln!(file, "{row}")?;
            }
            file.flush()?;
        }
        Ok(())
    }
}
