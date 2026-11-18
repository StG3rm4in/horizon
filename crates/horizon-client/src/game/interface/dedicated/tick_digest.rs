//! The exactness harness of the server tick (`tools/tick_digest.py`): a digest of every tick, and a
//! fixture mode that takes worker timing out of the tick.
//!
//! The digest (`HORIZON_TICK_DIGEST=FILE`): one line per server tick with hashes of the level in place
//! once the tick ends (the Overworld while it is the only level ticking): its random, the block writes
//! of the tick, its scheduled ticks, its entities, the players, the loaded chunks and, every
//! `HORIZON_TICK_DIGEST_FULL` ticks (default 20, 0 for never), the blocks and light of every loaded
//! section. Two runs of one fixture must write the same lines: the first line that differs names the
//! tick and the part. `HORIZON_TICK_DIGEST_DETAIL=FROM-TO` also writes the rows behind the hashes of
//! those ticks to `FILE.detail`, to find what differs.
//!
//! The fixture (`HORIZON_TICK_FIXTURE_TICKS=N`): the server runs N ticks back to back, then stops.
//! Before each tick it runs the console commands of `HORIZON_TICK_FIXTURE_COMMANDS` (lines
//! `TICK COMMAND`, ticks counted from 1) due then, waits for every chunk job in flight and applies
//! the results in a fixed order (`streaming::set_deterministic`), as many rounds as the work needs.
//! The tick itself is unchanged.

use super::super::Game;
use super::super::entity::{Entity as _, entity_lists_of, for_each_entity};
use std::{
    hash::{DefaultHasher, Hash, Hasher},
    io::{BufWriter, Write},
};

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
}

/// One simulated player of a fixture, as `horizon-bots --behaviour mixed` plays: it enters through the
/// same events a network connection posts (`NetEvent::Joined`, then `NetEvent::Packet`), before the
/// tick that takes them, and answers what the server waits for from the server's own state (the
/// teleport to accept, the chunk batches to acknowledge). Three of five wander within 48 blocks of
/// their spawn, one flies away at 22 blocks a second, one stands; all 24 blocks above their spawn,
/// moving only while the server has sent the chunk they are in. What it is sent is drained unread.
struct Phantom {
    conn: u32,
    rx: std::sync::mpsc::Receiver<horizon_client::lan::Out>,
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
            && next >= 1 + fixture.players.len() as u64 * fixture.join_every
        {
            let index = fixture.players.len();
            let name = format!("fixture{index}");
            let Some(conn) = shared.next_id() else {
                break;
            };
            let (tx, rx) = std::sync::mpsc::channel();
            let joined = horizon_client::lan::Joined {
                conn,
                uuid: horizon_protocol::crypto::offline_uuid(&name),
                name,
                properties: Vec::new(),
                info: horizon_protocol::packets::configuration::ClientInformation::new(10),
                brand: "fixture".into(),
                sender: horizon_client::lan::PlayerSender::new(tx, std::sync::Arc::default()),
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
            };
            phantom.heading = phantom.unit() * std::f64::consts::TAU;
            fixture.players.push(phantom);
        }
        for phantom in &mut fixture.players {
            let Some(guest) = host.hub.guests.iter().find(|g| g.conn == phantom.conn) else {
                continue;
            };
            while phantom.rx.try_recv().is_ok() {}
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
            for packet in packets {
                let _ = shared.events.send(horizon_client::lan::NetEvent::Packet(
                    phantom.conn,
                    Box::new(packet),
                    16,
                ));
            }
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
        let mut rows: Vec<String> = Vec::new();
        let mut line = format!(
            "{tick} time={} dim={:?} random={:016x} rtv={}",
            self.world.game_time,
            self.dim,
            self.world.random.state(),
            self.block_ticks.random_tick_value,
        );

        let writes = self
            .world
            .digest_writes
            .replace(Vec::new())
            .unwrap_or_default();
        let mut part = Part::new();
        for (pos, state) in &writes {
            part.add((pos, state));
            if detailed {
                rows.push(format!("w {} {} {} {state}", pos.0, pos.1, pos.2));
            }
        }
        line += &format!(" writes={}", part.show());

        let mut part = Part::new();
        for (fluid, ticks) in [
            (false, &self.block_ticks.blocks),
            (true, &self.block_ticks.fluids),
        ] {
            for row in ticks.digest_rows() {
                part.add((fluid, row));
                if detailed {
                    rows.push(format!("s {fluid} {row:?}"));
                }
            }
        }
        line += &format!(" scheduled={}", part.show());

        let mut part = Part::new();
        for_each_entity!(entity_lists_of!(self), entity => {
            let (p, v, r) = (entity.position(), entity.velocity(), entity.rotation());
            part.add((
                entity.id(),
                entity.native_type(),
                [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()],
                [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()],
                [r[0].to_bits(), r[1].to_bits()],
            ));
            if detailed {
                rows.push(format!(
                    "e {} {} {:?} {:?} {:?}",
                    entity.id(),
                    entity.native_type(),
                    [p.x, p.y, p.z],
                    [v.x, v.y, v.z],
                    r,
                ));
            }
        });
        line += &format!(" entities={}", part.show());

        let mut part = Part::new();
        if let Some(host) = &self.lan {
            for guest in &host.hub.guests {
                let p = guest.position;
                part.add((
                    guest.uuid,
                    guest.entity_id,
                    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()],
                ));
                if detailed {
                    rows.push(format!("p {} {:032x} {p:?}", guest.entity_id, guest.uuid));
                }
            }
        }
        line += &format!(" players={}", part.show());

        // The chunks that random tick (their count is the random tick draws of a tick) and the ones
        // whose entities are not loaded yet (they don't block tick).
        let mut ticking = Vec::new();
        self.random_tick_chunks(&mut ticking);
        let mut part = Part::new();
        for key in &ticking {
            part.add(key);
            if detailed {
                rows.push(format!("c {} {}", key.0, key.1));
            }
        }
        line += &format!(" ticking={}", part.show());
        let mut pending: Vec<(i32, i32)> = self.world.entities_pending.iter().copied().collect();
        pending.sort_unstable();
        let mut part = Part::new();
        for key in &pending {
            part.add(key);
        }
        line += &format!(" pending={}", part.show());

        // The chunk results applied since the last line, in order (fixture mode).
        let mut part = Part::new();
        for row in horizon_client::streaming::take_applied() {
            part.add(row);
            if detailed {
                rows.push(format!("a {row:?}"));
            }
        }
        line += &format!(" applied={}", part.show());

        // The order the chunk map walks its chunks in (passes such as postprocess_chunks walk it):
        // it follows the history of the insertions, not only which chunks there are.
        let mut part = Part::new();
        for key in self.world.chunks.keys() {
            part.add(key);
            if detailed {
                rows.push(format!("m {} {}", key.0, key.1));
            }
        }
        line += &format!(" order={}", part.show());
        let mut keys: Vec<(i32, i32)> = self.world.chunks.keys().copied().collect();
        keys.sort_unstable();
        let mut part = Part::new();
        for key in &keys {
            part.add(key);
        }
        line += &format!(" chunks={}", part.show());

        if digest.full_every > 0 && tick % digest.full_every == 0 {
            let mut part = Part::new();
            let mut blocks = Box::new([0u32; 4096]);
            let mut light = vec![0u8; 4096];
            for key in &keys {
                let chunk = &self.world.chunks[key];
                for (s, section) in chunk.sections.iter().enumerate() {
                    section.unpack(&mut blocks);
                    for (i, l) in light.iter_mut().enumerate() {
                        *l = section.light.get(i);
                    }
                    part.add((key, s, &blocks[..], &light[..]));
                }
                part.add(chunk.block_entities.len());
            }
            line += &format!(" full={}", part.show());
        }

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
