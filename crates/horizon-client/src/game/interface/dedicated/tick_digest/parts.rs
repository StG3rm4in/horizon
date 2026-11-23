//! The parts of a digest line (`tick_digest.rs`; docs/tick-digest.md lists them): those of a level,
//! read while it is in place, and the server's own. They only read the game: no draw, no id or UUID
//! handed out, no cache filled.

use super::super::super::Game;
use super::super::super::entity::{Entity as _, entity_lists_of, for_each_entity};
use super::super::super::{FallingBlock, boats, fireworks, general, minecarts, tnt, trident};
use super::{Fixture, Part, Phantom, canonical, packet_id, skipped_packets};
use horizon_client::join_seq::JoinSeq;
use horizon_client::join_seq::Joined as _;
use horizon_client::lan::Out;
use horizon_core::nbt::{JavaString, Tag};
use horizon_core::random::LegacyRandom;
use horizon_gameplay::dim::Dim;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Mutex, OnceLock, PoisonError};

/// Whether line `tick` is a full one: every `every` lines, never for 0.
pub(super) fn full_tick(every: u64, tick: u64) -> bool {
    every > 0 && tick.is_multiple_of(every)
}

/// The hash of one value, for a detail row.
fn hash_of(value: impl Hash) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

/// Where the `fields` part reads an entity's fields (`DigestFields`): into one hash, and on a
/// detailed line into its row, `name=value`.
struct Fields {
    h: DefaultHasher,
    row: Option<String>,
}

impl Fields {
    fn new(detailed: bool) -> Self {
        Self {
            h: DefaultHasher::new(),
            row: detailed.then(String::new),
        }
    }
    fn int(&mut self, name: &str, value: impl Into<i64>) {
        let value = value.into();
        value.hash(&mut self.h);
        if let Some(row) = &mut self.row {
            *row += &format!(" {name}={value}");
        }
    }
    /// A float by its bits: `-0.0` is not `0.0`.
    fn float(&mut self, name: &str, value: impl Into<f64>) {
        let value = value.into();
        value.to_bits().hash(&mut self.h);
        if let Some(row) = &mut self.row {
            *row += &format!(" {name}={value:?}");
        }
    }
    fn flag(&mut self, name: &str, value: bool) {
        self.int(name, value);
    }
    /// An entity by its id (`Who`), never by its UUID, which is random.
    fn who(&mut self, name: &str, value: Option<horizon_client::mobs::Who>) {
        value.hash(&mut self.h);
        if let Some(row) = &mut self.row {
            *row += &format!(" {name}={value:?}");
        }
    }
    fn effect(&mut self, id: &str, amplifier: i32, duration: i32) {
        (id, amplifier, duration).hash(&mut self.h);
        if let Some(row) = &mut self.row {
            *row += &format!(" effect={id}:{amplifier}:{duration}");
        }
    }
}

/// What the `fields` and `randoms` parts read of an entity beyond the `Entity` trait, from its own
/// fields as they are (upstream's `DigestParts` in `state_digest.rs` is the model): never their
/// `Debug` text, which a field an optimization adds would change.
trait DigestFields {
    fn digest_fields(&self, _fields: &mut Fields) {}
    /// Its own random (`Entity.random`), for a kind that has one.
    fn digest_random(&self) -> Option<&LegacyRandom> {
        None
    }
}

impl DigestFields for horizon_client::mobs::Mob {
    fn digest_fields(&self, f: &mut Fields) {
        f.float("health", self.health);
        f.float("absorption", self.absorption);
        f.int("hurt", self.hurt_time);
        f.int("cooldown", self.invulnerable);
        f.int("invulnerable", self.invulnerable_time);
        f.int("death", self.death_time);
        f.int("fire", self.fire_ticks);
        f.int("frozen", self.ticks_frozen);
        f.int("air", self.air);
        f.float("fall", self.fall_distance);
        f.flag("ground", self.on_ground);
        f.flag("removed", self.removed);
        f.flag("persistent", self.persistent);
        f.float("head", self.head_yaw);
        f.float("body", self.body_yaw);
        f.int("age", self.age);
        f.int("idle", self.ai.no_action_time);
        f.int("swing", self.ai.swing_time);
        f.flag("swinging", self.ai.swinging);
        f.int("swing_length", self.ai.swing_duration);
        f.who("target", self.ai.target);
        f.who("hurt_by", self.ai.last_hurt_by);
        f.int("hurt_by_time", self.ai.last_hurt_by_time);
        // An attacker known by its UUID alone (not loaded yet): only whether there is one.
        f.flag("hurt_by_uuid", self.ai.last_hurt_by_ref.is_some());
        for effect in &self.extra.effects {
            f.effect(effect.id, effect.amplifier, effect.duration);
        }
    }
    fn digest_random(&self) -> Option<&LegacyRandom> {
        Some(&self.random)
    }
}

impl DigestFields for horizon_client::entities::ItemEntity {
    fn digest_fields(&self, f: &mut Fields) {
        f.int("health", self.health);
        f.int("fire", self.fire_ticks);
        f.int("count", self.stack.count);
        f.int("age", self.age);
        f.int("pickup", self.pickup_delay);
        f.float("fall", self.fall_distance);
        f.flag("ground", self.on_ground);
    }
    fn digest_random(&self) -> Option<&LegacyRandom> {
        Some(&self.random)
    }
}

impl DigestFields for horizon_client::entities::ExperienceOrb {
    fn digest_fields(&self, f: &mut Fields) {
        f.int("value", self.value);
        f.int("count", self.count);
        f.int("age", self.age);
        f.int("health", self.health);
        f.int("fire", self.fire_ticks);
        f.float("fall", self.fall_distance);
        f.flag("ground", self.on_ground);
        f.int("following", self.following.map_or(-1, i64::from));
    }
}

impl DigestFields for horizon_client::entities::ArmorStand {
    fn digest_fields(&self, f: &mut Fields) {
        f.float("health", self.health);
        f.int("fire", self.fire_ticks);
        f.int("hit", self.last_hit);
        f.flag("ground", self.on_ground);
    }
}

impl DigestFields for boats::Boat {
    fn digest_fields(&self, f: &mut Fields) {
        f.float("damage", self.damage);
        f.int("hurt", self.hurt_time);
        f.int("fire", self.fire_ticks);
        f.int("frozen", self.ticks_frozen);
        f.float("fall", self.fall_distance);
        f.flag("ground", self.on_ground);
    }
    fn digest_random(&self) -> Option<&LegacyRandom> {
        Some(&self.random)
    }
}

impl DigestFields for minecarts::Minecart {
    fn digest_fields(&self, f: &mut Fields) {
        f.float("damage", self.damage);
        f.int("hurt", self.hurt_time);
        f.int("frozen", self.ticks_frozen);
        f.int("fuse", self.fuse);
        f.int("fuel", self.fuel);
        f.float("fall", self.fall_distance);
        f.flag("ground", self.on_ground);
    }
    fn digest_random(&self) -> Option<&LegacyRandom> {
        Some(&self.random)
    }
}

impl DigestFields for tnt::PrimedTnt {
    fn digest_fields(&self, f: &mut Fields) {
        f.int("fuse", self.fuse);
        f.float("fall", self.fall_distance);
        f.flag("ground", self.on_ground);
    }
}

impl DigestFields for FallingBlock {
    fn digest_fields(&self, f: &mut Fields) {
        f.int("time", self.age);
        f.float("fall", self.fall_distance);
        f.flag("ground", self.on_ground);
    }
}

impl DigestFields for horizon_client::projectiles::Projectile {
    fn digest_fields(&self, f: &mut Fields) {
        f.int("age", self.age);
        f.int("fire", self.fire_ticks);
        f.float("fall", self.fall_distance);
        f.flag("stuck", self.stuck);
        f.flag("ground", self.on_ground);
        f.who("owner", self.owner);
    }
    fn digest_random(&self) -> Option<&LegacyRandom> {
        Some(&self.random.0)
    }
}

impl DigestFields for fireworks::Rocket {
    fn digest_fields(&self, f: &mut Fields) {
        f.int("life", self.life);
        f.int("lifetime", self.lifetime);
        f.float("fall", self.fall_distance);
        f.flag("ground", self.on_ground);
    }
    fn digest_random(&self) -> Option<&LegacyRandom> {
        Some(&self.random)
    }
}

impl DigestFields for trident::ThrownTrident {
    fn digest_fields(&self, f: &mut Fields) {
        f.int("life", self.life);
        f.int("in_ground", self.in_ground_time);
        f.int("fire", self.fire_ticks);
        f.int("returning", self.returning);
        f.flag("dealt", self.dealt_damage);
        f.float("fall", self.fall_distance);
        f.flag("ground", self.on_ground);
    }
    fn digest_random(&self) -> Option<&LegacyRandom> {
        Some(&self.random)
    }
}

impl DigestFields for general::GeneralRow {
    fn digest_fields(&self, f: &mut Fields) {
        f.float("fall", self.fall_distance);
    }
}

/// Keys whose values are entity UUIDs, random (`save::random_uuid`, seeded from the clock): an
/// entity's own (`UUID`: a bee in its hive, a player's ender pearls, the vehicle it rides and its
/// `Attach`), a trial spawner's `current_mobs`, a creaking heart's `creaking`, a conduit's
/// `Target`, a sculk listener's `source` and `projectile_owner`, a bee's `AngryAt`. Zeroed before
/// hashing, as `canonical` does in `ADD_ENTITY`: how many there are still counts.
const UUID_KEYS: [&str; 8] = [
    "UUID",
    "Attach",
    "current_mobs",
    "creaking",
    "Target",
    "source",
    "projectile_owner",
    "AngryAt",
];

/// Whether `key` is one of `UUID_KEYS`.
fn names_uuid(key: &JavaString) -> bool {
    let units = || key.0.iter().copied();
    UUID_KEYS.iter().any(|k| units().eq(k.encode_utf16()))
}

/// `tag` with every value `UUID_KEYS` names zeroed, at any depth.
fn mask_uuids(tag: &mut Tag) {
    match tag {
        Tag::Compound(map) => {
            for (key, value) in map {
                if names_uuid(key) {
                    zero_ints(value);
                } else {
                    mask_uuids(value);
                }
            }
        }
        Tag::List(list) => list.iter_mut().for_each(mask_uuids),
        _ => {}
    }
}

/// Every int array in `tag` (a UUID is four ints) zeroed, its length kept.
fn zero_ints(tag: &mut Tag) {
    match tag {
        Tag::IntArray(ints) => ints.fill(0),
        Tag::List(list) => list.iter_mut().for_each(zero_ints),
        Tag::Compound(map) => map.values_mut().for_each(zero_ints),
        _ => {}
    }
}

/// A detail row per key of a compound, `{head} {key} {value}`: a number or a string as it reads,
/// anything else by its hash, so that the first row that differs names the key.
fn tag_rows(head: &str, tag: &Tag, rows: &mut Vec<String>) {
    if let Tag::Compound(map) = tag {
        for (key, value) in map {
            let value = match value {
                Tag::Byte(v) => v.to_string(),
                Tag::Short(v) => v.to_string(),
                Tag::Int(v) => v.to_string(),
                Tag::Long(v) => v.to_string(),
                Tag::Float(bits) => format!("{:?}", f32::from_bits(*bits)),
                Tag::Double(bits) => format!("{:?}", f64::from_bits(*bits)),
                Tag::String(text) => format!("{:?}", text.to_string_lossy()),
                other => format!("#{:016x}", hash_of(other)),
            };
            rows.push(format!("{head} {} {value}", key.to_string_lossy()));
        }
    }
}

/// What the fixture's players were sent and no line has hashed yet, by connection, in their order
/// of sending: kept by `receive` while a digest runs (`start`), so that a line also hashes what was
/// sent before their packets of its tick were posted (by the fixture's commands, the console).
static RECEIVED: Mutex<Option<HashMap<u32, Vec<Out>>>> = Mutex::new(None);

/// What the levels put back out of place before a line is written left for it
/// (`Game::tick_digest_level`). Set when the digest starts.
static LEVELS: OnceLock<Mutex<Levels>> = OnceLock::new();

struct Levels {
    /// The line the server tick running now writes.
    next: u64,
    full_every: u64,
    detail: Option<(u64, u64)>,
    /// Their parts, each under its level's name, in the order the levels ticked, and their rows.
    parts: String,
    rows: Vec<String>,
}

/// The digest starts (`TickDigest::from_env`): what the fixture's players are sent and the other
/// levels' parts are kept for its lines from now on.
pub(super) fn start(full_every: u64, detail: Option<(u64, u64)>) {
    *RECEIVED.lock().unwrap_or_else(PoisonError::into_inner) = Some(HashMap::new());
    let levels = Levels {
        next: 1,
        full_every,
        detail,
        parts: String::new(),
        rows: Vec::new(),
    };
    let _ = LEVELS.set(Mutex::new(levels));
}

/// The parts the other levels left for line `tick`, and their rows; the next line's are kept from
/// now on.
pub(super) fn other_levels(tick: u64, rows: &mut Vec<String>) -> String {
    let Some(levels) = LEVELS.get() else {
        return String::new();
    };
    let mut levels = levels.lock().unwrap_or_else(PoisonError::into_inner);
    levels.next = tick + 1;
    rows.append(&mut levels.rows);
    std::mem::take(&mut levels.parts)
}

/// Reads what `phantom` was sent since it was last read, and keeps it for the next line while a
/// digest runs (else it is dropped: a fixture player reads none of it). The one place a fixture
/// player's packets are read, before its packets of a tick are posted (`fixture_players`) and as a
/// line is written (`packets`): what a phantom makes of them goes here, for each of `bodies(&out)`,
/// so that it has seen the same bodies in the same order each time it posts, with a digest or
/// without.
pub(super) fn receive(phantom: &mut Phantom) {
    let mut kept = RECEIVED.lock().unwrap_or_else(PoisonError::into_inner);
    while let Ok(out) = phantom.rx.try_recv() {
        for body in bodies(&out) {
            phantom.observe(body);
        }
        if let Some(kept) = kept.as_mut() {
            kept.entry(phantom.conn).or_default().push(out);
        }
    }
    // Everything sent is read: the budget a network writer would have given back by now.
    phantom.pending.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// What `receive` kept of connection `conn`'s since the last line.
fn take_received(conn: u32) -> Vec<Out> {
    let mut kept = RECEIVED.lock().unwrap_or_else(PoisonError::into_inner);
    match kept.as_mut() {
        Some(kept) => kept.remove(&conn).unwrap_or_default(),
        None => Vec::new(),
    }
}

/// The bodies of one thing a fixture player was sent, in order.
fn bodies(out: &Out) -> &[Vec<u8>] {
    match out {
        Out::Packet(body) => std::slice::from_ref(body),
        Out::Packets(bodies) => bodies,
        Out::Disconnect(_) => &[],
    }
}

/// The `packets` part: what each fixture player was sent since the last line, in its order of
/// sending, the players in the order they joined. The other parts see no packet, neither its body
/// nor its order. A player's arm swings come last and sorted: `lan_swings` sends them in the order
/// of a `HashMap` (`LanHost::play`), which differs between two runs of one build.
fn packets(fixture: &mut Fixture, detailed: bool, rows: &mut Vec<String>) -> String {
    let skip = skipped_packets();
    let mut part = Part::new();
    let swing = horizon_protocol::ids::play::clientbound::SWING_ANIMATION as u32;
    for phantom in &mut fixture.players {
        receive(phantom);
        let mut swings: Vec<Vec<u8>> = Vec::new();
        for out in take_received(phantom.conn) {
            if let Out::Disconnect(_) = out {
                part.add((phantom.conn, "disconnect"));
            }
            for body in bodies(&out) {
                let Some(body) = canonical(body) else {
                    continue;
                };
                let id = packet_id(&body);
                if skip.contains(&id) {
                    continue;
                }
                if id == swing {
                    swings.push(body.into_owned());
                    continue;
                }
                part.add((phantom.conn, &*body));
                if detailed {
                    rows.push(format!(
                        "k {} {id:#04x} {} {:016x}",
                        phantom.conn,
                        body.len(),
                        hash_of(&*body)
                    ));
                }
            }
        }
        swings.sort_unstable();
        for body in &swings {
            part.add((phantom.conn, &**body));
            if detailed {
                rows.push(format!(
                    "k {} {swing:#04x} {} {:016x}",
                    phantom.conn,
                    body.len(),
                    hash_of(&**body)
                ));
            }
        }
    }
    part.show()
}

impl Game<'_> {
    /// The parts of the level in place (docs/tick-digest.md). `name` starts the parts' names and
    /// `at` ends the rows' kinds of a level that is not in place when the line is written; both
    /// are empty for the one that is.
    pub(super) fn level_parts(
        &mut self,
        name: &str,
        at: &str,
        full: bool,
        detailed: bool,
        rows: &mut Vec<String>,
    ) -> String {
        let gauss = self.world.random.cached_gaussian();
        let mut line = match gauss {
            Some(g) => format!(" {name}gauss={:016x}", g.to_bits()),
            None => format!(" {name}gauss=-"),
        };
        if detailed {
            rows.push(format!("g{at} {gauss:?}"));
        }
        let writes = self
            .world
            .digest_writes
            .replace(Vec::new())
            .unwrap_or_default();
        let mut part = Part::new();
        for (pos, state) in &writes {
            part.add((pos, state));
            if detailed {
                rows.push(format!("w{at} {} {} {} {state}", pos.0, pos.1, pos.2));
            }
        }
        line += &format!(" {name}writes={}", part.show());

        let mut part = Part::new();
        for (fluid, ticks) in [
            (false, &self.block_ticks.blocks),
            (true, &self.block_ticks.fluids),
        ] {
            for row in ticks.digest_rows() {
                part.add((fluid, row));
                if detailed {
                    rows.push(format!("s{at} {fluid} {row:?}"));
                }
            }
        }
        line += &format!(" {name}scheduled={}", part.show());

        // Each entity of the entity lists, list by list: its motion (`entities`), its fields
        // and its join number (`fields`), its own random (`randoms`).
        let (mut entities, mut fields, mut randoms) = (Part::new(), Part::new(), Part::new());
        // A mob's target is `Who::Player` whoever the player is: the guest it is bound to
        // (`LanHost::mob_players`) says which one.
        let guests = self
            .lan
            .as_ref()
            .map(|host| host.mob_players.clone())
            .unwrap_or_default();
        for_each_entity!(entity_lists_of!(self), entity => {
            let (p, v, r) = (entity.position(), entity.velocity(), entity.rotation());
            entities.add((
                entity.id(),
                entity.native_type(),
                [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()],
                [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()],
                [r[0].to_bits(), r[1].to_bits()],
            ));
            let mut own = Fields::new(detailed);
            entity.digest_fields(&mut own);
            own.int("guest", guests.get(&entity.id()).map_or(-1, |&g| i64::from(g)));
            fields.add((entity.id(), entity.join_seq(), own.h.finish()));
            let random = entity.digest_random();
            let random = random.map(|g| (g.state(), g.cached_gaussian().map(f64::to_bits)));
            if let Some(random) = random {
                randoms.add((entity.id(), random));
            }
            if detailed {
                rows.push(format!(
                    "e{at} {} {} {:?} {:?} {:?}",
                    entity.id(),
                    entity.native_type(),
                    [p.x, p.y, p.z],
                    [v.x, v.y, v.z],
                    r,
                ));
                let (id, join) = (entity.id(), entity.join_seq().get());
                rows.push(format!("h{at} {id} join={join}{}", own.row.unwrap_or_default()));
                if let Some((state, gauss)) = random {
                    rows.push(format!("r{at} {id} {state:012x} {gauss:?}"));
                }
            }
        });
        line += &format!(" {name}entities={}", entities.show());
        line += &format!(" {name}fields={}", fields.show());
        line += &format!(" {name}randoms={}", randoms.show());

        // The block entities that tick, in their order (`Level.blockEntityTickers`), each as its
        // save would write it: a timer or a content that changed shows on the tick it changed.
        let mut part = Part::new();
        let tickers = self.world.block_entities.ticker_slots();
        for &(pos, slot) in tickers.iter() {
            match self.world.block_entities.live_entity((pos, slot)) {
                Some(entity) => {
                    let (block, tag) = self.digest_block_entity_tag(pos, entity);
                    part.add((pos, slot, &block, &tag));
                    if detailed {
                        let head = format!("t{at} {} {} {} {slot} {block}", pos.0, pos.1, pos.2);
                        tag_rows(&head, &tag, rows);
                    }
                }
                // A slot freed during the tick stays listed until the next tick sweeps it.
                None => part.add((pos, slot)),
            }
        }
        // Let go of the list before the next tick changes it (`Arc::make_mut`).
        drop(tickers);
        line += &format!(" {name}tickers={}", part.show());

        // The chunks that random tick (their count is the random tick draws of a tick) and the ones
        // whose entities are not loaded yet (they don't block tick).
        let mut ticking = Vec::new();
        self.random_tick_chunks(&mut ticking);
        let mut part = Part::new();
        for key in &ticking {
            part.add(key);
            if detailed {
                rows.push(format!("c{at} {} {}", key.0, key.1));
            }
        }
        line += &format!(" {name}ticking={}", part.show());
        let mut pending: Vec<(i32, i32)> = self.world.entities_pending.iter().copied().collect();
        pending.sort_unstable();
        let mut part = Part::new();
        for key in &pending {
            part.add(key);
        }
        line += &format!(" {name}pending={}", part.show());

        // The order the chunk map walks its chunks in (passes such as postprocess_chunks walk it):
        // it follows the history of the insertions, not only which chunks there are.
        let mut part = Part::new();
        for key in self.world.chunks.keys() {
            part.add(key);
            if detailed {
                rows.push(format!("m{at} {} {}", key.0, key.1));
            }
        }
        line += &format!(" {name}order={}", part.show());
        let mut keys: Vec<(i32, i32)> = self.world.chunks.keys().copied().collect();
        keys.sort_unstable();
        let mut part = Part::new();
        for key in &keys {
            part.add(key);
        }
        line += &format!(" {name}chunks={}", part.show());

        if full {
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
                    if detailed {
                        let (mut b, mut l) = (DefaultHasher::new(), DefaultHasher::new());
                        blocks[..].hash(&mut b);
                        light[..].hash(&mut l);
                        rows.push(format!(
                            "f{at} {} {} {s} {:016x} {:016x}",
                            key.0,
                            key.1,
                            b.finish(),
                            l.finish()
                        ));
                    }
                }
            }
            line += &format!(" {name}full={}", part.show());
            // Every block entity, by position: a chunk's own list is empty once the chunk is in
            // (`World::install_chunk_block_entities` moves them into `block_entities`).
            let mut part = Part::new();
            let mut positions: Vec<_> = self.world.block_entities.keys().copied().collect();
            positions.sort_unstable();
            for pos in positions {
                if let Some(entity) = self.world.block_entities.get(&pos) {
                    let (block, tag) = self.digest_block_entity_tag(pos, entity);
                    part.add((pos, &block, &tag));
                    if detailed {
                        let head = format!("b{at} {} {} {} {block}", pos.0, pos.1, pos.2);
                        tag_rows(&head, &tag, rows);
                    }
                }
            }
            line += &format!(" {name}blockents={}", part.show());
        }
        line
    }

    /// The block entity at `pos` as the save writes it (`block_entity_encode::encode`), its UUIDs
    /// zeroed (`mask_uuids`), and its block.
    fn digest_block_entity_tag(
        &self,
        pos: (i32, i32, i32),
        entity: &horizon_client::containers::BlockEntity,
    ) -> (String, Tag) {
        let block = self.data_block_name([pos.0, pos.1, pos.2]);
        let mut tag = horizon_client::save::block_entity_encode::encode(pos, entity, &block);
        mask_uuids(&mut tag);
        (block, tag)
    }

    /// Whether a level records its block writes for the digest from the moment it opens
    /// (`open_level`): on a dedicated server with `HORIZON_TICK_DIGEST` set (read once).
    pub(in crate::game) fn tick_digest_writes(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        self.options.dedicated.is_some()
            && *ON.get_or_init(|| std::env::var_os("HORIZON_TICK_DIGEST").is_some())
    }

    /// The end of the tick of a level the server tick puts back out of place before its line is
    /// written (every level but the first, `levels.rs`): its parts under its name
    /// (`the_nether.writes=…`) and its rows (`w@the_nether …`), for that line.
    pub(in crate::game) fn tick_digest_level(&mut self, dim: Dim) {
        let Some(levels) = LEVELS.get() else {
            return;
        };
        let (full, detailed) = {
            let levels = levels.lock().unwrap_or_else(PoisonError::into_inner);
            let tick = levels.next;
            let detailed = matches!(levels.detail, Some((from, to)) if (from..=to).contains(&tick));
            (full_tick(levels.full_every, tick), detailed)
        };
        let name = dim.id().trim_start_matches("minecraft:");
        let (prefix, at) = (format!("{name}."), format!("@{name}"));
        let mut rows = Vec::new();
        let mut parts = format!(
            " {name}.random={:016x} {name}.rtv={}",
            self.world.random.state(),
            self.block_ticks.random_tick_value,
        );
        parts += &self.level_parts(&prefix, &at, full, detailed, &mut rows);
        let mut levels = levels.lock().unwrap_or_else(PoisonError::into_inner);
        levels.parts += &parts;
        levels.rows.append(&mut rows);
    }

    /// The server's own parts (docs/tick-digest.md).
    pub(super) fn server_parts(
        &self,
        fixture: Option<&mut Fixture>,
        detailed: bool,
        rows: &mut Vec<String>,
    ) -> String {
        // The next entity id and the next join number: one taken and not used shows here first.
        let upcoming = JoinSeq::upcoming().get();
        let mut line = format!(" counters={}/{upcoming}", self.next_entity);
        if detailed {
            rows.push(format!("n {} {upcoming}", self.next_entity));
        }
        let (mut part, mut data) = (Part::new(), Part::new());
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
                // What its file would hold now (`playerdata::player_tag`), its UUIDs zeroed.
                let mut tag = horizon_client::lan::playerdata::player_tag(guest);
                mask_uuids(&mut tag);
                data.add((guest.entity_id, &tag));
                if detailed {
                    tag_rows(&format!("d {}", guest.entity_id), &tag, rows);
                }
            }
        }
        line += &format!(" players={}", part.show());
        line += &format!(" playerdata={}", data.show());
        if let Some(fixture) = fixture {
            line += &format!(" packets={}", packets(fixture, detailed, rows));
        }
        // The chunk results applied since the last line, in order (fixture mode).
        let mut part = Part::new();
        for row in horizon_client::streaming::take_applied() {
            part.add(row);
            if detailed {
                rows.push(format!("a {row:?}"));
            }
        }
        line += &format!(" applied={}", part.show());
        line
    }
}
