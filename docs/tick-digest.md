# The tick exactness harness

An optimization of the server tick must not change what the tick does: the same blocks, entities,
scheduled ticks and random draws, in the same order. `tools/tick_digest.py` checks that: it runs one
fixture on a dedicated server twice (two builds, or one build twice) and compares a digest of every
tick. The first tick that differs, and the part of the digest, point at the change; the rows behind the
hashes of that tick show what changed.

```sh
python3 tools/tick_digest.py check --ticks 400 --players 20 --join-every 5 --repeat 3   # one build, 4 runs
python3 tools/tick_digest.py check --bin .local/base/release --bin2 .local/candidate/release \
    --ticks 400 --players 20 --join-every 5                                              # two builds
python3 tools/tick_digest.py compare DIGEST_A DIGEST_B
```

`--bin` defaults to `.local/digest-target/release` (a folder with `horizon-server` and
`horizon-client`). The run folders go to `/data/tick-digest` (`--work`): `run-a`, `run-b` and, after a
difference, `run-a-detail` and `run-b-detail`, each emptied before it is used again (a check removes
the detail ones of the check before). A check holds its `--work` folder while it runs
(`.tick_digest.lock`, `flock`): a second check on the same folder stops at once instead of removing
the first one's worlds.

Each run's game gets fresh `HORIZON_HOME`, `HORIZON_LAUNCHER_HOME`, `XDG_DATA_HOME` and
`XDG_CONFIG_HOME` folders under `RUN/home/`, with `HORIZON_NO_AUDIO=1` and
`HORIZON_LAUNCHER_NO_SELF_UPDATE=1` (`lab.game_env`): nothing of the machine's own is read or written.
The prepared game files are still found: `horizon-server` looks for them in the server folder's
`resources/` and as `.local/horizon-client/26.3` in the program's folder or one of the three above it.
`--parallel` sets `HORIZON_PARALLEL_TICK` for the game: `serial` (`0`), `exact`, or `inherit` (the
default), which keeps what the caller set. Each run prints the mode the game runs in and writes the
`HORIZON_*` and `XDG_*` variables it passed to `RUN/env.txt`.

## The server side

`crates/horizon-client/src/game/interface/dedicated/tick_digest.rs`, switched on by environment
variables (the tool sets them):

| Variable | What |
|---|---|
| `HORIZON_TICK_DIGEST=FILE` | one line per tick, and a line `0` for the state before the first tick |
| `HORIZON_TICK_DIGEST_FULL=N` | the blocks and light of every loaded section and the NBT of every block entity every N ticks (default 20, 0 never) |
| `HORIZON_TICK_DIGEST_DETAIL=FROM-TO` | the rows behind the hashes of those ticks, in `FILE.detail` (on full ticks, one per section: its blocks' and its light's hash) |
| `HORIZON_TICK_FIXTURE_TICKS=N` | the fixture mode: N ticks back to back, then the server stops |
| `HORIZON_TICK_FIXTURE_COMMANDS=FILE` | console commands run before given ticks (lines `TICK COMMAND`) |
| `HORIZON_TICK_FIXTURE_PLAYERS=N` | simulated players, one every `HORIZON_TICK_FIXTURE_JOIN_EVERY` ticks (default 10) |
| `HORIZON_TICK_DIGEST_SKIP_PACKETS=IDS` | more packet ids (hex, comma-separated) for the `packets` part to leave out, to find the packet that differs between two runs |

A digest line holds, once the tick ends: the parts of the level in place (the first level the tick
ran, the Overworld while it ticks) under plain names, the server's own parts, then the parts of every
other level that ticked, under its name (`the_nether.writes=…`), read when that level's tick ends
(`levels.rs` calls `tick_digest_level` where it calls the state digest). With the default fixture
only the Overworld ticks. A level that does not tick in a tick has no parts in its line. The rows of
`HORIZON_TICK_DIGEST_DETAIL` start with their kind; another level's kinds end with `@level`
(`w@the_nether`). The code: `tick_digest/parts.rs`.

| Part | Rows | What is hashed | When |
|---|---|---|---|
| `time`, `dim`, `random`, `rtv` | | the game time, the level in place, its random's 48-bit state, `Level.randValue`; another level's `NAME.random`, `NAME.rtv` | every tick |
| `gauss` | `g` | the level random's cached gaussian (`-` for none) | every tick |
| `writes` | `w` | every block write since the level's last line: position and new state, in order | every tick |
| `scheduled` | `s` | every block and fluid tick, sorted, with its priority and sub-tick order | every tick |
| `entities` | `e` | each entity of the 12 entity lists, list by list: id, type, position, velocity, rotation, as bits | every tick |
| `fields` | `h` | each entity's join number and fields, read in place: a mob's health, absorption, hurt time, damage cooldown, invulnerable and death time, fire, freeze, air, fall distance, on-ground, removed and persistent flags, head and body yaw, age, no-action time, swing (time, on, length), target and last attacker by entity id, the time it was hit, effects (id, amplifier, duration); the other kinds' own (an item's health, count, age and pickup delay, an orb's value and count, a TNT's fuse, a cart's damage, fuse and fuel, a projectile's age and owner...) | every tick |
| `randoms` | `r` | each entity's own random (mobs, items, boats, minecarts, projectiles, rockets, tridents): state and cached gaussian | every tick |
| `tickers` | `t` | the block entity ticker list in its order: position, slot and, for a live one, its block and NBT as the save writes it (`block_entity_encode::encode`) | every tick |
| `ticking` | `c` | the chunks that random tick, sorted | every tick |
| `pending` | | the chunks whose entities are not loaded yet, sorted | every tick |
| `order` | `m` | the chunk map's walk order | every tick |
| `chunks` | | the loaded chunks, sorted | every tick |
| `full` | `f` | per section of every loaded chunk: the decoded blocks and the light | every `FULL` ticks |
| `blockents` | `b` | every block entity of the loaded chunks, by position: its block and NBT | every `FULL` ticks |
| `counters` | `n` | the next entity id and the next join number, as numbers | every tick |
| `players` | `p` | each player, in the hub's order: uuid, entity id, position | every tick |
| `playerdata` | `d` | each player's file as it would be saved now (`playerdata::player_tag`): health, food and exhaustion, experience, effects, inventory, abilities, game mode, motion, fall distance, fire, air, attributes... | every tick |
| `packets` | `k` | fixture mode: what each fixture player was sent since the last line, what the fixture's commands and the console sent included, in its order of sending, the players in the order they joined | every tick |
| `applied` | `a` | the chunk results applied since the last line, in order (fixture mode) | every tick |

What differs between two runs of one build is set aside: the `UPDATE_ADVANCEMENTS` packet (its criteria
carry the date they were met), `ADD_ENTITY`'s UUID, and in the NBT of block entities and players every
entity UUID (`UUID`, a ridden vehicle's `Attach`, a trial spawner's `current_mobs`, a creaking heart's
`creaking`, a conduit's `Target`, a sculk listener's `source` and `projectile_owner`, a bee's
`AngryAt`), zeroed before hashing: they are random (`save::random_uuid`, seeded from the clock), and
how many there are still counts. A mob's target and last attacker are hashed by entity id; one known
by its UUID alone (not loaded yet) only as present or not. A player's arm swings (`SWING_ANIMATION`) are
hashed sorted, after its other packets of the tick: `lan_swings` sends them in the order of a `HashMap`.

Not in the digest: what an entity holds beyond the fields above (AI goals and paths, brain memories,
attributes, equipment, a villager's trades), the entities outside the 12 lists (item frames,
paintings, leash knots, end crystals, the dragon, area effect clouds, lightning, evoker fangs, ominous
item spawners), the loot a container has not rolled yet (kept beside its block entity), the world's
saved data (POI, raids, weather, world border, scoreboard), a level in a tick it does not tick, and
the advancements packet. The
fixture's players are in creative mode and fly 24 blocks above their spawn; the 100-player check runs
600 ticks of daytime in the Overworld. So the digest shows that a change keeps what that scene does; it
is not a comparison with vanilla, and survival play, nights, the Nether and the End are less covered.

## The fixture mode

Worker timing decides, in a normal run, which chunk results a tick sees and in which order they are
applied: the chunk map's insertion order, which saved ticks unpack first, which chunks are loaded when a
player arrives. The fixture mode takes timing out (`streaming::set_deterministic`):

- every take of chunk results first waits until no job is queued or running (`Streaming::quiescent`,
  a count of the jobs a worker took and has not answered), then applies them sorted by kind and column
  (`Done::order_key`);
- before each tick, rounds of that until no chunk work is left (`Game::fixture_settle`);
- `Streaming::wait` (`/forceload add`'s synchronous loads, the startup) waits for every job instead of
  taking the first result;
- the spawn search of a new world runs in the same rounds;
- no wait between ticks, no chunk pumping between them, no far-ring brake from the measured MSPT;
- entity randoms from a counter per thread (`HORIZON_TEST_SEEDS=1`) instead of the clock.

The tick itself runs unchanged. Light results that were superseded before they were applied are left
out of the applied log: whether one ran before it was replaced depends on timing, and it changes
nothing.

## Simulated players

A fixture player plays as `horizon-bots --behaviour mixed`: of every five, three wander within 48
blocks of their spawn, one flies away at 22 blocks a second, one stands; all 24 blocks above their
spawn, moving only while the server has sent them the chunk they are in. They enter through the events
a network connection posts (`NetEvent::Joined`, then `NetEvent::Packet` for each packet), before the
tick whose `Hub::poll` takes them, so the join, the packet handling and the chunk sending are the
server's own. What a real client waits for, the player takes from the server's state: the teleport to
accept (`awaiting_teleport`), the chunk batches to acknowledge (`unacknowledged`, at 64 chunks a tick).
What it is sent goes to a channel read once a tick (`parts::receive`): in creative mode the players make
nothing of it; in the survival scene they read their deaths and the monsters from it (`Phantom::observe`).
Their configuration phase is skipped, and
they never answer keep-alives (a channel sender has none).

## The survival scene

`--scene survival` plays a second scene, where the default one never goes. The default scene is
unchanged (its properties, commands and players), so its baselines still hold.

```sh
python3 tools/tick_digest.py check --scene survival --ticks 600 --players 100 --join-every 2 --repeat 2
python3 tools/tick_digest.py check --scene survival --bin .local/base/release --bin2 .local/candidate/release \
    --ticks 600 --players 100 --join-every 2
```

- **The server**: survival mode, forced, on hard, flight allowed (a player standing where a creeper
  opened the floor is not kicked for floating). Tick 1 sets midnight (`time set 18000`: 600 ticks
  stay in the night) and `respawn_radius 0`, so that every player joins and respawns in the middle
  of the arena.
- **The arena**, built on tick 1 around the world spawn's block (the console's `~`): a 25 × 25 stone
  floor cleared 9 blocks up. Nothing that collides stands within 5 blocks of its middle, so the
  players' straight walks never run into a block. On the ring 5 blocks out: cobwebs, a stone
  pressure plate, sweet berry bushes, powder snow, magma, wither roses, eyeblossoms (open and
  closed), fire on netherrack and a weighted plate. Past the ring: a zombie spawner with light-blind
  spawn rules (the fire and the lava light the arena), a trial spawner (4 zombies, then its rewards,
  a cooldown of 200 ticks), a vault, a lit furnace, a chest over a hopper into a chest, lava, two
  cacti and a tripwire; 40 blocks out, a spawner and a trial spawner that no player comes near.
  Tick 60 brings zombies given absorption, skeletons, spiders, a creeper with a blast of 1, a
  villager, an iron golem and two bees; tick 120 gives absorption to the players near the middle;
  tick 300 brings a second wave with a husk and a witch.
- **The Nether**: a room at y 70 (a netherrack floor, magma, nether wart on soul sand, soul fire,
  lava) and, above it, the roof: one flat layer of bedrock at y 127. **The End**: an end stone
  platform at (1200, 59, 0) with a chorus flower. It is far from the dragon's island: the fight
  needs a player within 192 blocks of it and the island's chunks loaded, so it never starts (no
  dragon, no crystals, no boss bar). The three areas are force-loaded on tick 1, so the Nether and
  the End tick from then on.
- **The players**: out of creative mode, a fixture player plays
  `crates/horizon-client/src/game/interface/dedicated/tick_digest/scene.rs`. It walks on the floors
  the scene built, at their height, and claims the ground. On its tick in 10 it hits the nearest
  monster within 2.5 blocks across (an attack, then the arm's swing), or one time in four swings at
  the air. It respawns 20 ticks after it dies. Player i is teleported 5 ticks after it joins:
  - to the Nether if i % 10 = 9: onto the roof if i % 20 = 19, where it walks straight on (new
    chunks), else into the room, where it walks round a square;
  - to the End if i % 25 = 12, where it walks round a square.

  The others stay in the Overworld, by i % 5: 0 and 1 patrol the ring (one way or the other, in by
  one of its 4 spokes), 2 chase the nearest monster inside the ring, 3 guard the spawner, the trial
  spawner, the vault or the furnace, 4 stand in one of the hazards. With 100 players: 86 in the
  Overworld, 10 in the Nether (5 on the roof), 4 in the End. A player who dies elsewhere respawns in
  the arena and takes the Overworld walk of its index.
- **What the players read**: what they know of the entities around them and of their own death
  comes from what they are sent (`Phantom::observe` takes every body drained from their channel);
  what the server waits for they take from its state, as the creative players do.
- **Not reached**: ominous trial spawners and ominous item spawners (nobody has Bad Omen),
  `/item modify`, the dragon's fight, raids and weather.
- **Between two runs of one build**, beyond the two packets set aside above: the trial spawner's
  `current_mobs` holds the UUIDs of its mobs (`save::random_uuid`, from the clock), and any other
  entity UUID a part hashes (an arrow's owner, a neutral mob's anger at a mob) differs too.

## Generation

The default is the CPU. `--gpu` generates with every GPU stage, the density kernels compiled before the
first chunk (`HORIZON_GPU_DENSITY=wait`; with `1`, the chunks generated while they compile take the
CPU). A light job on the GPU counts as running until its result is sent, so the fixture mode waits for
it as for a worker's job. A fixture gives the same digest with the GPU as with the CPU, tick for tick.

## Measured on 18/11/2026

- One build, CPU generation: 300 ticks without players (256 forced chunks, 21 to about 600 entities)
  and 400 ticks with 20 players joining every 5 ticks (400 to 1,575 loaded chunks, 785 entities at the
  end): 4 runs each, identical at every tick.
- 600 ticks with 100 players joining every 2 ticks (4,843 loaded chunks and 2,141 entities at the end,
  51 s a run): 2 runs, identical at every tick.
- A deliberate change (postprocess_chunks walking its chunks sorted instead of in the chunk map's
  order), built apart (`~/bancs/scripts/build_digest.sh digest-b-target`): caught at tick 1, the block
  writes and the scheduled ticks the same in number but not in order; the rows of tick 1 start, for the
  change, with the lowest chunk of the sorted order.

What it took to get there, in the order the digest showed it: the startup's spawn search took chunk
results as they came; `/forceload add` loads each chunk at once, taking the first result to finish;
the take of a plan update while jobs still ran; and superseded light results.

With the GPU (later the same day): the light jobs on the GPU were not counted as running, so their
results came in other batches; then 3 runs in 24 still differed in content, one chunk at a time, and
the rows per section showed a chunk with the surface of badlands (biome 0), terracotta down to y -16,
around it the trees it no longer had. A new set of surface buffers is zeroed on its own CUDA stream
while the surface fills it on the stream of the chunk's codes, and event tracking is off: the zeroing
could land after the upload of the biomes and of the preliminary surface. That happens while the
buffer pool fills, at the start of a server, and the chunk is saved so. Fixed in
`SurfaceGpu::resource` (horizon-gpu), which waits for its stream: 72 one-tick runs identical to each
other and to the CPU's, and 400 ticks with 20 players, 4 GPU runs identical to each other and to a
CPU run at every tick.
