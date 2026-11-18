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
`horizon-client`). The run folders go to `/data/tick-digest` (`--work`).

## The server side

`crates/horizon-client/src/game/interface/dedicated/tick_digest.rs`, switched on by environment
variables (the tool sets them):

| Variable | What |
|---|---|
| `HORIZON_TICK_DIGEST=FILE` | one line per tick, and a line `0` for the state before the first tick |
| `HORIZON_TICK_DIGEST_FULL=N` | the blocks and light of every loaded section every N ticks (default 20, 0 never) |
| `HORIZON_TICK_DIGEST_DETAIL=FROM-TO` | the rows behind the hashes of those ticks, in `FILE.detail` |
| `HORIZON_TICK_FIXTURE_TICKS=N` | the fixture mode: N ticks back to back, then the server stops |
| `HORIZON_TICK_FIXTURE_COMMANDS=FILE` | console commands run before given ticks (lines `TICK COMMAND`) |
| `HORIZON_TICK_FIXTURE_PLAYERS=N` | simulated players, one every `HORIZON_TICK_FIXTURE_JOIN_EVERY` ticks (default 10) |

A digest line holds, for the level in place once the tick ends (the Overworld while nobody is in
another dimension): the game time, the level random's state, `Level.randValue`, the block writes of the
tick (position and new state, in order), every scheduled tick (sorted, with its sub-tick order), every
entity of the entity lists in their order (id, type, position, velocity, rotation, as bits), the
players (uuid, entity id, position), the chunks that random tick, the chunks whose entities are still
pending, the chunk map's walk order, the loaded chunks, the chunk results applied (fixture mode) and,
every `FULL` ticks, the decoded blocks and light of every loaded section.

Not in the digest: entity NBT beyond the motion (health, AI memories, inventories), block entity
contents (only their count), the levels that are not in place, what is sent to the players.

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
What it is sent goes to a channel that is drained unread. Their configuration phase is skipped, and
they never answer keep-alives (a channel sender has none).

## Generation

`--gpu` generates with every GPU stage; the default is the CPU. With the GPU, two runs of one fixture
can differ before the first tick: blocks and light (the full hash) and the scheduled fluid ticks of a
few chunks, and the light results come back in another order. GPU generation is not yet the same from
one run to the next, which is a generation question, not a tick one; until it is, check the tick on CPU
generation, or on a world generated once and copied for each run.

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
