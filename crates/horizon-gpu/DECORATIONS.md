# Where the decoration CPU time goes

Measurements of 15/11. Everything else is on the GPU (density, aquifers, beardifier, surface, ores,
caves, light, sections, biomes). The throughput of `/pregen` is limited by the 6 physical cores
(THROUGHPUT.md): each percent of CPU saved per chunk shows up roughly as throughput.

Decorations (`Decoration::decorate`, generator_decoration.rs) take **34 to 35% of the CPU**, i.e.
**2.5 to 3.0 ms per chunk**.

## Per feature

`HORIZON_FEATURE_TIMES=1` measures the time of each placed feature, of the structures (per structure),
of the GPU ore prediction and of the rest of `decorate`. The ranking is written to latest.log every
10,000 chunks (`FeatureTimes`, generator_decoration.rs). It is wall time, measured while the other
workers run. Script `/tmp/ft1.sh "x z" ...` (profiling binaries, gpu copy, radius 1,600 blocks).

By family, as a share of the decoration time:

| Family | 3150000/−3170000 (2,963 µs per chunk) | −3200000/3210000 (2,535 µs per chunk) |
|---|---|---|
| Ores (GPU prediction included) | ~31% | ~35% |
| Trees | ~23% | ~19% |
| Deep dark sculk | 7.5% | 4.6% |
| Amethyst geode | 5.8% | 6.1% |
| Lush caves, lichen, vines | 5.9% | 5.7% |
| Speleothems | 4.9% | 5.1% |
| Snow and ice (`freeze_top_layer`) | 4.5% | 5.4% |
| Structures | 3.3% | 4.8% |
| Monster rooms | 1.7% | 1.8% |
| `decorate` setup | 1.6% | 1.8% |

The heaviest items of the first zone:

| Feature | Share | Per call | Calls | Placed |
|---|---|---|---|---|
| `trees_birch_and_oak_leaf_litter` | 8.4% | 909 µs | 10,924 | 71% |
| ore prediction | 8.3% | 246 µs | 40,000 | |
| `sculk_patch_deep_dark` | 7.1% | 1,059 µs | 7,946 | 7% |
| `ore_dirt` | 6.1% | 180 µs | 40,002 | 96% |
| `amethyst_geode` | 5.8% | 172 µs | 40,002 | 4% |
| `trees_jungle` | 5.0% | 1,539 µs | 3,820 | 69% |
| `freeze_top_layer` | 4.5% | 134 µs | 40,000 | 100% |
| `pointed_dripstone` | 3.3% | 251 µs | 15,726 | 70% |
| `dark_forest_vegetation` | 3.2% | 1,558 µs | 2,451 | 75% |
| `ore_coal_lower` | 2.5% | 72 µs | 40,001 | 98% |
| `glow_lichen` | 2.1% | 63 µs | 40,001 | 76% |
| granite, diorite, tuff, andesite | ~2% each | ~60 µs | 40,002 | 99% |

The monument costs 25 to 29 ms per chunk that has one, but these chunks are rare (0.4 to 1.6%).

## Inside the features (perf profile)

Profiles `/tmp/profil_b1.sh` (zones 2800000/−2820000 and 3250000/−3270000), analysed by
`/tmp/sous_deco.py` and `/tmp/sous_filtre.py` (functions in inclusive time and in self time under a
given frame). Shares of the total CPU:

- **Ores, ~10%**:
  - writing the cells the GPU computed: `OrePlan::place` 5.8%, of which
    `TerrainChunk::replace_row_raw` 4.4%. Per cell, there is a read (`TerrainBlocks::block`), the
    target test, then `BlockSection::set`, which recomputes the position and looks the state up in
    the palette.
  - prediction: `predict_ores` 3.8%, with the placement chain and the biome test 2.5%, the spheres
    0.9% and the upload to the GPU 0.7%.
- **Geode, 2.6%**, of which 1.8% for the noise: `NoiseStack::get3` is evaluated for each block of the
  33 × 33 × 33 box, about 4 ms per placed geode.
- **Snow and ice, 2.2 to 3.5%**: the biome temperature costs 1.2%, of which 0.9% for its cache alone
  (SipHash hashing and an eviction mimicking vanilla's LRU cache). On top of that come the "should
  snow" and "should freeze" tests (biome per position), snow survival and the writes.
- **Sculk, 3.1% where there is deep dark**: `spread_all` 2.5%, of which `can_attach_to` 1.1% and the
  block reads 0.8%.
- **Trees, 2.9%**: a cost spread between shape updates (0.8%), writes, foliage, the set of leaves in
  Java order and the heightmaps.

## Leads, from the most to the least profitable

1. **Writing the ore cells** (~6% of the CPU). Replace a row directly in the section: one section
   lookup, one position per cell for the read and the write, and memoizing the target test by palette
   id rather than by state. Same writes, in the same order. Estimated gain: 2 to 3% of the CPU.
2. **Geode** (~2%). The noise only shifts a sum that is compared with `outer_crust`, and a block below
   that threshold is skipped without a random draw. With a rigorous bound of the noise, there is no
   need to evaluate it where the sum cannot reach the threshold, which is almost the whole box. That
   bound has to be written: Horizon has no equivalent of vanilla's `maxValue`. Estimated gain: about
   1.5%.
3. **Snow and ice** (1.7 to 3.5%). The temperature is a pure function; its cache only avoids
   recomputations, and costs more than the computation itself. A direct computation, and a single
   biome lookup per column. Estimated gain: 1 to 2%.
4. **Ore prediction** (~3%): biome tests and spheres. Uncertain gain, around 1%.
5. **Sculk** (~2% where there is some): `can_attach_to` cached per state and per face.
6. **Trees** (~7%): no single hot spot.

Leads 1 to 3 together would give about 5% of the CPU per chunk, so of throughput. All of them can
stay exact, and are checked by the probe, the vanilla parity and the ore test mode.

## What was done (15/11)

Originals in `/tmp/deco-avant/` on the server.

1. **Snow and ice.** `BiomeTemperature::uncached` (biome_temperature.rs) computes the temperature
   without a cache. `World::biome_should_freeze` and `biome_should_snow` (generator_decoration.rs) use
   it, and read the weather bindings through the reference to the decoration instead of cloning them
   at each call (two `Arc`s shared by all workers). `freeze_top_layer` goes from 134-136 to
   **48-49 µs per chunk (−64%)**.
2. **Ore cells.** `BlockSection::replace_with` (chunk.rs) reads the slot once, passes the state to the
   test and writes to the same slot, with the logic of `set` (moved into `set_slot`).
   `TerrainChunk::replace_row_raw` (terrain.rs) uses it for the sections that have non-air blocks,
   and keeps reading `block` elsewhere (`air` in an air section, `void_air` outside the height). The
   big blobs (dirt, granite, diorite, tuff, andesite, gravel) gain 7 to 13%, i.e. 5 to 20 µs per chunk
   each.
3. **Geode: abandoned.** The rigorous bound of the noise is too wide to skip anything. For one Perlin
   layer, `range` gives 2; an analysis of the interpolation gives 1.5 at best. The geode's
   `NormalNoise` has two layers of amplitude ~1.11, multiplied by 0.05 and by the 3 or 4 points of the
   geode: the margin exceeds 0.67, while the `outer_crust` threshold is about 0.48. No block was
   skipped, and the added sum made the geode 13% slower. The change was removed. Still possible:
   caching, per cell of the Perlin lattice, the gradients of the eight corners, for the blocks that
   share it along a row (exact, about 0.5% of the CPU).

Checks:

- chunk, terrain, temperature, ore and snow unit tests: 19 pass;
- probe byte-identical to the previous outputs on 4 zones (including a snowy_taiga, a snowy_plains
  and a frozen_river zone), on the CPU path as on the all-GPU path;
- vanilla parity at 0 differences on 785 chunks, everything on the GPU as on the CPU.

Measurements. Time per feature (`/tmp/ft1.sh`, same zones as above, gpub copy): the decoration goes
from 2,963 to 2,921 µs per chunk and from 2,535 to 2,376 µs (geode still included). Release bench
`/tmp/ab20.sh` (previous binaries in `target/avant-deco/`, gpub copy, against the new ones, gpu copy,
without the geode):

| Zone | Throughput before → after | CPU per chunk |
|---|---|---|
| 3400000/−3420000 | 1,584 → 1,593 chunks/s (+0.6%) | 5.69 → 5.61 ms (−1.4%) |
| −3430000/3440000 | 1,836 → 1,779 (−3.1%) | 4.81 → 4.81 ms (0%) |
| 3450000/−3470000 | 1,756 → 1,865 (+6.2%) | 4.94 → 4.78 ms (−3.2%) |

About −1.5% of CPU per chunk: less than the estimated 3 to 5%, because the geode was abandoned and
writing the ore cells only gains about ten percent, not a third. The throughput stays within the
noise between zones.


## Ore cells by row and by state (16/11, dual channel)

Profile before (`/tmp/profil_b1.sh`, zone 4750000/−4770000, `/tmp/sous_deco.py`): the ores weigh
11.5% of the CPU, 7.4% for writing the cells the GPU computed (`OrePlan::place`; per cell, a dynamic
call to the target test, `set_slot`, and two property lookups for the non-air count) and 4.6% for
the prediction.

What changed:

- `TerrainChunk::replace_row_states` (terrain.rs) is `replace_row_raw` for a decision that depends
  only on the state: it asks the decision once per state met in the row (`StateMemo`, with the change
  to the non-air count), reads the slot once (`BlockSection::slot_state`) and writes through
  `BlockSection::set_at`, which only changes the word and two counts when the new state is the last
  one looked up and no palette entry dies or fills the section; otherwise `set_slot` as before.
  `PlacementContext::replace_row_states` (default: `replace_row`) and `World` route it per chunk.
  `OreFeature::place_cells` uses it: its rules are pure and only read the position's Y, the row's.
- `predict_ores` keeps its vectors from one chunk to the next (`OreBuffers`, per worker thread): a
  chunk's plan holds some 20,000 spheres, which growing vectors copied again and again.

Checks:

- unit test `replace_row_states_matches_replace_row_raw`: 40 rounds of 3,000 random rows through
  uniform, air, growing and shrinking sections, with decisions that sometimes write air: same
  answers, same sections (`to_nbt`) and same non-air counts;
- probe byte-identical on 4 zones, all on the GPU and with `HORIZON_GPU_ORES=test` (0 mismatches);
  vanilla parity at 0 differences on 785 chunks.

Bench `/tmp/ab26.sh` (release, lz4; binaries from before in `target/avant-ores/`; the copies
alternate between the variants):

| Zone | CPU per chunk before → after | Throughput before → after |
|---|---|---|
| 4800000/−4820000 | 3.35 → 3.32 ms (−0.9%) | 2,343 → 2,344 chunks/s |
| −4830000/4840000 | 3.68 → 3.59 ms (−2.4%) | 2,225 → 2,269 (+2.0%) |
| 4850000/−4870000 | 2.89 → 2.85 ms (−1.4%) | 2,539 → 2,540 |

**About −1.6% of CPU per chunk.** Profile after (zone 5000000/−5020000): writing the cells goes
from 7.4% to 6.6% of the CPU (`set_slot` 1.3% → 0.3%, the decision 1.0% → 0.3%) and the prediction
from 4.6% to 4.2%. What is left of the writes is the loop over the cells itself, 3.7% in self time:
locating each cell in the packed words and reading it, spread over the sections of the 3 × 3
region. The prediction left is the placement chain and its biome tests (2.8%), shared with the
CPU path.


## Fixed collision faces for `can_attach` (16/11)

The shape updates looked like the next target (3% of the CPU in a forest zone), but they only weigh
1.2% on the zone of the ore profile (5000000/−5020000), plus 0.9% on the main thread, spread over
`update_shape_at`, tick scheduling and survival checks; `updateShape` reads other blocks and
schedules ticks, so it cannot be cached safely. Left as is.

That zone has deep dark: the sculk patches weigh **9.9% of the CPU** there (`spread_all` 7.9%), of
which `BlockSurvivalRegistry::can_attach_to` 3.7%. For each face, `can_attach` cloned the
neighbour's collision shape, took its face, built a full block to compare with, and dropped the
three (`collision_shape`, `face_shape`, `VoxelShape::block`, `drop_glue`).

`BlockSurvivalRegistry::fixed_face_full` (block_survival.rs) answers that second test once per
state, for its six faces, when the state's collision shape is the cached one (`collision_shape`
clones exactly that shape): a byte per state, bit 7 once known, kept in atomics shared by the
workers. States with a dynamic shape, and any face whose test fails, go the long way as before. It
serves the sculk spread and every multiface block (glow lichen, sculk veins, vines).

Checks: 507 horizon-world tests; probe byte-identical on 4 zones; vanilla parity at 0 differences on
785 chunks.

Bench `/tmp/ab26.sh` (release, lz4; binaries from before, with the ore change, in
`target/avant-sculk/`; the copies alternate):

| Zone | CPU per chunk before → after | Throughput before → after |
|---|---|---|
| 5000000/−5020000 (deep dark) | 3.50 → 3.30 ms (−5.7%) | 2,207 → 2,301 chunks/s (+4.3%) |
| 5050000/−5070000 | 3.18 → 3.14 ms (−1.3%) | 2,403 → 2,403 |
| −5080000/5090000 | 3.01 → 2.95 ms (−2.0%) | 2,516 → 2,521 |

From −1.3% to −5.7% of CPU per chunk depending on how much deep dark and lichen a zone has.


## Average over six zones (16/11, dual channel)

The composition of the decorations changes a lot from one zone to the next: sculk weighed 1.4% of
the CPU in one profile and 9.9% in another, trees 10.6% and 4.2%. To choose the next target on an
average, `/tmp/ft1.sh` (`HORIZON_FEATURE_TIMES=1`, profiling binaries, everything on the GPU)
measured six fresh zones of 40,401 chunks spread over the map (5200000/−5220000, −5230000/5240000,
5250000/5270000, −5280000/−5290000, 5300000/−5320000, −5330000/5340000), averaged by
`/tmp/ft_avg.py` with each zone weighted the same.

The decoration takes 1,525 to 2,450 µs per chunk, **1,890 on average**:

| Family | µs per chunk | Share | Range over the zones |
|---|---|---|---|
| Ores (GPU prediction included) | 596 | 31.6% | 575 to 624 |
| Deep dark sculk | 262 | 13.9% | 74 to 591 |
| Trees | 259 | 13.7% | 77 to 377 |
| Other features | 135 | 7.1% | 98 to 211 |
| Amethyst geode | 134 | 7.1% | 128 to 138 |
| Structures | 118 | 6.2% | 68 to 189 |
| Lush caves, lichen, vines | 107 | 5.7% | 65 to 150 |
| Speleothems | 79 | 4.2% | 24 to 180 |
| Snow and ice | 49 | 2.6% | 29 to 79 |
| Monster rooms | 38 | 2.0% | 36 to 41 |
| `decorate` setup and the rest | 113 | 6.0% | |

The largest single features: `sculk_patch_deep_dark` 251 µs (70 to 568), the ore prediction 148,
`ore_dirt` 136, `amethyst_geode` 134, `trees_birch_and_oak_leaf_litter` 129 (26 to 223).

These are wall times measured while the other workers run. `ore_dirt`, the first planned ore of
the step, also waits for the chunk's GPU ore batch (its 136 µs against 37 for granite, diorite or
andesite): part of the ores' share is that wait, which leaves the core to the other thread and
does not cost CPU. Since the decoration is about 45% of the CPU, the averages in CPU terms are
roughly: ores below 14%, sculk 6%, trees 6%, geode 3%.

## Shared decoration infrastructure (16/11)

Features (trees, sculk, geodes, monster rooms, placement modifiers, block predicates) all read
blocks through `TerrainRegion::block`. A profile with call graphs (`/tmp/profil_b1.sh 5400000
-5420000`) put the decoration code that belongs to no feature at 17% of the CPU, but spread out: the
callers of the block reads are scattered over every feature. The profiling binaries (no LTO, 16
codegen units) inline less than the release build, so the targets were taken from a flat profile of
the release binaries themselves (`/tmp/profil_rel.sh copy x z`, `perf record -F 999` without call
graph, zone 5450000/−5470000):

- **Block reads, 12.1% of the CPU** in three functions that release never inlined:
  `BlockSection::state` 4.8%, `TerrainRegion::block` 4.5%, `ChunkIndex::get` 2.8%.
  `ChunkIndex::get` stayed out of line because of its `HashMap` arm (chunks spread far apart), and
  `TerrainRegion::block` returned a `Result<_, RegionError>` of at least 24 bytes, through memory.
- **`state_face_sturdy`, 1.4%**, 92% of it on `lock incq` / `lock decq`: every call cloned
  `BlockBehavior::support`, an `Arc` shared by the eleven workers, so its counter's cache line
  travelled between cores. `center_sturdy`, `state_collision_empty` and `collision_full_block`
  cloned `support` or `collision` the same way. The clones were not needed: `World::blocks` gives an
  `&'a BlockBehavior` that does not borrow the world.
- An audit of the atomic instructions of every function above 0.15% (`perf annotate`, share of the
  samples on `lock` instructions) found only `StartCache::get` besides (0.7%, half of it on its
  `RwLock`), left as it is.

Changes:

- `TerrainRegion::block_at` (feature_region.rs) returns an `Option<BlockStateId>`, in a register;
  `block` wraps it. The chunk lookup (`ChunkIndex::get`, its sparse arm moved to a cold
  `sparse_get`), `TerrainChunk::block`, `TerrainBlocks::block`, `BlockSection::state` and
  `PackedStorage::read` are `#[inline(always)]`: a block read is one call instead of three. In the
  release binary only `block_at` and `sparse_get` remain as functions.
- `World` borrows `support` and `collision` instead of cloning them (generator_decoration.rs).

Checks: 507 horizon-world tests and 162 horizon-core tests; probe identical on the 4 overworld zones
(CPU and everything on the GPU), the 2 Nether zones and the 3 End zones; vanilla parity 0
differences on 785 chunks.

In the release profile afterwards (zone 5560000/−5570000), the block reads weigh 7.6% (`block_at`)
instead of 12.1%, and `state_face_sturdy` 0.08% instead of 1.4%. What is left of the read is mostly
its loads (section, storage word, palette), which miss the cache. The bench `/tmp/ab26.sh`
(`HS_AVANT` = `target/avant-infra/`, copies alternated) measures less than the profile suggests:

| Zone | CPU per chunk before → after | Throughput before → after |
|---|---|---|
| 5500000/−5520000 | 3.42 → 3.34 ms (−2.3%) | 2,341 → 2,368 chunks/s |
| −5530000/5540000 | 3.27 → 3.21 ms (−1.8%) | 2,411 → 2,404 chunks/s |

About 2% less CPU per chunk; the throughput difference is within the noise.

Left in the shared infrastructure, from the same profile: the biome filter (`BiomeZoom::quart_at_block`
0.9%, `uniform_section` 0.8%, `chunk_biomes` 0.7%, `biome_identity` 0.6%), the write path
(`BlockSection::set_slot` 1.9%, `set_block_worldgen` 1.7%, `World::set_block` 0.5%) and the
placement modifiers' own code (`modify_inner` 1.2%).
