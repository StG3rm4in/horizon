# `/pregen` progress

## The finding

With everything on the GPU (density, aquifers, beardifier, surface, ores, caves, light), `/pregen`
only keeps 8 to 9 cores busy out of 12. The `HORIZON_PREGEN_TRACE=1` trace (average of each step over
one second in latest.log, `game/commands/pregen.rs`) shows, on a zone of 201 × 201 chunks:

| Measure | Value |
|---|---|
| Worker queue empty | 28 to 57% of the updates |
| Decorations in flight | 5.5 on average (limit: 2 per worker, 24) |
| Generations in flight | 15 to 20 (limit: 3 per worker, 36) |
| Lights in flight | 4 to 8 |

## Why

`/pregen` sweeps the zone in bands of columns (up to 256 wide) over a few rows (at least 1,024
chunks: 6 rows here). `Streaming::set_band` gives the chunks of the band their decoration rank row by
row, column after column, so that the order does not depend on timing.

But a chunk is only decorated once all the chunks of lower rank less than two away from it are
decorated (`DecorationOrder::blocked`: their write regions overlap). Column after column, each chunk
waits for the one on its left: a row is decorated serially, and only the 5 or 6 rows of the band move
forward together.

## The change

The default since 15/11 (`HORIZON_PREGEN_ORDER=rows` for the old order): each row in three passes,
one column out of three in each pass. Two chunks three columns apart have no common write region, nor
any common chunk in their 3 × 3: a pass decorates up to 67 chunks at once over 201 columns. The order
stays fixed by the geometry of the band, independent of timing.

The world does not change in quality, but not identically either: where the decorations of
neighbouring chunks overlap (trees, veins, sculk at the edges), the order decides what goes first.
Neither order is vanilla's, which has no pregeneration (vanilla's order around a player,
`DecorationOrder::reassign`, is untouched).

## Measurements

Trace (server built in parity-target, same fresh zone 1550000/−1570000):

| Order | Throughput | Empty queue | Decorations | Generations |
|---|---|---|---|---|
| column after column | 1,260 chunks/s | 28 to 39% | 5.5 | 18 |
| three passes | 1,380 chunks/s (+9.5%) | 4 to 8% | 6 to 12 | 29 to 30 |

Release benches (radius 1,600 blocks, fresh zones, everything on the GPU):

| Zone | Column after column | Three passes | Busy cores |
|---|---|---|---|
| 1600000/−1620000 | 1,305 chunks/s | 1,386 (+6.2%) | 8.6 → 9.4 |
| −1630000/1640000 | 1,386 chunks/s | 1,635 (+18.0%) | 7.6 → 9.4 |
| 1650000/−1670000 | 1,303 chunks/s | 1,344 (+3.2%) | 8.7 → 9.3 |

The CPU time per chunk rises by 4 to 6%: more cores work at the same time on a machine with a single
memory stick. More workers (`HORIZON_WORKER_THREADS=14` or `16`) bring nothing clear
(−2.6 to +2.7%).

## Reproducibility

A measurement on fresh servers, CPU and GPU, with `tools/gpu_worldgen.py pregen --keep` and
`compare-worlds`, is in BENCHMARKS.md (17/11). The first comparison, below, came before it.

`tools/gpu_worldgen.py compare-worlds` compares the chunks two worlds saved. Same zone of
51 × 51 chunks, generated twice (12 then 5 workers):

| Order | Chunks with different blocks | With different biomes |
|---|---|---|
| column after column | 3 out of 2,601 | 0 |
| three passes | 1 out of 2,601 | 0 |

The few mismatches, and those of the block entities (loot seeds, counters) and of the scheduled
ticks, already existed on the CPU. They are fluid states at chunk borders; freezing the server's
ticks during `/pregen` does not remove them (BENCHMARKS.md, 17/11).

## Correction of 15/11

The throughput was already limited by the CPU: "8 to 9 cores out of 12" counts logical threads, and
the 10600K only has 6 physical cores. See THROUGHPUT.md.
