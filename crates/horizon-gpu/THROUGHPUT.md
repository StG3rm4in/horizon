# What limits `/pregen`

Measurements of 15/11, everything on the GPU (density, aquifers, beardifier, surface, ores, caves,
light, sections, biomes), on the server's i5-10600K: 6 physical cores, 12 threads, a single 16 GB
memory stick at 3,600 MT/s.

## During a `/pregen`

Script `/tmp/pg1.sh x z` (analysis `/tmp/pg1_analyse.py`). Profiling binaries, fresh zone
2900000/−2920000, 1,450 chunks/s.

CPU per thread, in cores:

| Threads | Cores | Per thread |
|---|---|---|
| 11 workers | 5.22 | 0.47 |
| 6 save compression threads | 1.69 | 0.28 |
| main thread | 0.69 | |
| region writing | 0.59 | |
| GPU (events, light) | 0.10 | |
| **total** | **8.29** | |

Occupancy: 73% of the 12 logical CPUs, from 71 to 79% each; the two threads of each physical core
are as busy as each other.

`perf stat` over 10 s: IPC 1.15. The memory controller reads 6.5 GiB/s and writes 4.3 GiB/s, about
40% of what a single DDR4-3600 channel allows.

Off-CPU time of the workers (`offcputime`, 10 s, user stacks): 49 s out of 110 s. The causes:

| Cause | Time |
|---|---|
| empty work queue (`jobs::work`) | 16.8 s |
| waiting for a chunk's GPU fill (`Engine::chunk_codes`) | 10.5 s |
| structure starts lock (`StructureState::build_starts`) | 4.0 s |
| shared `OnceLock`s | 3.1 s |
| waiting for the GPU ores | 2.3 s |
| waiting for the GPU caves | 1.5 s |
| region reads | 1.0 s |
| density regions | 0.5 s |
| biome regions | 0.3 s |

Trace (`HORIZON_PREGEN_TRACE=1`):

| Step | Value |
|---|---|
| generations in flight | 28 to 31 (limit 33, three per worker) |
| decorations | 8 to 18 (limit 22) |
| lights | 8 to 20 |
| empty queue | 2 to 9% of the updates |

## By number of workers

Script `/tmp/ab19.sh "copy workers x z" ...` (release, radius 1,600 blocks, fresh zones).
`HORIZON_WORKER_THREADS=N` gives N − 1 workers. Each number goes once through each copy:

| Zone | 5 workers | 11 (default) | 15 workers |
|---|---|---|---|
| 3000000/−3020000 | 1,440 chunks/s (−15.9%) | 1,711 | 1,603 (−6.3%) |
| 3050000/−3070000 | 1,437 (−9.1%) | 1,581 | 1,606 (+1.6%) |
| 3100000/−3120000 | 1,386 (−8.9%) | 1,521 | 1,547 (+1.7%) |
| CPU per chunk | 4.05 to 4.42 ms | 5.18 to 5.83 ms | 5.65 to 6.11 ms |
| Busy cores | 5.8 to 6.1 | 8.6 | 9.0 to 9.4 |

## Conclusion

**The limit is the 6 physical cores.** "8 to 9 cores busy out of 12" counts logical threads: each
physical core almost always has at least one active thread. With 5 workers, about one per core,
`/pregen` keeps 89% of its throughput. The 6 extra workers, which share the same cores through
hyperthreading, only add 12% of throughput, for 30% more CPU per chunk. Beyond that, nothing.

Nothing else is saturated:

- memory bandwidth: ~40%;
- GPU: ~60%;
- main thread: 0.69 core;
- region writing: 0.59 core;
- compression threads: 0.28 core each.

The workers' waits (GPU, structures, empty queue) leave the core to the other thread: removing them
would only give back the hyperthreading share.

The conclusions of PREGEN.md and GPU_WAIT.md ("throughput not limited by the CPU, 8 to 9 cores out of
12") were wrong for the same reason: the CPU was already the limit, through its physical cores.

## What would make it faster

Only less CPU per chunk. The items, from the profile of 15/11 with the biomes on the GPU (BIOMES.md):

| Item | Share of the CPU | Note |
|---|---|---|
| Decorations | 29.6% | |
| Saving | 17.9% | NBT encoding and allocations, including `mi_page_malloc_zero` 3% |
| GPU wait and driver | 11.2% | |
| zlib compression | 9.9% | level 6, like vanilla |
| Main thread | 7.8% | |

For zlib, Horizon already reads `region-file-compression` from server.properties (`deflate`, `lz4`,
`gzip`, `none`, like vanilla since 1.20.5). The bench copies are in `deflate`.


## Measured again in lz4 (15/11, evening)

With the regions in lz4 (bench copies), the palettes without repacking (SAVING.md), the optimized
snow and ores (DECORATIONS.md). Script `/tmp/lim1.sh`: profile, per-thread diagnosis, workers bench.
`/tmp/pg1.sh` now runs perf stat with `LC_ALL=C`, otherwise the locale's decimal comma breaks the
reading of the counters.

By number of workers (`/tmp/ab19.sh`, release, fresh zones):

| Zone | 5 workers | 11 (default) | 15 workers |
|---|---|---|---|
| 3950000/−3970000 | 1,422 chunks/s (−16.4%) | 1,701 | 1,730 (+1.7%) |
| −3980000/3990000 | 1,686 (−14.2%) | 1,965 | 2,014 (+2.5%) |
| CPU per chunk | 3.06 to 3.67 ms | 4.39 to 5.19 ms | 4.61 to 5.48 ms |

During a `/pregen` (profiling binaries, zone 3900000/−3920000, 1,768 chunks/s):

| Threads | Cores |
|---|---|
| 11 workers | 5.31 (0.48 each) |
| 6 compression threads | 0.82 (0.14 each) |
| main thread | 0.67 |
| region writing | 0.36 |
| **total** | **7.34** (65% of the logical CPUs, against 73% in deflate) |

- Memory: 6.7 GiB/s read and 4.4 written (~40%); IPC 1.08.
- Off-CPU time of the workers: 39% of the time. Empty queue 11%, the chunk's GPU fill
  (`chunk_codes`) 10%, `build_starts` lock 5%.

**The limit is still the CPU, through its 6 physical cores.** 5 workers keep 85% of the throughput,
15 workers only add 2%. The main thread (0.67 core) is not a bottleneck; neither are the compression
and the region writing.

Profile (`/tmp/profil_b1.sh`, zone 3850000/−3870000, 1,861 chunks/s with the profiling binaries):

| Item | Share of the CPU |
|---|---|
| Decorations | 36.2% |
| Saving (NBT tree, writing) | 14.3% |
| GPU wait and driver | 12.5% |
| Main thread | 8.4% |
| Generation (rest) | 6.7% |
| Structures | 4.8% |
| Chunk finalization | 4.3% |
| Biomes and climate | 3.2% |
| Caves (CPU) | 2.6% |
| Compression (lz4) | 2.3% |

- The most expensive function in self time is `_mi_page_malloc_zero` (4.6%), mimalloc's allocation
  path. `/tmp/appelants_alloc.py` shows it is scattered; the biggest caller is the save's NBT tree
  (`JavaString::from` strings, compounds), about 1.5%.
- The CUDA driver (`/tmp/sous_gpu.py`) weighs 5.2%: the surface launches (`enqueue`) 1.4%, and the
  light, cave and ore uploads 2.1%.


## Dual channel (16/11)

The server now has two 16 GB sticks, one per channel, at 3,600 MT/s (XMP), and new thermal paste.
Same code as the light cache bench (`/tmp/ab25.sh`, 1,999 to 2,070 chunks/s and 4.01 to 4.20 ms of
CPU per chunk with 11 workers in single channel). Script `/tmp/ab19.sh`, release, lz4, radius 1,600
blocks, fresh zones; each setting goes once through each copy:

| Zone | `HORIZON_WORKER_THREADS=5` (4 workers) | default (11 workers) | 15 (14 workers) |
|---|---|---|---|
| 4650000/−4670000 | 1,344 chunks/s (−34.1%) | 2,040 | 2,137 (+4.8%) |
| −4680000/4690000 | 1,498 (−32.8%) | 2,231 | 2,301 (+3.1%) |
| 4700000/−4720000 | 1,516 (−34.3%) | 2,308 | 2,355 (+2.0%) |
| CPU per chunk | 2.64 to 2.98 ms | 3.54 to 4.00 ms | 3.63 to 4.09 ms |
| Busy cores | 3.9 to 4.0 | 7.9 to 8.0 | 8.1 to 8.6 |

- With the default 11 workers: **2,193 chunks/s on average (+7% over single channel), 3.70 ms of CPU
  per chunk (−8%)**. The zones differ from the single channel ones, so a few percent of that is zone
  noise.
- The hyperthreads now pay off: the default gives +51% over 4 workers, against +18% in single
  channel. In single channel, the threads sharing a core mostly waited for the same memory channel;
  with two channels, the extra workers add half again as much throughput.
- The CPU is still the limit: 15 workers only add 2 to 5%.

## After merging upstream main (16/11)

The merge of upstream main at 54717b2965 (436 commits, the mod API merge among them) leaves the
throughput where it was. `/tmp/ab26.sh` against the binaries from before the merge (kept in
`~/bancs/avant-merge/bin/`, next to a copy of the resources they were prepared with), copies
alternated:

| Zone | CPU per chunk before → after | Throughput before → after |
|---|---|---|
| 5600000/−5620000 | 3.62 → 3.59 ms | 2,183 → 2,231 chunks/s |
| −5630000/5640000 | 3.29 → 3.28 ms | 2,425 → 2,432 chunks/s |

After a merge, the resources must be prepared again (`horizon-prepare`, a few seconds) before the
tests, the probe or the parity harness say anything: with the resources of 14/11, 202 horizon-world
tests failed on pure upstream as on the merge, upstream's own probe output differed in 4 of the 9
zones, and the vanilla parity found 372 chunks apart. Prepared again, the probe of the merge is
byte-identical to pure upstream's in the three dimensions (CPU, everything on the GPU, test modes)
and the parity finds 0 differences.

The game binary is now built from `horizon-game`: `cargo build --release -p horizon-game --bin
horizon-client -p horizon-server`.

## The scheduling thread (16/11)

The thread that schedules the chunks (the server's main thread) took 10.7% of the CPU under
`/pregen` (call-graph profile after the merge, `~/bancs/perfg-merge.data`): `Game::update_world`
8.6%, of it `Streaming::complete_commit` 5.1% (`DecorationOrder::blocked` 2.0%, `rank` 0.9%), the
plan updates 2.5%; the server tick itself 2.1%. Counted with uprobes (`/tmp/calls1.sh`, bpftrace on
`update_world` and `complete_commit`), it ran **3,300 passes a second**, about 250 us of CPU each,
for 0.85 of a core: `pump_chunks` waits for one job result (at most 5 ms) and runs `update_world`,
which rescans every proto chunk the streamer holds, for each result.

`pump_chunks` now leaves at least `HORIZON_PUMP_GAP_US` (1 ms by default) between two passes; the
results that come back in between wait in the channel, and `Streaming::update` takes them all. The
decoration order does not depend on when a pass runs (`DecorationOrder::blocked` holds a chunk back
while a lower-ranked one within two of it is pending, in flight included).

`/tmp/ab26.sh` with the variants `gapN` (the same bench build, `HORIZON_PUMP_GAP_US=N`), two fresh
zones, the three copies:

| Gap | CPU per chunk, 5950000/−5970000 | −5980000/5990000 | Throughput |
|---|---|---|---|
| 0 (a pass per result) | 4.20 ms | 3.14 ms | 2,038 / 2,470 chunks/s |
| 1 ms | 3.97 ms (−5.5%) | 2.95 ms (−6.1%) | 2,050 / 2,470 |
| 2 ms | 3.94 ms (−6.2%) | 2.93 ms (−6.7%) | 2,037 / 2,403 |

At 1 ms the thread runs 516 passes a second and takes 0.45 of a core instead of 0.85. The
throughput does not move: the scheduling thread's CPU was not what held `/pregen` back, and the
workers do not take the core it leaves (the process uses about 7.4 of the 12 logical CPUs).

## What the workers wait for (16/11)

With the scheduling thread at 0.45 of a core, the 11 chunk workers were busy 58% of the time each
and the machine's logical CPUs 65%. Their time off the CPU over 10 s of `/pregen` (`/tmp/pg1.sh`,
offcputime on the bench build), 37.9 s in all:

| What | Time |
|---|---|
| No job queued (the pool's wait) | 12.1 s |
| The chunk's own GPU stream (`Engine::chunk_codes`: the copy, and the surface queued behind it) | 8.2 s |
| Structure starts (`StructureState::build_starts`, a lock) | 6.3 s |
| A region another worker computes (`OnceLock`, density and surface regions) | 2.8 s |
| Ore and cave batches, structure chunks, aquifers, biomes (stream waits) | 5.5 s |

The `/pregen` trace (`HORIZON_PREGEN_TRACE=1`) finds the job queue empty at 7 to 13% of the
passes, generation near its cap (about 29 of 33 jobs, `workers * 3`), decoration at 11 to 19 of 22.

**More workers.** Waiting threads leave their core to others, so more workers than the machine's
threads overlap the waits: with `HORIZON_WORKER_THREADS=18` (17 chunk workers), `/pregen` ran
2,301 → 2,418 and 2,126 → 2,213 chunks/s (+5.1 and +4.1%), for 5 to 7.5% more CPU per chunk. The
server's `[engine] worker_threads` sets the variable (docs/hosting.md).

**Computing the density regions ahead: no gain, not kept.** A thread computed the eight neighbours
of each region a worker first asked for, so that workers would find them ready (patch kept in
`~/bancs/avant/prefetch/prefetch.diff`; exact: probe identical, parity 0). Against the same build
with `HORIZON_GPU_PREFETCH=0`: 2,213 → 2,225 and 2,605 → 2,572 chunks/s, within the noise, for up to
2 GB more GPU memory. A region is 3 ms of GPU work for 64 chunks; each chunk then waits about 1 ms
on its own stream (its surface and its readback), which the regions computed ahead made longer
(8.2 → 10.6 s) by keeping the GPU busier, and the remaining `OnceLock` waits are mostly on the
surface's regions. What would remove the stream waits is doing the CPU's own work on a chunk while
its GPU work runs, rather than one after the other.

## Overlapping a chunk's CPU and GPU work: the ceiling (16/11)

A generation job runs its stages one after the other: biomes, structures (references, starts, the
beardifier), the cave batch sent to the GPU (already asynchronous), the fill (density, block codes
and surface on the GPU, then a synchronous wait of about 1 ms, then the CPU's writes), the carvers
(the cave batch's wait, then the mask applied), the heightmaps. The GPU's part of the fill could be
sent right after the biomes and the structures done while it runs, speculatively without a
beardifier (about 6% of the chunks have one, and would redo theirs once their structures are
known).

Waits overlapped inside a job are what more workers give for free, so the worker count bounds what
such an overlap could bring (`/tmp/ab19.sh`, two fresh zones, `HORIZON_PUMP_GAP_US=1000`):

| Chunk workers | 6250000/−6270000 | −6280000/6290000 | CPU per chunk |
|---|---|---|---|
| 11 (default) | 2,622 chunks/s | 2,411 | 2.68 / 3.02 ms |
| 17 (`worker_threads = 18`) | 2,756 (+5.1%) | 2,547 (+5.7%) | 2.88 / 3.22 ms |
| 23 | 2,785 (+6.2%) | 2,572 (+6.7%) | 3.05 / 3.38 ms |

Past 17 workers, a sixth more of them adds about 1% while the CPU per chunk keeps rising: the
hyperthreads share the 6 physical cores, which are close to saturated although only about 8 of the
12 logical CPUs show busy. Overlapping the GPU wait inside a job would meet the same ceiling, about
1 to 2% over 17 workers, for an asynchronous chunk API in horizon-gpu, a speculation on the
beardifier and GPU buffers held across the structure stage: not done. The throughput still comes
from less CPU per chunk.

At 17 workers too, the scheduling gap of 1 ms holds: against a pass per result, 2,411 → 2,425 and
2,580 → 2,605 chunks/s, CPU per chunk −4.5 and −3.0%.
