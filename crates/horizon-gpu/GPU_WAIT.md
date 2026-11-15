# Waiting for the GPU

## What it costs

`/pregen` profile of 15/11 with everything on the GPU (density, aquifers, beardifier, surface, ores,
caves, light; `/tmp/sous_gpu.py`): the "GPU wait and driver" step weighs **12.7% of the CPU**, of
which 8.9% in libcuda and cudarc. By caller:

| Caller | Share of the CPU |
|---|---|
| `chunk_codes` (waiting for a chunk's fill and surface) | 4.6% (+ 0.4% with structures) |
| `SurfaceGpu::enqueue` (launches) | 0.7% |
| light, cave and ore uploads | 1.2% |
| `OreGpu::rows` (`as_slice` for every blob) | 0.55% |
| region computation | 0.5% |
| ore and cave waits | 0.4% |

cudarc's `PinnedHostSlice::as_slice` calls `cuEventSynchronize` every time, even without event
tracking.

## What was tried

`/pregen` benches (`/tmp/ab12.sh` and `/tmp/ab13.sh`, release, radius 1,600 blocks, fresh zones,
everything on the GPU):

| Variant | 1200000/−1220000 | −1230000/1240000 | 1250000/−1270000 |
|---|---|---|---|
| no priorities (`HORIZON_GPU_PRIORITY=0`) | 1,376 chunks/s, 6.76 ms | 1,335, 6.26 ms | 1,386, 6.73 ms |
| priority streams | 1,381, 6.74 ms | 1,329, 6.28 ms | 1,386, 6.75 ms |
| priority streams and `HORIZON_GPU_SYNC=yield` | 1,369, 6.84 ms | 1,333, 6.35 ms | 1,376, 6.83 ms |

(chunks/s, then CPU time per chunk.) Neither the priorities nor `yield` change anything: the
throughput is no longer limited by the GPU wait or by the CPU (8 to 9 cores busy out of 12). It is
limited by the progress of `/pregen`, which sweeps the zone in bands of at least 1,024 chunks and only
moves a band forward once its first rows are lit.

Active waiting therefore only costs CPU. The blocking mode (the thread sleeps), set aside when the CPU
was saturated (calls of 0.9 ms instead of 0.5), no longer slows anything down:

| Variant | 1300000/−1320000 | −1330000/1340000 |
|---|---|---|
| active waiting (the default back then) | 1,307 chunks/s, 7.04 ms | 1,247, 6.58 ms |
| `HORIZON_GPU_SYNC=block` | 1,303, 6.62 ms (−6.0%) | 1,256, 6.30 ms (−4.3%) |

## What is kept

- **Blocking wait by default**: `HORIZON_GPU_SYNC` is `block` unless asked otherwise (`spin`,
  `yield`, `auto` for CUDA's heuristic, often active).
- Critical path streams (fill, surface, regions) at the GPU's highest priority, the asynchronous
  batches at the default priority. No measured effect today; `HORIZON_GPU_PRIORITY=0` puts them back
  on an equal footing.
- `OreGpu::wait` reads the readback buffer once; `rows` no longer calls the driver for every blob.
  Ore probe in test mode: identical, 0 mismatches.

Confirmation with the new default, without a variable (`/tmp/at2.sh`, fresh zones):

| Variant | 1400000/−1420000 | −1430000/1440000 |
|---|---|---|
| `HORIZON_GPU_SYNC=auto` (former default) | 1,200 chunks/s, 7.64 ms | 1,239, 6.56 ms |
| default (blocking) | 1,179, 7.26 ms (−5.0%) | 1,262, 6.15 ms (−6.3%) |

The throughput varies by −1.7% and +1.9%: that is the noise from one zone to another.

## Correction of 15/11

The throughput was already limited by the CPU: "8 to 9 cores out of 12" counts logical threads, and
the 10600K only has 6 physical cores. See THROUGHPUT.md.


## Calls and copies around the GPU (15/11, evening)

Profile in lz4, with direct NBT writing (`/tmp/sous_gpu2.py`: per horizon-gpu function and per kind
of leaf). The "GPU wait and driver" item also contains the CPU work of the horizon-gpu crate, not
only the driver calls:

| Origin | Share of the CPU |
|---|---|
| Light upload (`gpu_light_submit` → `LightGpu::submit`) | 3.7% |
| Surface copies (`chunk_blocks_surface`) | 1.3% |
| `chunk_codes`, of which `memchr` 0.9% | 1.4% |
| Ore and cave uploads | ~0.9% |
| Surface and substance launches | ~0.9% |
| Driver outside horizon-gpu (wake-ups, interrupts) | 1.1% |

### What was changed

- **The code check** (`chunk_codes`, `chunk_codes_beard`, lib.rs). The 98,304 codes the GPU writes
  were read twice from memory: `contains(&-1)`, then the copy. They are now copied first, then checked
  in the cache. On error, `out` is zeroed. The generator only reads these codes if the GPU succeeded
  (`from_codes`). `memchr` goes from 0.92% to 0.19% of the CPU.
- **The saved heightmaps** (SAVING.md). `Heightmap::prime_sections` (horizon-world heightmap.rs,
  `HeightmapSections` trait) computes the same heights section by section. The save's view gives for
  each cell a mask of the maps it covers, taken from the section's palette by `unpack_ids_mapped`. So
  the section and the state of each block are no longer looked up. The cost goes from 1.51% to 0.93%
  of the CPU.

Checks:

- unit tests `prime_sections_matches_prime` (random columns) and
  `heightmaps_by_sections_match_cells` (horizon-client-lib, game states);
- `HORIZON_HEIGHTMAP_TEST=1` computes each map both ways and compares. `/pregen` in test mode
  (radius 800, zone 4150000/−4170000): **more than 16,384 chunks, 0 mismatches**;
- probe identical on 4 zones all on the GPU, and vanilla parity at 0 differences on 785 chunks.

Bench `/tmp/ab24.sh` (previous binaries in `target/avant-gpu/`, gpub copy; lz4):

| Zone | Throughput before → after | CPU per chunk |
|---|---|---|
| 4200000/−4220000 | 2,207 → 2,225 chunks/s (+0.8%) | 3.71 → 3.65 ms (−1.6%) |
| −4230000/4240000 | 1,984 → 1,979 (−0.3%) | 4.39 → 4.37 ms (−0.5%) |
| 4250000/−4270000 | 2,009 → 1,999 (−0.5%) | 4.18 → 4.17 ms (−0.2%) |

About −0.8% of CPU per chunk, within the bench's noise. The profile sees −1.3%.

### What is left

Profile after these changes (zone 4300000/−4320000, 2,082 chunks/s with the profiling binaries): the
item weighs 14.6% of the CPU (driver 5.0%, libc 4.3%, code 2.7%, kernel 2.3%).

- **Light upload: 4.1%.** Each lit chunk copies the section words of its 9 chunks (3 × 3
  neighbourhood, ~430 KB) into the upload buffer: each chunk is copied 9 times. The remedy would be a
  section cache on the GPU. It would be indexed by the identity of the `Arc<Section>`s, which it would
  keep alive, so that a modification goes through a copy on write. It would need recycled slots that
  do not free those an in-flight job uses, and synchronization between streams so that a job does not
  read a section that is still being uploaded. Estimated gain: 2 to 3%.
- **Surface copies: 1.9%.** The compact sections and the codes are copied from the pinned buffer into
  `Vec`s. Reading them in place would require keeping the buffer until the chunk is filled.
