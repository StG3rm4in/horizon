# Caves on the GPU (`HORIZON_GPU_CAVES`)

## What it costs

`/pregen` profile of 15/11 (density, surface and ores on the GPU, 1,143 chunks/s with the profiling
binaries, `/tmp/sous_grottes.py`): caves account for **12.4% of the CPU**.

| Item | Share of the CPU |
|---|---|
| Mask (`generate_carving_mask`) | 4.65% |
| of which tunnel tracing (`CaveCarver::carve`) | 3.42% |
| of which origin biome (`carver_biome`) | 0.72% |
| of which canyons | 0.24% |
| Mask application (`apply_carving_mask_shared`) | 5.51% |
| of which aquifer (`compute_substance`) | 2.35% |
| of which material under a carved grass block (`top_material_cell`) | 1.19% |
| of which rule engine setup | 1.02% |
| of which writes | 0.49% |
| Building the CPU aquifer (`NoiseBasedAquifer::new`), now only used by the caves | 2.29% |

The mask of a chunk merges the carvers of the 17 × 17 origin chunks around it: each origin is traced
again for each of the 289 chunks it can touch.

## Why the mask can be computed elsewhere

- The draws of a carver call (`WorldCarver::carve` after its start test) do not depend on the target
  chunk: number of starts, positions, rooms, tunnel parameters and seeds, the canyon and its width
  table. Only the tracing (`can_reach` stops a tunnel) and the rasterization of the ellipsoids depend
  on the target.
- The mask is a union of cells: the order of the ellipsoids does not matter.
- The substance of a carved cell (`compute_substance` with a density of 0) only depends on its
  position: the statuses of the aquifer cells are pure functions, already computed on the GPU for the
  whole region.

## How it runs

1. **Planning, once per origin** (`WorldCarver::plan`, `carver_plans` cache, shared and per worker):
   the same draws as `carve`, in the same order, stored as rooms, tunnels (seed, position, thickness,
   yaw, pitch, end, vertical scale, run multipliers, floor) and canyons (start, state of their random
   source after the width table, table). `carve` is unchanged: the vanilla probe also checks the state
   of the randomness after an out-of-bounds error.
2. **Start of the chunk** (`Pipeline::submit_caves`): the elements of the 289 origins that can reach
   the chunk (a path moves at most one block per step) go to the GPU with one counter per call
   (Horizon's budget of 2^24 tested cells).
3. **GPU, `kcv_trace`** (`caves.cu`, one block of 128 threads per element): thread 0 traces the tunnel
   and its branches (stack), or the canyon, exactly like `CaveRun::tunnel` and `CanyonCarver::trace`
   (LegacyRandom, Horizon's sine table, float and double `_rn`, without contraction). It keeps the
   ellipsoids that pass `can_reach` and the range test. The block then rasterizes them like
   `carve_ellipsoid_with_budget` (axes precomputed per ellipsoid) into a mask in shared memory, merged
   into the chunk's mask by `atomicOr`.
4. **Chunk fill**: as soon as the chunk's region is on the GPU, `kcv_substance` (aquifer module)
   computes the substance of each cell of the mask, stored in bit order, and the result is copied,
   still on the batch's stream, off the worker's critical path.
5. **Cave step**: the CPU waits for the batch (already finished), rebuilds the `CarvingMask` from the
   words and applies it with `apply_carving_mask_with`, which reads the substances instead of the
   aquifer. The rule engine for the material under grass is only built if a chunk needs it.
   The CPU aquifer is only built if the chunk goes through the CPU fill or the CPU substance.

The chunk falls back to the CPU, without changing the result, if the GPU flags an error (capacity,
random bound, unsupported radius provider), if a call exceeds its budget (the CPU then raises the
error itself), or if the mask has more cells than the copied codes.

## Variables

- `HORIZON_GPU_CAVES=1`: caves on the GPU. The substance of the cells also needs the density and the
  aquifers on the GPU (`HORIZON_GPU_DENSITY`); without them, only the mask comes from the GPU.
- `HORIZON_GPU_CAVES=test`: compares each mask with `generate_carving_mask` and each substance with
  the CPU aquifer, then built for every chunk. On a mismatch, the chunk goes through the CPU.

At shutdown, a `GPU caves:` line gives the number of batches, of elements and cells per batch, the
upload and wait times, the fallbacks and the mismatches.

## Checks

- Unit test `tests/carver_plan.rs`: for vanilla's caves, Nether caves and canyons, 120 seeds and 85
  targets each, `plan` then `execute` carve the same cells as `carve`.
- `chunk_dump` probe (`/tmp/gr1.sh`), 3 zones of 256 chunks (0,0; −1250,940; 750,−1270): output
  byte-identical to the CPU with the GPU mask alone in test mode, everything on the GPU in test mode,
  and everything on the GPU. 0 mask mismatches, 0 substance mismatches, 0 fallbacks. 80 to 120
  elements and 850 to 1,700 carved cells per chunk on average (10,903 at most).
- Vanilla parity (`tools/test_chunk_generation_parity.py --seeds 1`): **0 differences on 785
  chunks**, with `HORIZON_GPU_CAVES=1` as with `test`. Per dimension, in test mode:
  - overworld (627 chunks compared, 64 structure sites): 3,479 batches, 3,378 of them with the GPU
    aquifer (the others have an empty mask), 0 mask mismatches, 0 substance mismatches;
  - Nether (113 chunks): 611 batches, GPU mask alone (no GPU density), 0 mismatches, including a chunk
    with 16,384 carved cells.

## Measurements

**Version 1**: one plan per origin (289 lookups per chunk), three GPU buffers per batch.
- `/pregen` benches (`/tmp/ab10.sh`, radius 1,600 blocks, density + surface + ores on the GPU in both
  variants):
  - zone 600000/−620000: 1,010 → 1,040 chunks/s (+3.0%), CPU 10.09 → 9.72 ms per chunk;
  - zone −630000/640000: 1,036 → 1,095 chunks/s (+5.7%), CPU 8.92 → 8.38 ms per chunk.
- Profile (zone 700000/−720000, 1,096 chunks/s with the profiling binaries, `/tmp/sous_grottes2.py`):
  the caves go from 12.4% to about 4.9% of the CPU.
  - preparing the upload (`submit_caves`): 1.77%, of which 0.9% for lookups in the plan caches (289
    per chunk, under a lock when the worker's cache runs dry) and 0.4% of CUDA driver;
  - mask application: 2.76%, of which 1.59% for the material under carved grass, itself 1.34% in the
    preliminary surface (interpolated `find_top_surface`), recomputed at every cell;
  - GPU aquifer enqueued: 0.23%; batch wait: 0.13%; CPU aquifer: 0.04%.
  - GPU wait and driver go from 6.7 to 8.0%.

**Version 2**:
- plans by blocks of 8 × 8 origins: 9 lookups per chunk;
- a single GPU buffer per batch: one copy on the way up (elements, widths and the output's zeros),
  one on the way back (output and substances); 5 driver calls per chunk instead of 8;
- `MaterialDensitySources::retarget` keeps the preliminary surface while the column does not change
  (the same plane gives the same values).

Checks of version 2: probe identical on the 3 zones in the 3 modes; vanilla parity at 0 differences
(full, and overworld in test mode); test mode at 0 mask mismatches and 0 substance mismatches
(overworld 3,479 batches, Nether 611 batches). Uploading a batch: 9 to 12 µs instead of 13 to 18 µs.

Benches of version 2 (fresh zones):

| Zone | Without GPU caves | With | CPU per chunk |
|---|---|---|---|
| 800000/−820000 | 957 chunks/s | 996 chunks/s (+4.1%) | 10.73 → 10.15 ms (−5.4%) |
| −830000/840000 | 1,119 chunks/s | 1,185 chunks/s (+5.9%) | 8.41 → 7.74 ms (−8.0%) |
| 850000/−870000 | 1,066 chunks/s | 1,111 chunks/s (+4.3%) | 9.69 → 9.08 ms (−6.3%) |

GPU utilization goes from 27-31% to 33-38%. The parity harness only shows the first 5 error lines of
the probe: with `HORIZON_GPU_ORES` as well, the `GPU caves` line does not appear there; run the test
mode without the ores to read it.
