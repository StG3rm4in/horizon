# Ores on the GPU (`HORIZON_GPU_ORES`)

## What it costs

`/pregen` profile of 15/11 (density and surface on the GPU, 1,023 chunks/s): `ore_feature` accounts
for 10.5% of the CPU in inclusive cost. Breakdown by line of `ore_feature.rs`:

| Item | Share of the CPU |
|---|---|
| Block writes (`set_section_block`) | 1.9% |
| Block reads (`section_block`) | 1.6% |
| Blob geometry: sphere pruning, per-axis squares, `pass_run`, cell loop, `tested` bitset | about 5% |
| Column lookup, rules, misc. | the rest |

The reads and writes depend on the state of the world at the ore step: they stay on the CPU. The
geometry only depends on the random draws: that is what moves to the GPU.

## Why the geometry can be computed in advance

A chunk's decoration depends on the order: the features of the neighbours decorated before it write
into its region. The ores therefore read a world that is only known at step 6 (`underground_ores`).
But for a *plannable* ore:

- `discard_chance_on_air_exposure` = 0: no per-cell draw, no neighbour read;
- pure target rules (Horizon's `heights`: no draw);
- placement modifiers that read no block (`count`, `in_square`, `height_range`, `rarity_filter`,
  `biome`, `random_chance`, `offset`, `fixed`, `randomly_selected` among those).

Each feature is reseeded by `set_feature_seed`. The draws of an attempt (position, then direction and
heights of the ends, then `size` × `next_double` for the spheres) then only depend on the world
through the column lookup (`OCEAN_FLOOR_WG` under the blob), which decides whether the sphere draws
happen. In vanilla Minecraft 26.3, this covers dirt, gravel, granite, diorite, andesite, tuff,
coal_upper, iron (×3), copper (×2), redstone (×2), lapis, emerald, gold_extra, infested and clay.
Excluded: coal_lower, gold, gold_lower and the diamonds, with a chance strictly between 0 and 1, as
well as lapis_buried and diamond_buried, at 1.

With a zero chance, each cell of the blob is read once, and only its own write depends on it. The
visiting order of a blob's cells therefore changes nothing: the GPU returns a set, not an ordered
list. The blobs of a feature are still applied in attempt order, since a blob reads what the previous
one wrote.

## How it runs

1. **Start of the chunk's decoration** (`Decoration::predict_ores`). For each plannable ore of step 6,
   in order and on the same biome selection as `FeatureDecorationPlan::decorate` (`step_features`),
   the CPU replays the placement chain on a separate random source, with `OreFeature::predict`. The
   column lookup is done on the region as it is at that moment. The spheres of the attempts that find
   a column are sent to the GPU (`OreGpu::submit`).
2. **GPU** (`ores.cu`, `kore_blobs`, one block of 128 threads per blob):
   - sphere pruning in the CPU's exact order, with the `i2` loop cut into packs of 32 within a warp.
     The first `i2` that removes `i1` stops the loop, since a removed `i1` removes nothing more;
   - per-axis squares `((v + 0.5) − c) / r` squared, over the bounds `max(floor(c ∓ r), start)`;
   - for each row (x, y), the mask of the z such that `(xd² + yd²) + zd² < 1`, merged by `atomicOr`
     in shared memory.

   Everything is computed in double with the `__d*_rn` intrinsics, without contraction
   (`--fmad=false`). As output, `size_xz × size_y` masks per blob, plus an error flag if a cell falls
   outside the box: the CPU would then have raised `WorkLimit`.
3. **Steps 0 to 5 on the CPU** meanwhile.
4. **Step 6, planned feature** (`OrePlan::place`):
   - if a write may have moved a heightmap since the prediction (the world's `floor_writes` counter,
     incremented with `raise_floor_bound`), the CPU redoes all the column lookups of the feature. Ore
     writes do not touch the heightmaps, so they can be done first;
   - if one of them differs from the prediction (steps 0 to 5 may have raised the ground), the feature
     goes through the normal CPU path. Nothing has been drawn or written, and the feature's randomness
     is intact;
   - otherwise, after waiting for the batch (once per chunk), each blob is applied by
     `OreFeature::place_cells`, with the same per-cell test as `do_place` (`place_cell`, shared).

The caches the prediction fills early do not change the answers:
- the ground bounds `floor_bounds` stay upper bounds, raised at each write;
- heightmaps primed and then updated give the same result as a late priming.

## Measurements

**Version 1**: cells applied one by one by `place_cell`, batch uploaded in 3 copies, read back in 2.
- `/pregen` benches (`/tmp/ab9.sh`, radius 1,600 blocks):
  - zone 410000/−430000: 981 → 986 chunks/s, CPU 10.49 → 10.59 ms per chunk;
  - zone −440000/450000: 1,018 → 1,043 chunks/s, CPU 9.10 → 8.93 ms per chunk.
- Profile (`/tmp/sous_ore3.py`): the ores go from 10.5% to 9.1% of the CPU.
  - planned step 6: 5.7%, of which about 5% of per-cell work (write zone, read, rule memo, write);
  - prediction: 2.2%, of which 0.6% for the upload to the GPU and 0.5% for the sphere draws;
  - ores left on the CPU: 1.15%.

  So the geometry was only about 2.4 points, not 5: the first per-line breakdown counted the work
  that remains per cell inside the visiting loop.

**Version 2**:
- application by row:
  - `PlacementContext::replace_row` handles a whole row, cell by cell by default;
  - `World` overrides it to handle one chunk at a time: write zone and height checked once, then
    `TerrainChunk::replace_row_raw`;
  - the Y intervals of the rules are computed once per row;
- batch uploaded in a single copy, read back in a single one.

Benches of version 2:
- zone 500000/−520000: 937 → 965 chunks/s (+3.0%), CPU 10.84 → 10.77 ms per chunk;
- zone −530000/540000: 1,026 → 1,048 chunks/s (+2.1%), CPU 8.97 → 8.91 ms per chunk.

Probe identical, test mode at 0 mismatches, vanilla parity at 0 differences (1,775 batches,
0 fallbacks).

Profile of version 2 (zone 560000/−580000, 1,143 chunks/s with the profiling binaries): the ores
account for **7.2% of the CPU** (against 10.5% without the GPU), split as follows:
- planned step 6: 3.7%, of which 2.4% of reads and writes per row;
- prediction: 2.4%, of which 0.5% for the sphere draws, 0.5% for placement chains, 0.5% for the
  upload and 0.3% for column lookups;
- ores left on the CPU (air exposure discard chance between 0 and 1): 1.0%.

In return, GPU wait and driver go from 5.5 to 6.7%. The CUDA driver makes three more calls per chunk,
and the ore kernels share the GPU with those of the density and the surface. The throughput gains 2
to 3%, and the CPU time per chunk drops less than the share of the ores.

## Variables

- `HORIZON_GPU_ORES=1`: ores on the GPU, independently of `HORIZON_GPU_DENSITY`.
- `HORIZON_GPU_ORES=test`: for each blob, also compares the GPU masks with the CPU's
  (`OreFeature::blob_rows`). On a mismatch, the feature goes through the CPU and the mismatch is
  counted.

At shutdown, a `GPU ores:` line gives the number of batches, of blobs per batch, the upload and wait
times, the CPU fallbacks and the mismatches.
