# Surface rules on the GPU: porting plan

Goal: compute Horizon's surface step (`material_stage::build_columns`) on the GPU, bit for bit, and
return the final state of each block of the chunk after filling and surface.
Starting measurement (14/11/2026, profile with call graph, density and aquifers on the GPU):
`build_surface_shared` = 19.9% of the CPU time of `/pregen` (rules 7.2%, blocks 5.3%, loop 3.2%,
veins 3.0%, extensions 1.0%).

## What the surface does (Horizon's code)

For each column (lx, lz) of the chunk, in this order (x then z):

1. `original_top` = first free y of the column; biome at (x, original_top, z).
2. Eroded badlands (`material_extensions::eroded_badlands`) if this biome is `eroded_badlands`.
3. `top` = first free y, slope `gx`, `gz` = differences of first free y between neighbours (clamped
   to the chunk), `update_xz`: surface depth `depth = surface_depth(x, z)`.
4. Descent from `top` to `min_y`: air → `above = 0`, `water = MIN`; fluid → `water = y + 1` at the
   first one; solid → computation of `bottom` (bottom of the solid layer), `above += 1`, then, within
   the write bounds, the rule is applied with `update_y(above, y - bottom + 1, water, y)`. The block
   written replaces the default block (the rule is only applied where Horizon applies it).
5. Frozen oceans: icebergs (`frozen_ocean_with_melt`) after the descent.

Horizon's optimizations (height bands, variants above or below the preliminary surface, runs of
blocks, possible folding per biome) do not change the result: the GPU evaluates the generic rule.

## Rules and conditions

- Rules: `Block`, `Sequence` (the first child that answers), `Condition`, `Bandlands`, `OreVein`
  (density > 0, then `random.at(pos).next_float() > density` → nothing, richness, gap < 0, raw ore if
  `next_float() < raw_ore_chance`); randomness `root.from_hash_of("minecraft:ore")`.
- Conditions (pure, caches without effect on the result): `Hole` (depth <= 0), `Steep`
  (gx <= -4 or gz >= 4), `Biome`/`BiomeTag`, `Temperature` (cold_enough_to_snow),
  `AbovePreliminarySurface` (y >= floor(preliminary) + depth - 8), `NoiseThreshold`
  (`noise.get3(x, is_3d ? y : 0, z)` in f64 within [min, max]), `YAbove`, `Water`, `StoneDepth`
  (with the secondary noise), `VerticalGradient` (`root.from_hash_of(name).fork_positional()`,
  `next_float() < 1 + (y - low) / (high - low) * -1`), `Not`.

## Inputs

- Per column: `surface_depth` = `(int)((surface noise * 2.75 + 3) + root.at(x, 0, z).next_double() * 0.25)`,
  secondary noise, plane of the preliminary surface (floor), terracotta band (clay_bands_offset
  noise, table of 192 states drawn once).
- Per block: biome (biome zoom over the 3 × 3 chunks, `BiomeZoom::quart_at_block`), temperature
  (`BiomeTemperature`, temperature noises), vein densities (as volumes).

## Milestones (each checked bit for bit against the CPU before the next one)

- **S1**: inputs per column and random draws (depth, secondary, bands, condition noises, vertical
  gradients, vein randomness).
- **S2**: biome per block (zoom) and temperature.
- **S3**: rule tree translated to CUDA and descent of the columns; block by block comparison.
- **S4**: extensions (eroded badlands, icebergs).
- **S5**: integration: the GPU returns the final states, the CPU writes the sections; probe, vanilla
  parity, `/pregen`.

## Progress

- **S1, done (14/11/2026)**: `random.cu` (Xoroshiro, next_float, next_double, at), `generate_noises`,
  `evaluate_noises`, `positional_draws`, `PositionalRandom::xoroshiro_state` (horizon-core).
  Check `HORIZON_GPU_SURFACE_TEST=1` (generator.rs, `surface_gpu_test`): 64 noises over 20,000 points,
  draws of the root, ore, bedrock_floor, bedrock_roof and deepslate factories, and surface depth:
  identical to the CPU.
- **S2, done (14/11/2026)**: `surface.cu` (zoom `sf_quarter`, `sf_biome_quarter` over the 3 x 3
  chunks, 2D simplex `sf_simplex2`, `sf_temperature`), `generate_surface_s2`, `surface_s2_check`;
  horizon-core `Noise::as_simplex`, `BiomeZoom::seed`; horizon-world `temperature_noises()`.
  Zoom over 20,000 positions and 20 temperature combinations: identical to the CPU.

- **S3, done (14/11/2026)**: the surface rule and the column descent on the GPU, compared block by
  block with the blocks written by the CPU surface (`HORIZON_GPU_SURFACE_TEST=1`, `compare_surface_gpu`
  in generator.rs; `HORIZON_GPU_SURFACE_BIOMES=1` adds the columns per biome).
  - Neutral form in horizon-gpu: `SurfaceRule`, `SurfaceCondition`, `SurfaceSetup`; translation of the
    `MaterialRule` by `SurfaceIr` (generator.rs; references memoized by `Arc` address, anchors resolved
    on the CPU, biomes as ids, condition noises by name). 231 rules, 127 conditions, 9 noises.
  - `MaterialRule::OreVein.roots: Option<[FunctionId; 3]>` (filled by `MaterialRuleLibrary`, `None`
    for rules built without a graph, like the probe's fixture).
  - `generate_surface`: functions `C_k` / `R_k`, `ksurface` kernel (one thread per column).
  - Preliminary surface: `chunk_surface_level` = `interpolated` (cells 16 × 1 × 16) around
    `find_top_surface`; the CPU samples it **as a volume** on the chunk's y = 0 plane
    (`MaterialDensitySources`). The GPU does the same: a 16 × 1 × 16 volume program (`level`), with
    `find_top_surface` in volume mode (upper bound as a volume, density point by point).
  - Fixed along the way: an interpolation read by a 2D pass (Y-free node) added a corner grid that
    was never generated (`kernels` now generates the 2D passes in the loop over the grids, with
    `y = MY`); `__device__` tables with a constant initializer (temperatures as bits).
  - Result: **0 mismatches over 3,668 chunks, 346 M blocks compared, of which 60.5 M changed by the
    surface**, and 0 conditions that cannot be evaluated. Samples: 5 zones of 64 chunks, then
    25 + 36 + 32 small scattered zones (script `/tmp/s3_zones2.sh N STEP DX DZ OUTPUT` on the server).
    40 biomes at the surface, including badlands (terracotta bands), snowy_slopes, grove, meadow,
    ice_spikes, swamp, desert, stony_shore, windswept_gravelly_hills, dripstone_caves, sulfur_caves.
    Not seen yet: the peaks (jagged/frozen/stony_peaks), mangrove_swamp, mushroom_fields, cherry_grove,
    wooded_badlands. 38,269 columns ignored (eroded_badlands, frozen_ocean, deep_frozen_ocean: S4
    milestone).
  - No speed measurement at this stage: one launch and allocations per chunk, 256 threads.
- **S4, done (14/11/2026)**: the extensions in `ksurface` (`sf_pillar`, `sf_iceberg` in `surface.cu`,
  `SurfaceExtensions` in the configuration: biomes, 6 noises, packed ice, snow).
  - Eroded badlands pillar before the descent of its column (the pillar is solid for the descent,
    default block where the rule gives nothing); iceberg after, with its draws in the order of the
    short circuits and the melting (`temperature(x, sea, z) > 0.1`). Both only depend on the fill
    codes: the iceberg only writes into air and water, never over a block of the rule.
  - Column order (x then z): the extensions raise the first free y, which the slopes read. Three
    heights in shared memory (original, after the pillar, final); the slope reads the final one for
    x − 1 and z − 1 (already processed), the original one for x + 1 and z + 1, and the height after
    the pillar for the column itself. Logic copied from `build_columns`, but not checked by the data:
    no `Steep` condition is evaluated next to a column with an extension in the tested zones.
  - Result: **0 mismatches over 1,468 chunks (144 M blocks), of which 349,000 blocks written by the
    extensions** (4,595 from pillars over 13,640 eroded badlands columns, 344,000 from icebergs over
    frozen_ocean and deep_frozen_ocean), plus wooded_badlands; the snowy slopes zone stays at
    0 mismatches. `HORIZON_GPU_SURFACE_SKIP_EXT=1` still skips these columns if needed.

- **S5, integration (15/11/2026)**: `HORIZON_GPU_SURFACE=1` (with `HORIZON_GPU_DENSITY`).
  - `SurfaceGpu::chunk_states` returns the final state of each block (fill then surface) on 32 bits:
    the state on 30 bits, bit 30 if the surface writes a fluid state there (swamp water), bit 31 if the
    fill schedules a fluid update (code + 4 on water or lava).
  - Horizon: `terrain.rs::fill_noise_states` writes these states in one pass in fill order (runs of
    the same fluid-free state through `fill_solid_run`, heightmaps updated like `fill_noise_codes`),
    marks the fill's post-processing, then the surface's in surface order (x then z, decreasing y):
    the post-processing lists are ordered. The CPU surface is then skipped for the chunk. CPU fallback
    if a condition cannot be evaluated or on a GPU error.
  - Startup checks (generator.rs): states on 30 bits; extensions and default block without a fluid;
    the rule's fluid states go into `fluid_states`.
  - Speed (probe, one call at a time): 2.7 ms per chunk at the first try, **0.32 to 0.35 ms** after:
    - kernel split in three: `ksf_heads` (one warp per column: first free y by vote, scalars and
      extensions), `ksf_descent` (one warp per column: below, above and water by masks of 32 blocks,
      `ballot`/`popc`), `ksf_blocks` (one thread per block: the rule). Kernel 2.4 ms → 0.14 ms;
    - preliminary surface per region of 8 × 8 chunks (128 × 128 plane, host cache of 1,024 regions):
      159 → ~20 µs per chunk; Y-free subgraphs of `find_top_surface` computed before the loop;
    - buffers, streams and pinned memory reused (`SurfaceResources`).
    Left per chunk: veins ~86 µs (3 programs, 30 kernels), uploads and readback ~70 µs.
    `HORIZON_GPU_SURFACE_PROFILE=1` gives the details at shutdown.
  - Checks: `chunk_dump` probe byte-identical on 3 zones (land, frozen oceans, eroded badlands, 59
    chunks with structures); **vanilla parity at 0 differences on 785 chunks**, with 3,479 chunks
    surfaced by the GPU, 0 fallbacks.
  - `/pregen`, radius 1,600 blocks, fresh zones 230000/−250000 and −260000/270000 (script
    `/tmp/ab6.sh`, the gpus variant runs in the `gpub` copy):

    | Variant | Zone 1 | Zone 2 | CPU per chunk |
    |---|---|---|---|
    | CPU only | 684 chunks/s | 603 chunks/s | 16.3 and 16.7 ms |
    | GPU (density, aquifers, beard) | 943 (+38%) | 803 (+33%) | 11.3 and 12.0 ms |
    | **GPU + surface** | **1,067 (+56%)** | **906 (+50%)** | **9.6 and 10.3 ms (−41 and −38%)** |

    The surface on the GPU adds +13% and removes ~1.7 ms of CPU per chunk. 0 fallbacks over 42,849
    chunks per zone. Under load, a call takes 0.60 to 0.63 ms (0.35 ms alone); GPU at 24-25%, 2 to
    3.7 GB, 120-129 W.
- **S5, latency reduction (15/11/2026)**:
  - **Veins per region** of 8 × 8 chunks, over the `positive_y` y only (union of the veins, aligned on
    16; the `OreVein` rule returns "nothing" outside its bounds, without a draw, like the CPU) and kept
    on the GPU with the plane of the preliminary surface (`RegionSurface`, 48 regions,
    `HORIZON_GPU_SURFACE_REGIONS`). Only ~11 operations per chunk instead of about forty.
  - **Surface on the codes' stream**: `GpuFinalDensity::chunk_blocks_surface` launches the surface
    right after the codes kernel (`follow_up` hook of `copy_codes` and of the structures path); the
    codes no longer go back up to the GPU and there is only one wait. `SurfaceGpu::chunk_states`
    (comparison mode) keeps its own stream.
  - **Blocking wait** (`HORIZON_GPU_SYNC=block`): less CPU but calls of 0.87-0.92 ms instead of
    0.51-0.53 ms, lower throughput (969 / 999 against 1,001 / 1,019 chunks/s): active waiting is kept.
  - `/pregen` (`gpu` and `gpub` copies, scripts `/tmp/ab7.sh` and `/tmp/ab8.sh`, gain of the surface on
    the GPU over the GPU without surface, fresh zones at each pass):

    | Step | Zones | GPU without surface | GPU + surface | Gain |
    |---|---|---|---|---|
    | Veins per region | 270000/−290000, −300000/310000 | 863, 871 | 1,001, 1,019 | +16%, +17% |
    | + codes' stream | 310000/−330000, −340000/350000 | 979, 814 | **1,125, 954** | +15%, +17% |

    CPU per chunk with the surface: 9.2 to 9.6 ms (16-18% less than without). The codes' stream is
    neutral in the measurement, kept for the simplicity of the exchanges. Probe identical and vanilla
    parity at 0 differences after each step (3,479 chunks surfaced by the GPU, 0 fallbacks).
  - Rest of the CPU time: section writes, lighting, decorations, compression; the surface now only
    weighs the GPU call (~0.5-0.7 ms of active waiting under load).

## Design of S3 (done, kept for the record)

- Source: `Pipeline.rule` (uncompiled `MaterialRule`, references as `Arc`s), translated to CUDA by a
  generator (one function per referenced subtree, memoized by `Arc` address, to avoid a code
  explosion). `Block` → state; `Sequence` → first child that answers; `Condition`; `Bandlands`
  (`clay_bands_offset` noise and table of 192 states); `OreVein` (density and richness in **volume**
  mode, computed with the region like the final density; gap in **point** mode; randomness
  `root.from_hash_of("minecraft:ore")`; the FunctionIds of the roots are needed, through
  `MaterialRuleLibrary`).
- Conditions: biomes → bit mask on the id; `BiomeTag` → members; `Temperature` → table per biome
  (base, frozen); `YAbove` and `VerticalGradient`: anchors resolved on the CPU
  (`VerticalAnchor::resolve(generation)`); `NoiseThreshold` → noise `pipeline.noises[name]` through
  `get3`; `AbovePreliminarySurface` → `floor(surface_level(x, 0, z)) + depth - 8`.
- Kernel: one thread per column (256 per chunk), descent like `build_columns` (air, fluid, solid,
  `bottom`, `above`, `water`), write bounds, rule applied to the solid blocks; inputs: the chunk's
  fill codes (0/8 solid, 1 air, 2/3 fluid), biomes of the 3 x 3 chunks, columns (depth, secondary,
  slope, preliminary). To read first: `first_available` and `Column::kind` (material_stage.rs), the
  write bounds.
- Check: compare block by block with the blocks written by the CPU surface (`chunk_dump` probe,
  `HORIZON_PHASE=terrain`).
