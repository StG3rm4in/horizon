# Biomes on the GPU

## What it costs

`biome_chunk_or_busy` (horizon-world, generator.rs) builds the `BiomeChunk` of a chunk: 4 × 4 × 96
quarts in the overworld (1,536). For each one, the six climate fields (temperature, humidity,
continentalness, erosion, depth, weirdness), sampled as a volume on the CPU, then the search for the
closest biome in the climate tree (`ClimateIndex`). Profile of 15/11, everything else on the GPU:
**8.6% of the CPU**, of which 2.0% in `distance_until` and 0.7% in `search`.

## What the GPU does

`HORIZON_GPU_BIOMES=1`: for a multi-noise source, the GPU computes a region of R × R chunks
(`HORIZON_GPU_BIOMES_REGION`, 8 by default) at once, at the first chunk of the region asked for.

1. `kbi_climate`, one thread per column of quarts: the six climate roots of the router in point mode
   (`ClimateSampler::sample`: position = quart << 2), through a generated function
   (`Gen::column_points`, codegen.rs). As in the surface descent, the Y-free subgraphs
   (`domain_axes`) are computed once before the loop over the Y quarts, and shared between the six
   roots: five of the six fields, and the part of the depth that is not the Y gradient.
2. `kbi_search`, one thread per quart: the quantization `(v * 10000f) as i64` (saturating, NaN → 0),
   then the tree search **without a starting point**: depth first, children in order, a node is
   visited only if its distance is strictly below the best one. The kernel keeps the first leaf at
   the minimum, and that minimum.
3. `kbi_order`, one block per chunk: the CPU search starts from the chunk's previous result
   (`ClimateSearch::last`, a fresh resolver per chunk). It returns that previous result if its
   distance equals the minimum of the quart, otherwise the first leaf at the minimum, the one a
   search without a starting point finds: every leaf seen before it is farther, and its ancestors are
   at most at its distance. So the block applies `last` quart by quart, in fill order (sections from
   bottom to top, then x, y, z), by tiles of 128 quarts in shared memory, and writes the biome id.

The tree comes from `ClimateIndex::tree` (index.rs): per node, the six intervals and the squared
offset distance; and the links (first child and number of children; for a leaf: its entry and 0).
The distance is the CPU's, in modular 64-bit arithmetic; the early exit of `distance_until` does not
change any comparison.

On the CPU side (`gpu_biome_chunk`): one region per `OnceLock`, in a cache of the last 128 regions
(`GpuBiomeRegions`). The first worker that needs it computes it; the others wait, or return "busy"
when `biome_chunk_or_busy` is called without waiting. Each `BiomeChunk` is filled from the region, in
`fill` order.

Fallback to the CPU:

- a chunk that blends with an old world (non-empty blender);
- an entry without a biome id (the CPU raises the error);
- a failed region (traversal stack, search without a result, CUDA error);
- a source that is not multi-noise;
- legacy randomness (the Nether), as for the density.

## Variables

- `HORIZON_GPU_BIOMES=1`: biomes on the GPU.
- `HORIZON_GPU_BIOMES=test`: both paths, compared section by section; the CPU is authoritative.
  `GPU biomes test:` lines are written to latest.log at powers of two, with the first 16 mismatches.
- `HORIZON_GPU_BIOMES_REGION`: side of a region in chunks (8).
- `HORIZON_GPU_DUMP_BIOMES=file`: the generated CUDA source.

## Checks

- `chunk_dump` probe (seed 12345, full phase, everything on the GPU) on 3 zones of 256 chunks:
  byte-identical to the CPU, in `test` as in `1`; 484 chunks per zone compared in test mode,
  0 mismatches.
- Vanilla parity: overworld in test mode, 627 chunks, **0 differences**, 5,473 biome chunks compared,
  0 mismatches; full, everything on the GPU, **0 differences on 785 chunks**. The Nether (legacy
  randomness) stays on the CPU: 0 differences on 113 chunks.
- `/pregen` in test mode (radius 800, fresh zones 2600000/−2620000 then 2650000/−2670000): more than
  8,192 chunks compared, 0 mismatches.

## Measurements

GPU time of a region of 8 × 8 chunks (nsys, probe on 32 × 32 chunks):

| Version | Climate | Search | Order | Whole region |
|---|---|---|---|---|
| 1: climate and search per quart, order by one thread per chunk | 3.46 ms (with the search) | | 0.97 ms | ~5 ms |
| 2: climate per column, order by tiles | 0.11 ms | 0.33 ms | 0.23 ms | ~1.0 ms (max 1.4) |

The noise is in f64, which the 3080 computes slowly: in version 1, the climate evaluated at every
quart was the cost. With regions of 16 × 16, a region takes 2.0 ms: less per chunk, but a longer wait
for the first chunk asked for; 8 stays the default.

Benches (release, radius 1,600 blocks, fresh zones, everything else on the GPU,
sections included):

| Zone | CPU biomes | GPU biomes | CPU per chunk |
|---|---|---|---|
| 2700000/−2720000 | 1,669 chunks/s | 1,648 (−1.2%) | 5.43 → 5.29 ms (−2.6%) |
| −2730000/2740000 | 1,435 chunks/s | 1,485 (+3.5%) | 6.46 → 6.05 ms (−6.3%) |
| 2750000/−2770000 | 1,415 chunks/s | 1,393 (−1.5%) | 6.56 → 6.39 ms (−2.6%) |

The throughput does not move: `/pregen` is not limited by the CPU (8.4 to 9.2 cores busy out of
12). The CPU per chunk drops by 2.6 to 6.3% depending on the zone.

Profile (zone 2800000/−2820000, 1,714 chunks/s with the profiling binaries):

- `biome_chunk_or_busy` goes from 5.7% of the CPU to 0.1%, and the GPU path (`gpu_biome_chunk`,
  `BiomeGpu::region`) costs 0.18%;
- the "biomes and climate" category goes from 8.6% to 2.8%. The rest comes from calls outside the
  biome chunks: `fold` 1.4%, `carver_biome` 0.4%, `biome_id` 0.3%.
- The biggest items are now:

| Item | Share of the CPU |
|---|---|
| Decorations | 29.6% |
| Saving | 17.9% |
| GPU wait and driver | 11.2% |
| zlib | 9.9% |
| Main thread | 7.8% |
| Generation (rest) | 6.3% |
| Structures | 5.1% |
