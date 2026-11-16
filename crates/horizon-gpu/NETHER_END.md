# The Nether and the End on the GPU

Until 16/11 only the overworld ran on the GPU: `GpuFinalDensity::new` refused every dimension with
a legacy random source, and the Nether and the End have one. Everything they generate went through
the CPU.

## What kept them on the CPU

- **The blended noise.** The GPU code drew `BlendedNoise`'s three noise stacks itself, from a
  Xoroshiro factory; with a legacy source the CPU draws them from `LegacyRandom::new(seed)`. All
  the other noises come from the compiler's own instances, so they were already right.
  `DensityCompiler::blended_fbm_set` (horizon-core) now gives the stacks as the compiler draws them.
- **No aquifers.** The fill, the surface and the compact sections on the GPU all start from the
  block codes of the GPU aquifers, and neither dimension has aquifers (`DisabledAquifer`: the
  global fluid picker, lava below y = 32 in the Nether).
- **Legacy positional randomness.** The surface draws from positional factories (bedrock floor and
  roof gradients, surface depth); the GPU only knew Xoroshiro ones.
- **The End's density** has three nodes the CUDA translator did not know: `EndIslands` (the outer
  islands), `DistanceToPoint` (the central island) and a `Slice` in a volume.

## What changed

- **Codes without aquifers.** `AquiferSetup.roots` is an `Option`. Without roots, the aquifer
  module gets stub fields and every chunk's skip height is `i32::MIN`: `aq_substance` then returns
  the global picker's substance without reading any aquifer cell, exactly `DisabledAquifer`. The
  generator fills from these codes (`Pipeline::gpu_codes`), and accepts them when the CPU aquifer
  has no air limit (`air_above_y() == None`). Structure chunks (beardifier) and the substance of
  carved cells use the same kernels.
- **Legacy positional draws.** `horizon_gpu::Positional` is a Xoroshiro state or a legacy seed
  (`PositionalRandom::legacy_seed`, horizon-core); `lg_at`, `lg_next_float` and `lg_next_double`
  (random.cu) are `LegacyPositionalRandomFactory::at` and `java.util.Random`. The surface refuses a
  legacy root with ore veins or extension biomes (badlands pillars, icebergs), the only other users
  of its draws; neither dimension has them.
- **Biome height.** The Nether's noise covers 128 of its 256 blocks; the surface used the noise's
  height for the biome quarts too. `SurfaceSetup` now has the level's `biome_min_y` and
  `biome_height` (`SF_BMINY`, `SF_BH`).
- **The End's nodes.**
  - `end_islands` (in the CUDA operations) is `EndIslandSampler::density`: 25 × 25 candidate
    islands around the column, a 2D simplex noise each (moved from surface.cu to base.cu), float
    and double operations in the CPU's order. `DensityCompiler::end_island_sampler` gives the
    sampler, so the GPU reads the same noise table.
  - `DistanceToPoint` with its four metrics (`dist_*`).
  - `Slice` in a volume: the input itself when it does not read the sliced axis; otherwise the
    input in point mode on the plane, when nothing in it evaluates differently in a volume
    (constants, unary operations, clamps, gradients, distances: `Gen::pointwise`).
- **GPU biomes** no longer exclude legacy dimensions: the Nether's multi-noise source evaluates its
  climate roots like the overworld's.

Not on the GPU yet: the End's biomes (its biome source is not multi-noise; it evaluates the islands
once per chunk, cheaply) and the Nether's compact sections (they need a noise as high as the chunk;
the CPU writes the GPU's final states instead).

## Checks

- probe byte-identical to the CPU in the Nether (2 zones) and the End (3 zones, the central island
  included), with everything on the GPU; GPU biomes in test mode in the Nether: 968 chunks,
  0 mismatches; the GPU surface of the Nether: 800 chunks, 0 fallbacks;
- probe byte-identical on the 4 overworld zones, all on the GPU and in test modes;
- vanilla parity with everything on the GPU: **0 differences on 785 chunks**, three dimensions;
- 507 horizon-world tests and 162 horizon-core tests pass.

## Measurements

Bench `/tmp/ab26.sh` with `DIM` (`/pregen` through `execute in`), release, radius 1,600 blocks;
binaries from before in `target/avant-nether/` (the overworld fully on the GPU, the other
dimensions on the CPU); the copies alternate:

| Dimension, zone | CPU per chunk before → after | Throughput before → after |
|---|---|---|
| Nether 300000/300000 | 5.94 → 2.77 ms (−53%) | 1,811 → 3,310 chunks/s (+83%) |
| Nether −330000/340000 | 5.83 → 2.59 ms (−56%) | 1,844 → 3,394 (+84%) |
| End 400000/−400000 | 1.13 → 0.78 ms (−31%) | 7,562 → 7,351 |
| End −430000/440000 | 1.13 → 0.75 ms (−34%) | 7,336 → 7,067 |
| Overworld 5100000/−5120000 | 3.20 → 3.20 ms | 2,455 → 2,462 |
| Overworld −5130000/5140000 | 3.04 → 3.06 ms | 2,499 → 2,499 |

The Nether nearly doubles. The End's chunks are mostly void and already cheap: their CPU drops by
a third, but its `/pregen` is not limited by the CPU (5 to 7 cores busy). The overworld does not
move: the fixed-shape cache extended to the snow layer, sea pickle and decoration collision queries
(`BlockCollisionRegistry::fixed_facts`) weighs too little there to show.
