# Geode classification

`HORIZON_GPU_GEODES=1` enables CUDA classification for the bundled
`minecraft:amethyst_geode` placed feature. `test` compares every output class
against the scalar CPU classifier before replay. The CPU owns admission,
provider calls, world updates, ticks and crystal placement.

A worker combines independent admitted requests in batches of up to eight.
The request carries the exact two CPU Perlin tables and the prepared shape.
The kernel reads them with the shared scalar Perlin sampler and returns one
byte per cell in X-fastest order. `HORIZON_GPU_GEODES_BATCH=1` selects the
singleton launch path for latency measurements.

At shutdown the stage reports batch size coverage, CPU fallbacks, compared
cells, mismatches, and mean batch and request times. A CUDA batch error
returns affected requests to the CPU and disables future geode launches
process-wide. The startup density self-test does not cover geodes.
