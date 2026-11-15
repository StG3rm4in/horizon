# Light on the GPU (`HORIZON_GPU_LIGHT`)

## What it costs

`/pregen` profile of 15/11 (density, surface, ores and caves on the GPU, `/tmp/sous_lumiere.py`):
light accounts for **21% of the CPU**.

| Item | Share of the CPU |
|---|---|
| `light_chunk` (`Job::Light` task) | 16.3% |
| of which decoding the palettes of the 9 chunks (`unpack_ids_mapped`) | about 3.3% |
| of which copying the states into the region (`states_of`) | about 1.5% |
| of which the breadth-first fill of sky and blocks (`spread`) | about 7% |
| of which sky columns | about 0.8% |
| of which formatting the central chunk (`pack_lighting`, `Light::from_slice`) | 1.55% |
| Views of the neighbouring chunks (`light_view`, `block_column`) | 4.6% |

`light_chunk` lights a chunk over its 3 × 3 neighbours (48 × 48 × h): each chunk is decoded nine
times, once per light task of its neighbours.

## Why the GPU gives the same light

The final values of vanilla's engine (and of Horizon's fill) are the **fixed point** of the fill: for
each cell, the maximum over all paths from a source of `source − Σ max(1, attenuation)`, without
crossing an occluded face (`block_masks`). This fixed point is unique: a fill by parallel waves, in
any order, reaches it too. Integers have no rounding: there is nothing to reproduce but the rules.

## How it runs

1. **Light task** (`streaming.rs`, `Job::Light`): `light::gpu_light_submit` computes the height
   (`top_section`, like `light_chunk`) and copies, for the 9 chunks, the sections as they are stored:
   palette (translated to light infos by `LightTables::info`) and packed words. No cache on the GPU,
   so nothing to invalidate. The task sends the batch and returns right away.
2. **GPU** (`light.cu`, on the batch's stream):
   - `klt_states`: one block per (chunk, section), decodes the states and the emission, and stores
     the emitting cells in the seeds;
   - `klt_sky`: one column per thread, goes down to the first occluded edge
     (`packed_edge_occluded`) and writes the sky column;
   - `klt_fill`: one block of 1,024 threads, the sky then the blocks, by waves until no cell goes up.
     A cell goes up by CAS on the 32-bit word that holds its byte; the sky seeds are those of
     `light_chunk` (sources between the floor of their column and the deepest one of their
     neighbours);
   - `klt_output`: one block per section of the central chunk, which stores its light like
     `Light::from_slice` (uniform, sky only, block only or full, nibbles already packed).
3. **`horizon-gpu-light` thread** (one per dimension): waits for each batch on an event with blocking
   sync (the thread sleeps), builds the `Light`s and sends `Done::Lit` to the mailbox, as the task
   would have done.

A batch the GPU does not take (32 batches already in flight, a column of another height, GPU error)
is lit on the CPU as before. Each batch reserves 14 MB of GPU memory (24 sections).

## Variables

- `HORIZON_GPU_LIGHT=1`: light on the GPU.
- `HORIZON_GPU_LIGHT=test`: the thread compares each chunk with `light_chunk` (same kind of storage,
  same 4,096 values per section); on a mismatch, the CPU's light is kept and the mismatch counted.

A `GPU light:` line is logged (latest.log) at each power of two of chunks: chunks, bytes sent, upload
time (on the worker) and wait time (on the thread), batches in flight at most, errors, CPU fallbacks
and mismatches.

## Checks

- Test mode (`/tmp/lu1.sh`, server built in parity-target): zone 900000/−920000 (radius 400) then
  1100000/−1120000 (radius 1,200): **16,384 chunks lit by the GPU and compared with `light_chunk`,
  0 mismatches**, 0 errors. In test mode, the thread recomputes each chunk on the CPU: it becomes the
  bottleneck, the 32 batches are taken and part of the chunks are lit directly on the CPU
  (fallbacks), which does not change the result.
- Normal mode: 0 fallbacks, 21 batches in flight at most; upload 122 to 129 µs per chunk on the
  worker (about 180 KB copied), GPU wait 250 µs on the thread.

## Measurements

`/pregen` benches (`/tmp/ab11.sh`, release, radius 1,600 blocks, fresh zones; density, surface, ores
and caves on the GPU in both variants):

| Zone | Without GPU light | With | CPU per chunk |
|---|---|---|---|
| 1000000/−1020000 | 1,117 chunks/s | **1,333 chunks/s (+19.3%)** | 9.09 → 6.95 ms (−23.5%) |
| −1030000/1040000 | 1,124 chunks/s | **1,303 chunks/s (+16.0%)** | 8.24 → 6.33 ms (−23.2%) |
| 1050000/−1070000 | 1,173 chunks/s | **1,405 chunks/s (+19.8%)** | 8.84 → 6.74 ms (−23.8%) |

GPU utilization goes from 28-42% to 63-68%, and its memory from 4.9 to 5.3 GB. The server now only
keeps 8.2 to 9.3 cores busy out of 12 (10 before): the throughput is no longer limited only by the
workers' CPU.

Profile with light on the GPU (zone 1150000/−1170000, 1,337 chunks/s with the profiling binaries):
light drops from 16.6% to **0.9% of the CPU**. The biggest items are now the decorations (23.3%),
GPU wait and driver (12.7%, against 8.0%, of which 8.3% in libcuda: the synchronous calls of the
density and the surface wait longer for a busier GPU), saving (12.5%), writing the final states
(9.1%), the main thread (8.8%), the biomes (7.6%) and zlib compression (7.5%).

Light unit tests of horizon-client-render (`cargo test -p horizon-client-render --lib light`): 89
pass.


## Section cache on the GPU (15/11, evening)

### Why

A light batch reads the 9 chunks of the 3 × 3 neighbourhood. The upload copied the words and palettes
of all their non-uniform sections: about 180 KB per chunk, and each chunk copied 9 times. The lz4
profile puts this upload at **4.1% of the CPU** (copy, palette conversion, driver; GPU_WAIT.md).

### How

`LightGpu::submit_sections` (light.rs) receives the sections of a batch section by section
(`LightInput`: absent, uniform, or packed with its `Arc<Section>`); `gpu_light_submit` (light.rs)
describes them like the previous fill.

- **Identity**: the key of a section is the address of its `Arc` (`Arc::as_ptr`). The slot keeps a
  `Weak` to it. While it lives, no other allocation takes that address, and `Arc::make_mut` moves the
  section instead of changing it in place: an `Arc` still at that address therefore has the same
  content. All section writes go through `Arc::make_mut`. A strong `Arc` would have forced each light
  write to clone the section; with the `Weak`, it only moves it (a few hundred bytes).
- **Slots**: 16,384 of 1,152 words on the GPU (151 MB), which hold the palette infos then the words.
  A global palette of more than 1,024 words stays inline.
- **Descriptors**: kinds 4 (palette) and 5 (global) of `klt_states` read from the cache (`CACHE`)
  instead of the input.
- **Fill**: a section missing from the cache is sent inline in the input, as before, then `klt_cache`
  copies it into its slot, on the batch's stream. The resource's `fill_done` event is recorded right
  after.
- **Slot states**:
  - *reserved*: the fill is not enqueued yet, and the other batches send the section inline;
  - *filling*: the event is recorded, and the other batches wait for it (`stream.wait`). If the
    resource records the same event later, it is further along the same stream, so still after this
    fill;
  - *ready*: the batch that filled it is finished.
- **Pinning and eviction**: a batch pins the slots it reads or fills, until `finish`. The eviction
  (second-chance clock) only takes a ready, unpinned slot. If a batch's stream fails, its fills are
  forgotten.
- **Fallback**: without a free slot, or for a section another batch is filling, the section goes
  inline, as before.

### Variables

- `HORIZON_GPU_LIGHT_CACHE`: `0` without a cache, a number of slots, `test`. 16,384 by default.
- `HORIZON_GPU_LIGHT_CACHE=test`: each section found in the cache is also sent inline, and
  `klt_states` compares both cell by cell (flag `LT_E_CACHE`, counted in the summary). Combines with
  `HORIZON_GPU_LIGHT=test`, which compares the chunk's light with the CPU's.

### Checks

- The 89 light unit tests pass.
- `/pregen` of radius 800, zone 4400000/−4420000, cache in test mode and light in test mode:
  **0 cache mismatches, 0 mismatches with the CPU** (53% of the sections found).
- `/pregen` of radius 800, zone 4450000/−4470000, real cache and light in test mode: **0 mismatches
  with the CPU**.
- The batches "fallen back to the CPU" in these runs come from the test mode: the completion thread
  recomputes each chunk on the CPU, falls behind, and the 32 in-flight batches are taken.

### Measurements

Bench `/tmp/ab25.sh` (release, everything on the GPU, lz4; same binary, `HORIZON_GPU_LIGHT_CACHE=0` in
the gpub copy against the cache in the gpu copy; fresh zones):

| Zone | Throughput without → with | CPU per chunk |
|---|---|---|
| 4500000/−4520000 | 1,928 → 1,999 chunks/s (+3.7%) | 4.33 → 4.20 ms (−3.0%) |
| −4530000/4540000 | 1,956 → 2,055 (+5.1%) | 4.32 → 4.16 ms (−3.7%) |
| 4550000/−4570000 | 2,001 → 2,070 (+3.5%) | 4.19 → 4.01 ms (−4.3%) |

**−3.7% of CPU per chunk, +4.1% of throughput.**

| Measure | Without cache | With cache |
|---|---|---|
| Sections found in the cache (radius 1,600) | — | 73% |
| Sent per chunk | 180 KB | 48 KB |
| Upload | 124 µs per chunk | 75 µs per chunk |
| GPU memory | — | +224 MB |
