# Compact sections on the GPU (`HORIZON_GPU_SECTIONS`)

## What it costs

`/pregen` profile of 15/11, everything on the GPU (`/tmp/sous_etats.py`): `fill_noise_states`, which
writes the GPU's final states (fill then surface) into the `TerrainChunk`, weighs **9.1% of the
CPU**: `fill_solid_run` 4.9%, the loop itself 2.7% (of which 0.85% for rereading the 98,304 states
looking for bit 30), `set_block_raw` 0.7%, the heightmaps 0.5%. For this, the GPU sent back the
98,304 states as u32 (393 KB per chunk).

## Why the GPU can return the exact section

A generation section (`BlockSection`) keeps a dynamic palette: order of arrival of the states, a
counter per entry, free entries, a word width that only grows. Its proto chunk save
(`BlockSection::to_nbt`) writes this palette as it is, free entries included: to stay byte-identical,
the GPU must return exactly what `set` would have left.

`fill_noise_states` writes each non-air cell only once, into a fresh section (uniform air), column
after column (z then x), from top to bottom; `fill_run` is equivalent to `set`s in the same order.
So the result only depends on this sequence of writes:

- if some air is left: palette [air, states in the order of their first write], their counters, no
  free entry;
- no air left and a single state: uniform section;
- no air left, and the state of the last write was already in the palette: air keeps its entry, at
  0, which is free;
- no air left, and the state of the last write is new: it takes the air's entry;
- the word width is `bits_for_palette(length, 4)` of the final palette.

The non-air count follows both write paths: +1 per cell of a run (`fill_solid_run`), +1 if the state
is not air for a cell written alone (fluid or marked). The `OCEAN_FLOOR_WG` and `WORLD_SURFACE_WG`
maps are 1 + the highest opaque cell. The post-processing marks keep their order: bit 31 of the
written cells, columns by z then x, y going down; then bit 30 of all the cells, columns by x then z,
y going down.

## How it runs

1. After `ksf_blocks` (the final states on the GPU), on the same stream (`SurfaceGpu::enqueue`):
   - `ksec_heights`: one block per column, the two maps and the number of marks;
   - `ksec_marks`: the ranks of the columns, then the marks of only the columns that have some;
   - `ksec_sections`: one block per section; a table of 128 states (first write position, counter),
     their ranks, the rules of the last write, the packed words.
2. The GPU sends back 1,285 u32 words per section, the heights and the marks (about 125 KB per chunk
   instead of 393 KB).
3. `fill_noise_sections` (terrain.rs) places each section through `BlockSection::from_parts`, its
   non-air counters, the maps (`Heightmap::set_first_available`) and the marks.

More than 128 states in a section or more than 2,048 marks of one kind: the raw states come back and
`fill_noise_states` writes them as before. Only a fresh chunk of the whole height takes this path
(`TerrainChunk::is_fresh`).

## Variables

- `HORIZON_GPU_SECTIONS=1`: compact sections from the GPU (with `HORIZON_GPU_SURFACE=1`).
- `HORIZON_GPU_SECTIONS=test`: the raw states come back too; `fill_noise_states` writes them, then
  `check_noise_sections` compares each section (palette, counters, free entries, words), the non-air
  counters, the maps and the marks.

The `GPU surface:` line gives the chunks placed, the fallbacks through the raw states and the
mismatches.

## Checks

- `chunk_dump` probe (`/tmp/sec1.sh`), 3 zones of 256 chunks (0,0; −1250,940; eroded badlands
  −320,−295): byte-identical to the CPU in test mode and with the sections; 0 mismatches and
  0 fallbacks over 1,200 chunks in test mode.
- Vanilla parity: **0 differences on 785 chunks** with the sections; overworld in test mode:
  3,479 chunks, 0 mismatches, 0 fallbacks.
- horizon-world unit tests (chunk, heightmap, terrain): 34 pass.

## Measurements

The first version (a single block for the maps and the marks, uncoalesced reads) made the surface
call 140 µs longer (537 → 676 µs, probe without contention). With one block per column for the maps,
the marks reread only in the columns that have some, and coalesced reads in `ksec_sections`, the
overhead drops to 48 µs (498 → 546 µs).

`/pregen` benches (`/tmp/ab16.sh`, release, radius 1,600 blocks, fresh zones, everything on the GPU):

| Zone | Without sections | With | CPU per chunk |
|---|---|---|---|
| 2000000/−2020000 | 1,574 chunks/s | **1,690 (+7.3%)** | 5.96 → 5.31 ms (−10.9%) |
| −2030000/2040000 | 1,524 chunks/s | **1,665 (+9.3%)** | 6.34 → 5.60 ms (−11.7%) |
| 2050000/−2070000 | 1,303 chunks/s | **1,387 (+6.5%)** | 7.48 → 6.93 ms (−7.4%) |

(First version: +3.0 / +5.2 / +6.1%; CPU −6.8 / −11.4 / −9.2%.)
