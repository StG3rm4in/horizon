# Palettes of finished chunks

## What it costs

`block_column` (horizon-client, vanillagen.rs) converts the generation sections (`BlockSection`) into
chunk sections (horizon-core's `PalettedContainer`). It serves the views of the neighbouring chunks
for light and the final column of a chunk. Profile of 15/11, everything on the GPU: **5.7% of the
CPU**, of which 3.9% in `PalettedContainer::from_values` and 1.1% in `BlockSection::fill_states`.

The path went through the values: unpacking the 4,096 states of the section, then rebuilding a
palette in order of first appearance by index (`palette_ids`) and repacking all the ids.

## The direct path

`TerrainChunk::section_container` → `BlockSection::to_container`:

1. the live entries of the generation palette, sorted by their first cell (two entries of the same
   state merge): that is the order of first appearance `from_values` finds;
2. each id renumbered to this palette, directly in the words (`PalettedContainer::from_mapped`). From
   4 to 4 bits, the common case, a 256-byte table translates the two entries of a byte at once;
   otherwise entry by entry. The global palette (more than 256 states) receives the values, as in
   `from_values`.

For 4-bit words, the search for the first cells reads one word at a time: the mask of the ids it
holds, then the position only for an id seen for the first time.

The result is the same container, word for word (`to_wire`: width, palette, words), not only the same
values.

## Variables

- `HORIZON_COLUMN=values`: the old path (bench).
- `HORIZON_COLUMN_TEST=1`: both paths, compared by `to_wire`; a `column test:` line in latest.log at
  powers of two.

## Checks

- Unit test `to_container_matches_from_values` (horizon-world, chunk.rs): random sections of 1 to 300
  states, with dead and reused entries, up to the global palette.
- `/pregen` in test mode (zone 2100000/−2120000, radius 800, everything on the GPU): **262,144
  sections, 0 differences**.
- The probe and the vanilla parity harness do not go through `block_column`.

## Measurements

First version (reading the cells one by one for the first appearances), benches `/tmp/ab17.sh`
(release, radius 1,600 blocks, fresh zones, everything on the GPU, sections included):

| Zone | Through the values | Direct | CPU per chunk |
|---|---|---|---|
| 2200000/−2220000 | 1,432 chunks/s | 1,410 (−1.6%) | 6.59 → 6.45 ms (−2.1%) |
| −2230000/2240000 | 1,527 chunks/s | 1,530 (+0.2%) | 6.10 → 5.95 ms (−2.5%) |
| 2250000/−2270000 | 1,590 chunks/s | 1,697 (+6.7%) | 5.80 → 5.37 ms (−7.4%) |

Profile of this version (zone 2300000/−2320000, 1,593 chunks/s with the profiling binaries):
`block_column` drops to 3.4% of the CPU, of which 3.1% in `to_container`. `fill_noise_states` is at
0% (GPU sections). The biggest items are now the decorations (31.5%), saving (15.7%), zlib (8.9%),
GPU wait and driver (8.7%), the biomes (8.6%) and the main thread (7.2%).

Version with the search for first appearances by word (same checks: unit test, `/pregen` in test
mode on zone 2400000/−2420000: 262,144 sections, 0 differences):

| Zone | Through the values | Direct | CPU per chunk |
|---|---|---|---|
| 2500000/−2520000 | 1,544 chunks/s | **1,584 (+2.6%)** | 6.06 → 5.80 ms (−4.3%) |
| −2530000/2540000 | 1,524 chunks/s | **1,548 (+1.5%)** | 6.29 → 6.05 ms (−3.8%) |
| 2550000/−2570000 | 1,422 chunks/s | **1,521 (+7.0%)** | 6.47 → 6.16 ms (−4.8%) |
