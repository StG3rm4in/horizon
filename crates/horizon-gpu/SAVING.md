# Saving chunks

The throughput of `/pregen` is limited by the 6 physical cores (THROUGHPUT.md). In the profile of
15/11 with everything on the GPU (biomes included), saving takes **17.9% of the CPU** (NBT encoding,
region writing) and zlib compression **9.9%**. Both run in the 6 threads of the compression pool
(`save/compression_pool.rs`): `ChunkRecord::from_nbt` encodes the NBT, then `Compression::encode`
compresses it.

## Compression

Horizon writes the regions like vanilla:
- `region-file-compression` in server.properties picks `deflate` (default), `lz4`, `gzip` or `none`
  (`Compression::configure`, like `RegionFileVersion.configure`);
- deflate at `Deflater`'s default level, 6.

`HORIZON_ZLIB_LEVEL` (0 to 9, read once, `compression.rs`) changes this level for the benches. Any
reader reads a deflate stream back, whatever its level.

Bench (release, everything on the GPU, radius 1,600 blocks,
40,401 chunks per zone, fresh zones). Each variant goes once through each copy (Latin square). The
size is that of the zone's `region/`, `entities/` and `poi/` files:

| Zone | deflate 6 (default) | deflate 1 | lz4 |
|---|---|---|---|
| 3500000/−3520000 | 1,645 chunks/s, 5.46 ms CPU, 386 MiB | +4.3%, −6.0% CPU, +27.7% disk | +7.2%, −9.3% CPU, +46.6% disk |
| −3530000/3540000 | 1,672 chunks/s, 5.16 ms, 370 MiB | +11.0%, −8.3%, +26.8% | +11.8%, −9.5%, +44.8% |
| 3550000/−3570000 | 1,748 chunks/s, 4.94 ms, 358 MiB | +8.2%, −6.9%, +25.3% | +10.0%, −9.9%, +43.8% |

On average:

| Variant | Throughput | CPU per chunk | Disk per chunk |
|---|---|---|---|
| deflate 6 | — | — | 9.4 KiB |
| deflate 1 | +7.8% | −7.1% | 11.9 KiB (+27%) |
| lz4 | +9.7% | −9.6% | 13.7 KiB (+45%) |

lz4 removes almost all of zlib's cost. Its regions can only be read by vanilla 1.20.5 and later, and
by the tools that know this format (type 4, the lz4-java block). Deflate 1 keeps the usual format.

For a pregeneration of a 10,000-block radius (1.56 million chunks): 14.0 GiB in deflate 6, 17.7 GiB
in deflate 1, 20.4 GiB in lz4.

### The deflate level option

`[world] region_deflate_level` in server.toml (0 to 9, 6 by default like vanilla):

1. The supervisor writes it into server.properties as `horizon-region-deflate-level`
   (`config::TYPED`, bounds in `settings.rs`).
2. The engine passes it to `Compression::configure_deflate_level` at startup (`dedicated.rs`, right
   after `region-file-compression`). An invalid value is reported in the log, and 6 stays.
3. `HORIZON_ZLIB_LEVEL` is still used by the benches when nothing is configured.

The level only matters for `deflate` and `gzip`.

Checked by:
- a compression unit test (reading back at levels 1 and 9, refusing 10);
- the horizon-server tests: reference `docs/api/server-reference.md` regenerated (`HORIZON_WRITE_DOCS=1
  cargo test -p horizon-server reference`), API golden updated (`testdata/api-v1-config.json`),
  `defaults_are_vanilla`;
- a real server: same zone (radius 400), 26.1 MiB at level 6, 32.7 MiB at level 1 (+25%).

Since 15/11, the bench copies of the test world (`gpu`, `gpub`, `cpu`) write in **lz4**
(`region-file-compression = "lz4"` in the `[properties]` of their server.toml). The measurements that
follow therefore no longer compare directly with the previous ones, made in deflate 6.
The first version of this bench changed server.properties directly, which no longer has any
effect: the server rewrites it from server.toml.

## Encoding

Profile (zone 3600000/−3620000, deflate 6): saving weighs
27% of the CPU in total, compression included.

- `encode_saved` (the chunk as an NBT tree): 12.9%, of which
  - `PalettedContainer::pack` 6.1% (`palette_ids` 2.7%, `pack_words` 1.9%, `unpack` 1.1%);
  - strings 2.7% (`JavaString` is a `Vec<u16>`: each string is transcoded to UTF-16 and allocated);
  - scheduled ticks 2.3% (a `BTreeMap` and seven strings per tick);
  - compounds 2.3%.
- `nbt::write_named` (the tree as bytes): 2.2%.
- Compression: 9.7%.

### `pack` without repacking

`PalettedContainer::pack` (paletted.rs, vanilla's `PalettedContainer.pack`) unpacks the 4,096 cells,
rebuilds the palette in order of first appearance, then repacks the ids. Only uniform sections had a
shortcut.

But a container that `from_values` or `from_mapped` has just built is already in that form:
- its palette lists each value in order of first appearance, without a dead entry;
- `build` encoded the ids at the width `pack` picks, with the same `pack_words`.

That is the case of the chunk sections `block_column` builds (`BlockSection::to_container`). The
container's `packed` flag records it. `set`, the only write, clears it. `pack` then returns the
palette and the words as they are. The full computation stays in `pack_full`.

Checks:

- unit test `packed_containers_pack_as_in_full`: blocks and biomes, from 2 to 300 values, through
  `from_values`, `from_mapped` and after a write;
- `HORIZON_PACK_TEST=1` computes each `pack` both ways and compares. `/pregen` in test mode
  (radius 800, zone 3650000/−3670000): **more than 131,072 containers, 0 mismatches**;
- the bench regions, compared between the old and the new binary (`tools/gpu_worldgen.py compare-worlds`, and a
  comparison listing the paths of the tags that differ): 15 chunks out of 40,401 with
  different blocks. That is the known noise from one run to the next: fluids, plant age, berries,
  vines, ticks that run during `/pregen`. The whole NBT differs for ~5% of the chunks, but through
  fields that already change from one run to the next:
  - the order of the `block_entities` list;
  - the loot seeds;
  - the order of the `fluid_ticks`.

  No section with identical blocks is written differently.

Bench (release, everything on the GPU, deflate 6; previous binaries in
`target/avant-pack/`, gpub copy; the new ones in the gpu copy; fresh zones):

| Zone | Throughput before → after | CPU per chunk |
|---|---|---|
| 3700000/−3720000 | 1,704 → 1,795 chunks/s (+5.3%) | 5.14 → 4.85 ms (−5.6%) |
| −3730000/3740000 | 1,405 → 1,516 (+7.9%) | 6.41 → 6.03 ms (−5.9%) |
| 3750000/−3770000 | 1,676 → 1,745 (+4.1%) | 5.23 → 4.93 ms (−5.7%) |

**−5.7% of CPU per chunk, +5.8% of throughput.**

### Writing the NBT directly

Profile in lz4 (zone 3850000/−3870000): building the tree (`encode_saved`) cost 9.1% of the CPU, and
writing it (`write_named`) 2.7%. `JavaString` is a `Vec<u16>`: each string is transcoded to UTF-16
and allocated, then transcoded again when written. Each scheduled tick became a `BTreeMap` and seven
strings.

Now `encode_saved_bytes` (save.rs) writes the bytes `write_named(encode_saved(...))` would write,
without the tree of the big parts:

- `encode_root` is `encode_with`, but with the sections and the ticks left as empty lists. The rest
  of the root stays as tags: block entities, structures, heightmaps, keys of a loaded chunk.
- `nbt::RootWriter` (horizon-world nbt.rs) writes the root entry by entry with the same `Writer`.
  So modified UTF-8, the too-long string written empty, the canonical NaN and the byte limit are all
  kept. It only exists under limits without a node cap (`Limits::saved`), since what it writes
  directly is not counted.
- The entries of the root are walked in `BTreeMap` order. `sections`, `block_ticks` and `fluid_ticks`
  are written directly (`write_sections`, `write_ticks`), in the order in which their compounds would
  sort their keys:
  - section: `BlockLight`, `SkyLight`, `Y`, `biomes`, `block_states`;
  - `biomes` and `block_states`: `data`, `palette`;
  - tick: `i`, `p`, `t`, `x`, `y`, `z`.
- The state palettes (`write_state_palette`) write the name of a default state, or the compound of
  another one. This compound is serialized once by the current writer (`nbt::payload_bytes`) and
  cached per thread. A palette that mixes strings and compounds becomes, as in `Writer::payload`, a
  list of compounds where the strings are wrapped in `{"": …}`.
- The compression pool calls `Record::encoded`, then `ChunkRecord::from_bytes`. The other records
  (protos, POI, tags) go through the tree as before.
- `HORIZON_NBT=tree` brings the old path back. `HORIZON_NBT_TEST=1` writes both, compares the bytes,
  keeps the tree's on a mismatch, and writes `NBT test:` lines to latest.log.

Checks:

- unit test `direct_chunk_bytes_match_the_tree` (horizon-client-lib, with `HORIZON_TEST_RESOURCES`):
  states drawn from the whole registry, so mixed palettes, with and without light, with and without
  tables; the 240 save tests pass;
- `/pregen` in test mode (radius 800, zone 4000000/−4020000): **more than 8,192 chunks, 0 byte
  mismatches**.

Bench (release, everything on the GPU, lz4; same binary, `HORIZON_NBT=tree` in the gpub
copy against direct writing in the gpu copy; fresh zones):

| Zone | Throughput tree → direct | CPU per chunk |
|---|---|---|
| 4050000/−4070000 | 1,852 → 2,004 chunks/s (+8.2%) | 4.70 → 4.20 ms (−10.6%) |
| −4080000/4090000 | 2,029 → 2,093 (+3.1%) | 4.07 → 3.82 ms (−6.1%) |
| 4100000/−4120000 | 1,844 → 1,956 (+6.1%) | 4.77 → 4.38 ms (−8.2%) |

**−8.3% of CPU per chunk, +5.8% of throughput** on average.

### What is left

- The heightmaps are recomputed from the blocks at each save. Since the evening of 15/11, this is
  done section by section (`Heightmap::prime_sections`, see GPU_WAIT.md): 1.51% → 0.93% of the CPU.
- The block entities and the rest of the root still go through the tree; that is little per chunk.
- The protos (edge of `/pregen`) are still written through the tree.
