# Checking and measuring the GPU world generation

The other notes of this folder give results stage by stage. This one says how to check them and measure
them again: the tool, the machine they come from, and what each number means.

## The tool

`tools/gpu_worldgen.py` (run it with `--help`):

- `probe` dumps the chunks of nine zones of seed 12345 (four in the overworld, two in the Nether, three in
  the End) with `world_probe chunk_dump` and compares them with references: by default the same build on
  the CPU, enough for a change that leaves the CPU path alone; `--refs DIR` takes dumps of another build,
  and `--refs-probe` makes them (for example with a probe built from pure upstream). The overworld runs
  with everything on the GPU and in the GPU test modes, which compare every chunk with the CPU inside the
  probe; the Nether and the End with everything on the GPU. A GPU run that shows no GPU work fails: a
  stage that fell back to the CPU would compare equal and prove nothing.
- `verify` runs rustfmt and clippy `-D warnings` (`--lint`: the upstream CI's two commands, and
  horizon-gpu without CUDA), then `probe`, the vanilla parity of `tools/test_chunk_generation_parity.py`
  with everything on the GPU (785 chunks, 27 biomes, 74 structure sites), and the unit tests given with
  `--tests`. It ends on a verdict and can write a Markdown report (`--report`).
- `pregen` measures `/pregen` on fresh servers: every pass creates a server, pregenerates the same zones
  (radius 1,600 blocks, 40,401 chunks per zone by default) and stops it. It compares builds (`--bin`) and
  settings (`--variants`: `cpu`, `gpu` or any set of variables), per dimension, alternating their order on
  every repeat, and reports chunks a second, CPU time per chunk, cores busy, GPU use and GPU memory.
- `compare-worlds` compares the blocks and biomes two saved worlds hold, chunk by chunk (`pregen --keep`
  keeps the worlds of a run).

The load with players goes through upstream's `tools/server_bench.py`, with the GPU variables set in its
environment (`HORIZON_GPU_DENSITY=1` and the others below); `--bots-via` runs the bots on another machine,
so that they do not take the server's CPU.

## The GPU variables

| Variable | What it does |
|---|---|
| `HORIZON_GPU_DENSITY` | final_density and the aquifers (`1`; `wait` compiles before the first chunk) |
| `HORIZON_GPU_SURFACE` | the surface rules (`HORIZON_GPU_SURFACE_TEST=1`: compared with the CPU) |
| `HORIZON_GPU_ORES`, `HORIZON_GPU_CAVES`, `HORIZON_GPU_SECTIONS`, `HORIZON_GPU_BIOMES` | ores, carvers, section palettes, multi-noise biomes (`1`, or `test`) |
| `HORIZON_GPU_LIGHT` | the light of generated chunks (`1`, or `test`) |
| `HORIZON_GPU_DEVICE` | the CUDA device (0) |
| `HORIZON_GPU_MEMORY_MB` | GPU memory of the region cache (half of the free memory by default) |

The build needs no CUDA toolkit; the `cuda` feature (on by default) compiles the backend, and the
libraries are loaded at run time. See the doc of `crates/horizon-gpu/src/lib.rs`.

## The machine of the published numbers

| | |
|---|---|
| CPU | Intel Core i5-10600K, 6 cores / 12 threads, 4.8 GHz all cores |
| Memory | 2 × 16 GB DDR4, 3,600 MT/s |
| GPU | NVIDIA GeForce RTX 3080, 10 GB, driver 615.71, CUDA 13.4 libraries |
| System | Ubuntu 26.04.1, Linux 7.0 |
| Toolchain | rustc 1.100.0; release builds with frame pointers and line tables (thin LTO, one codegen unit) |

The memory and core settings changed over the measurements (overclocking from 16/11), so numbers compare
within one note's tables, measured the same day, not across notes. Numbers measured before 17/11 come from
one-off scripts that were not kept; `pregen` reproduces their protocol (fresh zones, radius 1,600 blocks,
alternated order) on fresh servers.

## CPU only against CPU and GPU, same commit

Measured on 17/11 with `pregen --variants cpu,gpu --dims overworld,the_nether,the_end --repeat 2`: the
bench build of the branch, two zones of 40,401 chunks per dimension, each pass on a fresh server with its
default settings (deflate regions, as many workers as the machine has threads). Both columns have the same
CPU optimizations (decorations, palettes, direct NBT writing, scheduling); only the GPU stages differ.

| Dimension | CPU only | CPU and GPU | Throughput | CPU per chunk |
|---|---|---|---|---|
| Overworld | 853 chunks/s, 12.92 ms | 2,287 chunks/s, 3.37 ms | ×2.68 | −74 % |
| Nether | 1,340 chunks/s, 8.35 ms | 3,183 chunks/s, 2.72 ms | ×2.38 | −67 % |
| End | 3,586 chunks/s, 3.08 ms | 6,879 chunks/s, 0.79 ms | ×1.92 | −74 % |

The two repeats differ by about 1 %. With the GPU, the cores busy go from about 11 to 7.6 (overworld), 8.5
(Nether) and 5.1 (End); the GPU is busy 56 to 76 % of the time and holds 5.5 to 6.3 GiB in the overworld,
2.2 to 2.6 GiB in the Nether and the End.

## Reproducibility of /pregen

The same run kept its worlds (`--keep`), and `compare-worlds` compared them zone by zone. Chunks whose
blocks differ, out of 40,401 per zone; the biomes are the same everywhere:

| Worlds compared | Overworld, zone 1 | Overworld, zone 2 | Nether | End |
|---|---|---|---|---|
| GPU, two passes | 8 | 16 | not compared | not compared |
| CPU, two passes | 9 | 8 | not compared | not compared |
| CPU against GPU | 11 | 12 | 0 | 0 |

Two passes with the same settings differ as much as the CPU and the GPU do, and only in the overworld: the
differences come from what changes from one run to the next, not from the GPU. Probably the order in which
neighbouring chunks are decorated across their borders, and the server's ticks during `/pregen`, which let
fluids flow (PREGEN.md); telling the two apart is still to do.

## What the numbers mean

- **chunks/s** is the server's own count at the end of `/pregen` (`Done. Pregenerating ...`): generation,
  light and saving of every chunk, with the server's normal tick around it.
- **CPU per chunk** is the CPU time of the game process during `/pregen` (Linux, `/proc`) divided by the
  chunks: the work saved, even where the throughput is held by something else (the 6 physical cores, the
  waits on the GPU).
- One pass varies by 1 to 2 % in throughput and 0.5 to 1 % in CPU per chunk between zones; smaller
  differences need repeats (`--repeat`) before being read as gains.
