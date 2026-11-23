# Horizon GPU

World generation on the GPU and a faster server tick, for Horizon.

[Français](README.fr.md)

This is a fork of [Horizon](https://github.com/horizon-rust/horizon), a from-scratch Rust reimplementation of
Minecraft: Java Edition 26.3 (client, server and launcher) that is checked against the vanilla game. The fork
adds two things, under the same rule as upstream: **nothing may differ from vanilla**.

1. **World generation on the GPU.** Density, aquifers, the beardifier, surface rules, ores, caves, light,
   sections and biomes run on an NVIDIA GPU (CUDA) in the Overworld. The Nether and the End have no aquifers;
   there the GPU runs the density and the caves, and in the Nether also the surface rules and the biomes; the
   End's biomes and the Nether's compact sections stay on the CPU (`crates/horizon-gpu/NETHER_END.md`). Every
   chunk comes out identical, block for block, to the CPU path, which is itself identical to vanilla.
2. **A faster dedicated server tick with 100 players.** Each optimization keeps the tick bit-identical to the
   code before it, which a tick-by-tick harness checks on a fixed scene (see below).

## Results

Measured on one machine: an Intel i5-10600K (6 cores, 12 threads) at 4.9 GHz, 32 GB of RAM in dual channel, an
NVIDIA RTX 3080.

### World generation (`/pregen`, fresh zones)

| Dimension | CPU only | CPU + GPU |
|---|---|---|
| Overworld | 853 chunks/s, 12.9 ms of CPU per chunk | **2,287 chunks/s (×2.7)**, 3.4 ms of CPU per chunk |
| Nether | — | ×2.4 |
| End | — | ×1.9 |

What limits it now is the 6 CPU cores: the GPU is busy about two thirds of the time.

### Server tick, 100 players (mean milliseconds per tick; 20 ticks per second needs less than 50 ms)

| Build | ms per tick |
|---|---|
| First bench, 16/11/2026 | 241 (about 4 ticks per second) |
| Exact mode, native build, 22/11/2026 | 55.0 |
| **Exact mode, PGO build** | **44.2** |
| **Exact mode, PGO, tick thread pinned to one core** | **42.1** |
| Mob islands (opt-in, not exact), PGO | 34.7 |
| Mob islands, PGO, pinned | 33.5 |

The bench is 100 bots flying around one world, each loading its own area; every figure is the mean of runs
taken in A B B A order. The first row includes upstream's own progress since, which the fork merges regularly.

## How exactness is checked

- **World generation:** the generated chunks are compared with the vanilla server jar's at seed 12345 (785
  chunks across 27 biomes and 74 structure sites: 0 differences, on the GPU and on the CPU), and a probe compares
  the CPU and the GPU paths in the three dimensions. One seed is compared with vanilla; the GPU stages also
  have test modes that compare each result with the CPU's.
- **Server tick:** a tick-digest harness runs two fixed scenes for 600 ticks with 100 simulated players: the
  default one (creative players flying in the Overworld by day) and a survival one (night, hard difficulty, an
  arena of spawners and hazards, mobs fighting the players, players in the Nether and the End). Every tick it
  hashes the level random and its gaussian, the block writes, the scheduled ticks, every entity's motion and its
  fields read in place (health, absorption, hurt and death times, fire, freeze, air, effects, target and the
  player it is bound to, yaws, swing...), every entity's own random, the counters, the ticking block entities'
  NBT, each player's saved data, every packet each player is sent, every level that ticks and the loaded chunks
  (and every 50 ticks the blocks and light of every section). Two builds must give the same lines, serially and
  with the parallel tick (`HORIZON_PARALLEL_TICK=exact`).

  This shows that an optimization changes nothing against the build before it, on those scenes; it is not a
  comparison with vanilla. The harness is part of every server build but runs only with `HORIZON_TICK_DIGEST`
  set (`tools/tick_digest.py` sets it).
- **Mob islands:** `HORIZON_PARALLEL_TICK=verify` ticks the mobs both in groups and serially, and compares them.

## What was optimized

- The GPU stages above, with a CPU fallback for every stage and for anything a GPU stage does not cover.
- Generation and saving on the CPU: decorations, a direct NBT writer for saved chunks, section palettes stored
  inline.
- The tick: shortcuts in the style of Lithium (a mob standing still on a full block, block contacts through a bit
  table, block entities with no player nearby), compact random tick masks, natural spawning that keeps per-chunk
  and per-pass work, players' bodies shared between turns, fewer lookups and allocations in hot paths.
- The build: profile-guided optimization (PGO, about −20 % in exact mode) and an optional pinning of the tick
  thread to one CPU (`HORIZON_TICK_CPU`).
- Mob islands (branch `parallel-mobs`): mobs that cannot affect each other are ticked in groups on helper
  threads, and the result is checked to equal the serial loop's.

## Branches

| Branch | What it holds |
|---|---|
| `gpu-worldgen` (default) | GPU world generation and the exact tick optimizations |
| `parallel-mobs` | `gpu-worldgen` plus the mob islands |
| `main` | upstream Horizon, unchanged |

## Using it

Build and run as upstream does (see the [upstream README](../README.md)). The options of the fork:

- `HORIZON_GPU_<STAGE>=1` for `DENSITY`, `SURFACE`, `ORES`, `CAVES`, `LIGHT`, `SECTIONS`, `BIOMES`: run that
  stage on the GPU (CUDA). The game falls back to the CPU when no GPU is available.
- `HORIZON_PARALLEL_TICK=exact`: the parallel tick, kept exact.
- `HORIZON_TICK_CPU=<cpu>` (Linux): pin the dedicated server's tick thread to one CPU.
- Mob islands: a build of `parallel-mobs` with the feature `upstream-parallel-mob-islands`, and
  `parallel_tick = true` in `server.toml`.

## Notes

- License: LGPL-3.0-or-later, as upstream. No Minecraft game files are included; Horizon prepares them from
  your own copy of the game.
- Not an official Minecraft product, and not approved by or associated with Mojang or Microsoft.
- The fork's code was written with Claude (Anthropic) under the repository owner's direction, and every change
  is checked by the harnesses above before it is merged.
