# Horizon GPU

World generation on the GPU and a faster server tick, for Horizon.

[Français](README.fr.md)

This is a fork of [Horizon](https://github.com/horizon-rust/horizon), a Rust reimplementation of Minecraft: Java Edition 26.3 with a client, server and launcher. The fork aims to preserve vanilla world generation and the serial tick's behavior in `exact` mode.

1. **World generation on the GPU.** The Overworld can use NVIDIA CUDA for final density, aquifers, the beardifier, surface rules, ore work, carving, lighting for generated chunks, compact sections and multi-noise biomes. The Nether also supports GPU density, carving, surface rules and biomes. The End supports GPU density. Both dimensions use the global fluid picker in place of enabled aquifers. End biomes and Nether compact sections stay on the CPU. See the [Nether and End notes](../crates/horizon-gpu/NETHER_END.md).
2. **A faster dedicated server tick.** The fork measures performance with 100 load-test bots and compares Horizon builds tick by tick in controlled scenes.

## Results

The benchmark server has an Intel i5-10600K (6 cores, 12 threads), 32 GB of dual-channel RAM and an NVIDIA RTX 3080.

### World generation (`/pregen`, fresh servers)

The 17 November 2026 comparison used the same fork build for CPU-only and CPU + GPU runs, with two repetitions.

| Dimension | CPU only | CPU + GPU | Throughput ratio |
|---|---:|---:|---:|
| Overworld | 853 chunks/s, 12.92 ms CPU/chunk | **2,287 chunks/s, 3.37 ms CPU/chunk** | ×2.68 |
| Nether | 1,340 chunks/s | 3,183 chunks/s | ×2.38 |
| End | 3,586 chunks/s | 6,879 chunks/s | ×1.92 |

In the profiled Overworld GPU runs, throughput is limited by the 6 physical CPU cores rather than GPU saturation.

### Dedicated server tick, 100 bots

Each number is one run's mean tick time. At 20 ticks per second the budget is 50 ms per tick.

| Build and mode | Date | Mean ms per tick |
|---|---|---:|
| Historical baseline | 16 Nov 2026 | 241 (~4 TPS) |
| `gpu-worldgen` (`d46ac9ab1f`), native, `exact` | 23 Nov 2026 | 42.6 / 42.9 |
| Fork build, fresh PGO, `exact` | 24 Nov 2026 | **37.8 / 38.1** |

The 23–24 November timings used 100 bots in the `mixed` workload, A–B–B–A comparisons, and a 4.9 GHz CPU clock during timing runs. The improvement since the initial baseline spans both fork changes and upstream merges.

## How exactness is checked

- **World generation.** `tools/test_chunk_generation_parity.py` compares sampled chunks with a Java oracle using the vanilla server JAR. The project recorded zero differences on the CPU and GPU paths for a sample of 785 chunks across 27 biomes and 74 structure sites, using seed 12345 and a Nether regression site. `tools/gpu_worldgen.py probe` compares CPU and GPU chunk dumps from sampled zones in all three dimensions, and density also has a startup self-test alongside stage comparison modes. These checks verify the sampled chunks and zones rather than universal vanilla equivalence or saved-world reproducibility across repeated runs. See the [benchmark and verification guide](../crates/horizon-gpu/BENCHMARKS.md).
- **Server tick.** The dedicated game engine includes a digest harness (`tools/tick_digest.py`, controlled by `HORIZON_TICK_DIGEST`). Validation runs cover 600 ticks and 100 simulated players across two scenes—creative daytime flight in the Overworld, and hard nighttime survival with combat, spawners, hazards, and Nether and End travel—with full section block and light checkpoints every 50 ticks. The digest hashes selected state from ticking levels and tracked entities: RNG state, recorded block writes, scheduled block and fluid ticks, loaded chunks, normalized block-entity NBT, tracked player save data, and normalized fixture packets. **This is a regression check between Horizon builds in the same mode (serial vs. serial, `exact` vs. `exact`), not a comparison with vanilla**; vanilla parity is checked on world generation with the Java oracle above.

## What was optimized

- The GPU generation stages use CPU paths when the GPU backend is unavailable or the work falls outside a stage's coverage. Optional GPU work outside chunk generation includes biome searches and explosion exposure rays.
- CPU generation and saving use faster decoration paths, direct chunk NBT writing and inline storage for small palette ID lists.
- The tick uses fast paths for stationary mobs on full blocks, block contacts and some block-entity proximity checks. It also uses compact random-tick masks, reuses work within natural-spawning passes and shares player collision bodies between mob turns, reducing lookups and allocations in hot paths.
- The build can use profile-guided optimization (PGO). In the matched 24 November `exact` comparison, mean tick time fell by about 12% across the paired runs. On Linux, the dedicated tick thread can also be pinned.

## Branches

| Branch | What it holds |
|---|---|
| `gpu-worldgen` (default) | GPU world generation and the `exact` tick optimizations |
| `parallel-mobs` | The shared code plus opt-in mob islands and natural spawning in lanes on helper threads |
| `main` | Upstream Horizon, advanced by fast-forward |

## Using it

Follow the [upstream README](../README.md) for setup. The game build enables `cuda` by default; if you build the dedicated game engine with `--no-default-features`, add `--features cuda` for GPU support. The CUDA driver and NVRTC are loaded at runtime.

| Variable | Values | Requires / Precedence | Effect |
|---|---|---|---|
| `HORIZON_GPU_DENSITY` | `1`, `wait` | CUDA runtime | Enables supported GPU final density (with aquifers and beardifier); `1` compiles in the background with CPU fallback meanwhile, `wait` blocks for initialization |
| `HORIZON_GPU_SURFACE` | `1` | GPU fill | Enables supported GPU surface rules |
| `HORIZON_GPU_SECTIONS` | `1` | GPU surface path (`HORIZON_GPU_SURFACE`) | Enables GPU compact sections |
| `HORIZON_GPU_CAVES` | `1` | `HORIZON_GPU_DENSITY` for carved-cell substance | Enables supported GPU carving (carving masks can run without GPU density) |
| `HORIZON_GPU_ORES` | `1` | CUDA runtime | Enables supported GPU ore-blob and underground ore-vein work |
| `HORIZON_GPU_LIGHT` | `1` | CUDA runtime | Enables GPU lighting for newly generated chunks |
| `HORIZON_GPU_BIOMES` | `1` | CUDA runtime | Enables GPU multi-noise biomes on eligible sources |
| `HORIZON_GPU_LOCATE` | `1` | CUDA runtime | Enables supported GPU biome searches (including `/locate biome`) |
| `HORIZON_GPU_TNT` | `1` | CUDA runtime | Lets a cost model select CUDA for explosion exposure rays |
| `HORIZON_PARALLEL_TICK` | `exact`, `0` | — | `exact` shares eligible tick work while keeping serial random streams; `0` selects the serial tick |
| `HORIZON_TICK_CPU` | `<cpu>` | Linux dedicated server; `HORIZON_SERVER_CORE=1` takes precedence | Pins the dedicated tick thread to a CPU core |

**Using `parallel-mobs`:** The `parallel-mobs` branch adds opt-in mob islands and natural spawning in lanes on helper threads. Build with `horizon-client/upstream-parallel-mob-islands` (which implies `upstream-parallel-mobs`) for both, or `horizon-client/upstream-parallel-mobs` for spawning lanes alone, and enable at runtime with `HORIZON_PARALLEL_TICK=1`, `[engine] parallel_tick = true` in `server.toml`, or the single-player Parallel Tick settings. These paths can use per-mob or per-chunk random streams outside `exact` parity (and therefore can differ from serial and vanilla draws); `HORIZON_PARALLEL_TICK=verify` on that branch checks eligible multi-group ticks against a single-group replayed-seed reference.

## Notes

- Horizon and this fork are licensed under LGPL-3.0-or-later. Horizon prepares Minecraft resources from your own copy of the game.
- This is an unofficial community fork.
- The project notes credit work with Claude, Codex and Gemini under the repository owner's direction.
