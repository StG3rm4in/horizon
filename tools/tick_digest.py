#!/usr/bin/env python3
"""The exactness harness of the server tick: runs a fixture on a dedicated server and compares the
per-tick digests of two runs (two builds, or one build twice).

The server side is `crates/horizon-client/src/game/interface/dedicated/tick_digest.rs`: with
HORIZON_TICK_FIXTURE_TICKS the server runs that many ticks back to back, waiting before each one for
the chunk work in flight and applying its results in a fixed order, then stops; with HORIZON_TICK_DIGEST
it writes one line of hashes per tick (random, block writes, scheduled ticks, entities, players, loaded
chunks, and every --full ticks the blocks and light of every loaded section).

    tick_digest.py check --bin A [--bin B] [--ticks N] [--commands FILE]   two runs, compared
    tick_digest.py run --bin A --out DIR [...]                             one run
    tick_digest.py compare DIGEST_A DIGEST_B                               the first tick that differs

A fixture is a seed, a number of ticks and console commands run before given ticks (lines
`TICK COMMAND`, ticks counted from 1). The default one force-loads the 16 x 16 chunks around the
origin before the first tick: generation, the mobs it spawns, random and scheduled ticks, no player.
When two runs differ, `check` runs both again with the rows behind the hashes of the first tick that
differs (`--detail`) and prints the first rows that differ.
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import lab

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / ".local/digest-target/release"
DEFAULT_COMMANDS = "1 forceload add -128 -128 127 127\n"
# The scenes `--scene` picks. `default` is the commands above with the server properties below:
# creative players flying by day in the Overworld. `survival` goes where it never does: survival
# mode on hard at midnight, players on the ground who hit the monsters in reach and respawn, about
# one in ten in the Nether and one in twenty-five in the End. Its players' play is
# crates/horizon-client/src/game/interface/dedicated/tick_digest/scene.rs (any player out of
# creative mode), whose offsets follow the arena below.
SCENES = ("default", "survival")
# The server properties of the scene in use, over the defaults (`main` sets them before any run).
SCENE_PROPERTIES = {}
# Flight allowed: a player standing where a creeper opened the floor is not kicked for floating.
SURVIVAL_PROPERTIES = {"gamemode": "survival", "difficulty": "hard", "allow-flight": "true"}
# The survival scene's world. `~` is the console's position, the world spawn's block: the arena's
# middle, where every player joins and respawns (`respawn_radius 0`). The arena is a 25 x 25 stone
# floor cleared 9 blocks up. Nothing that collides stands within 5 blocks of its middle, so the
# players' straight walks never run into a block; the ring 5 out, which they patrol, holds cobwebs,
# a pressure plate, berry bushes, powder snow, magma, wither roses, eyeblossoms, fire and a weighted
# plate. Past the ring, by the guards' posts: a zombie spawner (blind to light, as the fire lights
# the arena), a trial spawner and a vault; then a lit furnace, a chest over a hopper into a chest,
# lava, cacti and a tripwire; 40 blocks out, a spawner and a trial spawner no player comes near. Two
# waves of monsters follow, with a villager, an iron golem and bees. In the Nether: a room at y 70,
# under the roof (flat bedrock at y 127) where the other half of its players roam. In the End: a
# platform 1200 blocks out, as the dragon's fight needs a player within 192 blocks of its island
# and the island's chunks loaded.
SURVIVAL_COMMANDS = """\
1 forceload add -128 -128 127 127
1 forceload add ~-48 ~-48 ~47 ~47
1 gamerule respawn_radius 0
1 time set 18000
1 fill ~-12 ~-4 ~-12 ~12 ~-1 ~12 minecraft:stone
1 fill ~-12 ~ ~-12 ~12 ~8 ~12 minecraft:air
1 fill ~-3 ~ ~-5 ~-2 ~ ~-5 minecraft:cobweb
1 setblock ~ ~ ~-5 minecraft:stone_pressure_plate
1 fill ~2 ~-1 ~-5 ~3 ~-1 ~-5 minecraft:grass_block
1 fill ~2 ~ ~-5 ~3 ~ ~-5 minecraft:sweet_berry_bush[age=1]
1 fill ~5 ~ ~-3 ~5 ~ ~-2 minecraft:powder_snow
1 fill ~5 ~-1 ~1 ~5 ~-1 ~3 minecraft:magma_block
1 fill ~2 ~-1 ~5 ~3 ~-1 ~5 minecraft:grass_block
1 fill ~2 ~ ~5 ~3 ~ ~5 minecraft:wither_rose
1 setblock ~ ~ ~5 minecraft:light_weighted_pressure_plate
1 fill ~-4 ~-1 ~5 ~-2 ~-1 ~5 minecraft:grass_block
1 fill ~-3 ~ ~5 ~-2 ~ ~5 minecraft:open_eyeblossom
1 setblock ~-4 ~ ~5 minecraft:closed_eyeblossom
1 fill ~-5 ~-1 ~-3 ~-5 ~-1 ~-2 minecraft:netherrack
1 fill ~-5 ~ ~-3 ~-5 ~ ~-2 minecraft:fire
1 setblock ~8 ~ ~ minecraft:spawner{SpawnData:{entity:{id:"minecraft:zombie"},custom_spawn_rules:{block_light_limit:[I;0,15],sky_light_limit:[I;0,15]}},Delay:20,MinSpawnDelay:100,MaxSpawnDelay:200,SpawnCount:2,MaxNearbyEntities:6,RequiredPlayerRange:16,SpawnRange:3}
1 setblock ~-8 ~ ~ minecraft:trial_spawner{required_player_range:14,target_cooldown_length:200,normal_config:{spawn_potentials:[{data:{entity:{id:"minecraft:zombie"}},weight:1}],total_mobs:4.0f,simultaneous_mobs:2.0f,total_mobs_added_per_player:0.0f,simultaneous_mobs_added_per_player:0.0f,ticks_between_spawn:40}}
1 setblock ~ ~ ~8 minecraft:vault
1 setblock ~8 ~ ~8 minecraft:furnace{Items:[{Slot:0b,id:"minecraft:raw_iron",count:16},{Slot:1b,id:"minecraft:coal",count:8}]}
1 setblock ~-8 ~ ~8 minecraft:hopper[facing=south]
1 setblock ~-8 ~1 ~8 minecraft:chest{Items:[{Slot:0b,id:"minecraft:cobblestone",count:64},{Slot:1b,id:"minecraft:iron_ingot",count:32}]}
1 setblock ~-8 ~ ~9 minecraft:chest
1 fill ~8 ~-1 ~-4 ~8 ~-1 ~-3 minecraft:lava
1 setblock ~8 ~-1 ~-8 minecraft:sand
1 setblock ~8 ~ ~-8 minecraft:cactus
1 setblock ~-8 ~-1 ~-8 minecraft:sand
1 setblock ~-8 ~ ~-8 minecraft:cactus
1 setblock ~-12 ~ ~-3 minecraft:stone
1 setblock ~-11 ~ ~-3 minecraft:tripwire_hook[facing=east]
1 setblock ~-6 ~ ~-3 minecraft:stone
1 setblock ~-7 ~ ~-3 minecraft:tripwire_hook[facing=west]
1 fill ~-10 ~ ~-3 ~-8 ~ ~-3 minecraft:tripwire
1 setblock ~40 ~ ~ minecraft:spawner{SpawnData:{entity:{id:"minecraft:skeleton"}}}
1 setblock ~-40 ~ ~ minecraft:trial_spawner{normal_config:{spawn_potentials:[{data:{entity:{id:"minecraft:zombie"}},weight:1}]}}
1 execute in minecraft:the_nether run forceload add 0 0 31 31
1 execute in minecraft:the_nether run fill 8 68 8 24 69 24 minecraft:netherrack
1 execute in minecraft:the_nether run fill 8 70 8 24 74 24 minecraft:air
1 execute in minecraft:the_nether run fill 19 69 15 19 69 17 minecraft:magma_block
1 execute in minecraft:the_nether run setblock 12 69 12 minecraft:soul_sand
1 execute in minecraft:the_nether run setblock 12 70 12 minecraft:nether_wart
1 execute in minecraft:the_nether run setblock 21 69 21 minecraft:soul_soil
1 execute in minecraft:the_nether run setblock 21 70 21 minecraft:soul_fire
1 execute in minecraft:the_nether run setblock 16 69 21 minecraft:lava
1 execute in minecraft:the_end run forceload add 1184 -16 1215 15
1 execute in minecraft:the_end run fill 1192 58 -8 1208 59 8 minecraft:end_stone
1 execute in minecraft:the_end run fill 1192 60 -8 1208 64 8 minecraft:air
1 execute in minecraft:the_end run setblock 1207 60 7 minecraft:chorus_flower
60 summon minecraft:zombie ~-11 ~ ~-11
60 summon minecraft:zombie ~11 ~ ~-11
60 summon minecraft:zombie ~-11 ~ ~11
60 summon minecraft:zombie ~11 ~ ~11
60 summon minecraft:skeleton ~-11 ~ ~5
60 summon minecraft:skeleton ~11 ~ ~4
60 summon minecraft:spider ~4 ~ ~-11
60 summon minecraft:spider ~-4 ~ ~11
60 summon minecraft:creeper ~11 ~ ~-6 {ExplosionRadius:1b}
60 summon minecraft:villager ~2 ~ ~-8
60 summon minecraft:iron_golem ~-2 ~ ~-9
60 summon minecraft:bee ~-3 ~1 ~7
60 summon minecraft:bee ~3 ~1 ~7
60 effect give @e[type=minecraft:zombie,distance=..24] minecraft:absorption 120 1
120 effect give @a[distance=..12] minecraft:absorption 60 0
300 summon minecraft:zombie ~-10 ~ ~-10
300 summon minecraft:zombie ~10 ~ ~10
300 summon minecraft:husk ~-11 ~ ~1
300 summon minecraft:skeleton ~10 ~ ~-10
300 summon minecraft:spider ~-10 ~ ~10
300 summon minecraft:witch ~ ~ ~-11
300 effect give @e[type=minecraft:skeleton,distance=..24] minecraft:speed 60 0
"""


def survival_commands(players, join_every):
    """The survival scene's commands for PLAYERS joining every JOIN_EVERY ticks: its world, then
    each traveller's teleport 5 ticks after it joined (player i joins on tick i * JOIN_EVERY + 1).
    Players 9, 19, 29... go to the Nether: 19, 39... onto its roof, the others into its room.
    Players 12, 37, 62, 87 go to the End."""
    lines = [SURVIVAL_COMMANDS]
    for i in range(players):
        at = i * join_every + 6
        if i % 10 == 9:
            where = "8.5 128 8.5" if i % 20 == 19 else "16.5 70 16.5"
            lines.append(f"{at} execute in minecraft:the_nether run tp fixture{i} {where}\n")
        elif i % 25 == 12:
            lines.append(f"{at} execute in minecraft:the_end run tp fixture{i} 1200.5 60 0.5\n")
    return "".join(lines)
GPU_STAGES = ("DENSITY", "SURFACE", "ORES", "CAVES", "LIGHT", "SECTIONS", "BIOMES")


def free_port():
    """A port nothing listens on now, so that two harnesses (or a harness and a bench) can run at
    once; the game binds it a few seconds later, and a port taken in between ends the run with a
    clear message (see `run`)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def server_properties(seed, port):
    properties = dict(line.split("=", 1) for line in [
        f"level-seed={seed}", "view-distance=10", "simulation-distance=10", "online-mode=false",
        "white-list=false", "gamemode=creative", "force-gamemode=true", "spawn-protection=0",
        "max-players=200", "difficulty=easy", "sync-chunk-writes=true", f"server-port={port}",
        # no pause while nobody is on, no watchdog: a fixture tick waits for its chunk work
        "pause-when-empty-seconds=0", "max-tick-time=-1",
    ])
    # the scene's own (`--scene`), in place of these or after them
    properties.update(SCENE_PROPERTIES)
    return "".join(f"{key}={value}\n" for key, value in properties.items())


def run(bin_dir, out, ticks, commands, seed, gpu, full, players=0, join_every=10, detail=None,
        timeout=None):
    """One fixture run into OUT (a fresh server directory); returns the digest's path."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    hs = Path(bin_dir) / "horizon-server"
    (out / "server.properties").write_text(server_properties(seed, free_port()))
    (out / "eula.txt").write_text("eula=true\n")
    imported = subprocess.run([hs, "--dir", out, "import", out / "server.properties"],
                              capture_output=True, text=True)
    if imported.returncode != 0:
        sys.exit(f"import failed: {imported.stdout}{imported.stderr}")
    toml = out / "server.toml"
    text = toml.read_text().replace('bind = "127.0.0.1:8130"', 'bind = ""')
    toml.write_text(text + "\n[anticheat]\nenabled = false\n")
    (out / "fixture.txt").write_text(commands)
    digest = out / "digest.txt"
    # Fresh home and XDG folders in the run's folder (CLAUDE.md): nothing of the machine's own is
    # read or written. The prepared game files are found beside the program, not in a home.
    homes = {key: out / "home" / key.lower()
             for key in ("HORIZON_HOME", "HORIZON_LAUNCHER_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME")}
    for folder in homes.values():
        folder.mkdir(parents=True)
    env = lab.game_env(**homes)
    env.update({
        "HORIZON_TICK_DIGEST": str(digest),
        "HORIZON_TICK_DIGEST_FULL": str(full),
        "HORIZON_TICK_FIXTURE_TICKS": str(ticks),
        "HORIZON_TICK_FIXTURE_COMMANDS": str(out / "fixture.txt"),
        "HORIZON_TICK_FIXTURE_PLAYERS": str(players),
        "HORIZON_TICK_FIXTURE_JOIN_EVERY": str(join_every),
        # entity randoms from a counter per thread instead of the clock
        "HORIZON_TEST_SEEDS": "1",
    })
    for stage in GPU_STAGES:
        env.pop(f"HORIZON_GPU_{stage}", None)
        if gpu:
            env[f"HORIZON_GPU_{stage}"] = "1"
    if gpu:
        # The density kernels compiled before the first chunk: with `1` the chunks generated while
        # they compile take the CPU, as many as the timing gives.
        env["HORIZON_GPU_DENSITY"] = "wait"
    if detail:
        env["HORIZON_TICK_DIGEST_DETAIL"] = detail
    mode = env.get("HORIZON_PARALLEL_TICK")
    print(f"== {out}: HORIZON_PARALLEL_TICK={'unset' if mode is None else mode} ({tick_mode(mode)})")
    (out / "env.txt").write_text("".join(f"{k}={v}\n" for k, v in sorted(env.items())
                                         if k.startswith(("HORIZON_", "XDG_"))))
    # Generous: the startup's generation, then the ticks (a player's flight adds chunk work).
    timeout = timeout or 300 + ticks * (1 + players // 10)
    started = time.monotonic()
    count = lambda: len(digest.read_text().splitlines()) if digest.exists() else 0
    with open(out / "run.log", "w") as log:
        # The supervisor stays once the game has stopped: it is stopped here when the game (its
        # child) is gone, after the last digest line, after a crash, or after a clean stop before
        # the end (the port taken), which writes no more lines either.
        supervisor = subprocess.Popen([hs, "--dir", out, "run"], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        game_seen = False
        while supervisor.poll() is None:
            children = subprocess.run(["pgrep", "-P", str(supervisor.pid)], capture_output=True,
                                      text=True).stdout.split()
            game_seen = game_seen or bool(children)
            if game_seen and not children:
                tail = log_tail(out)
                if count() >= ticks or "crash" in tail or "stopping server" in tail:
                    break
            if time.monotonic() - started > timeout:
                print(f"{out}: no end after {timeout} s")
                break
            time.sleep(0.5)
        supervisor.terminate()
        try:
            supervisor.wait(30)
        except subprocess.TimeoutExpired:
            supervisor.kill()
    lines = digest.read_text().splitlines() if digest.exists() else []
    print(f"== {out}: {len(lines)} ticks in {time.monotonic() - started:.0f} s")
    if len(lines) not in (ticks, ticks + 1):
        why = " (its port was taken)" if "failed to bind to port" in log_tail(out) else ""
        sys.exit(f"{out}: {len(lines)} digest lines for {ticks} ticks{why} "
                 f"(see {out}/logs/latest.log)")
    return digest


def log_tail(out):
    log = Path(out) / "logs/latest.log"
    return log.read_text(errors="replace")[-4000:].lower() if log.exists() else ""


def fields(line):
    tick, *parts = line.split()
    return int(tick), dict(p.split("=", 1) for p in parts)


def compare(a, b):
    """The first tick whose digests differ and the parts that differ, or None."""
    la, lb = Path(a).read_text().splitlines(), Path(b).read_text().splitlines()
    for x, y in zip(la, lb):
        if x != y:
            (tick, fa), (_, fb) = fields(x), fields(y)
            parts = [k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)]
            return tick, sorted(parts), fa, fb
    if len(la) != len(lb):
        return min(len(la), len(lb)) + 1, ["length"], {}, {}
    return None


# The kinds of detail rows, in the order a line writes them: the level in place's, then the
# server's own (tick_digest/parts.rs). Another level's rows have them with `@level` after them.
KINDS = "gwsehrtcmfbnpdka"


def kind_order(kind):
    """The order detail_diff shows a kind of row in: the level in place's first, by KINDS."""
    base, _, level = kind.partition("@")
    return level, KINDS.index(base) if len(base) == 1 and base in KINDS else len(KINDS), base


def detail_diff(a, b, limit=20):
    """The first rows of two detail files that differ, by kind of row: every kind either file has."""
    rows = []
    for digest in (a, b):
        kinds = {}
        for row in Path(str(digest) + ".detail").read_text().splitlines():
            if not row.startswith("#"):
                kinds.setdefault(row.split(" ", 1)[0], []).append(row)
        rows.append(kinds)
    shown = 0
    for kind in sorted(rows[0].keys() | rows[1].keys(), key=kind_order):
        xa, xb = rows[0].get(kind, []), rows[1].get(kind, [])
        if xa == xb:
            continue
        print(f"-- rows '{kind}': {len(xa)} against {len(xb)}")
        for i, (x, y) in enumerate(zip(xa, xb)):
            if x != y:
                print(f"   row {i}:\n   A {x}\n   B {y}")
                shown += 1
                if shown >= limit:
                    return
        if len(xa) != len(xb):
            extra = xa[len(xb):] or xb[len(xa):]
            print(f"   {'A' if len(xa) > len(xb) else 'B'} has {len(extra)} more, first: {extra[0]}")


def tick_mode(value):
    """What the game makes of HORIZON_PARALLEL_TICK (horizon-client parallel.rs, `forced`)."""
    if value in ("0", "off", "false"):
        return "serial"
    if value in ("1", "on", "true"):
        return "parallel, with randoms of its own"
    if value in ("verify", "exact"):
        return value
    return "serial: a dedicated server has no setting to follow"


def hold(work):
    """Takes WORK for this check: a lock file in it, held (flock) until the process ends, so that a
    second check on the same folder stops instead of removing this one's worlds. The run folders in
    it are the fixed four, each emptied before it is used; the detail ones of an earlier check go."""
    import fcntl  # here: Windows has none, and `compare` keeps working there
    work.mkdir(parents=True, exist_ok=True)
    lock = open(work / ".tick_digest.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        holder = lock.read().strip() or "?"
        sys.exit(f"{work} is in use by another tick_digest.py (pid {holder}): use another --work")
    lock.seek(0)
    lock.truncate()
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    for stale in ("run-a-detail", "run-b-detail"):
        shutil.rmtree(work / stale, ignore_errors=True)
    return lock


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--bin", default=str(DEFAULT_BIN), help="folder with horizon-server and horizon-client")
        p.add_argument("--ticks", type=int, default=200)
        p.add_argument("--commands", help="fixture commands file (lines TICK COMMAND)")
        p.add_argument("--seed", default="12345")
        p.add_argument("--gpu", action="store_true",
                       help="generate with every GPU stage (default: the CPU)")
        p.add_argument("--full", type=int, default=20, help="full block and light hash every N ticks")
        p.add_argument("--players", type=int, default=0, help="simulated players (horizon-bots mixed)")
        p.add_argument("--join-every", type=int, default=10, help="ticks between two players joining")
        p.add_argument("--scene", choices=SCENES, default="default",
                       help="default: creative players by day; survival: see SURVIVAL_COMMANDS")
        p.add_argument("--detail", help="rows behind the hashes of ticks FROM-TO")
        p.add_argument("--work", default="/data/tick-digest", help="where the run directories go")
    sub.choices["check"].add_argument("--bin2", help="second build (default: --bin again)")
    sub.choices["check"].add_argument("--repeat", type=int, default=1,
                                      help="runs of the second build, each compared with the first")
    sub.choices["run"].add_argument("--out", required=True)
    for name in ("check", "run"):
        sub.choices[name].add_argument(
            "--parallel", choices=("serial", "exact", "inherit"), default="inherit",
            help="HORIZON_PARALLEL_TICK for the game: 0, exact, or the caller's (the default)")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args()

    if args.cmd == "compare":
        diff = compare(args.a, args.b)
        print("identical" if diff is None else f"first difference at tick {diff[0]}: {', '.join(diff[1])}")
        sys.exit(diff is not None)

    if args.parallel != "inherit":
        os.environ["HORIZON_PARALLEL_TICK"] = {"serial": "0", "exact": "exact"}[args.parallel]

    if args.scene == "survival":
        SCENE_PROPERTIES.update(SURVIVAL_PROPERTIES)
        scene_commands = survival_commands(args.players, args.join_every)
    else:
        scene_commands = DEFAULT_COMMANDS
    # --commands replaces the scene's commands, not its server properties
    commands = Path(args.commands).read_text() if args.commands else scene_commands
    common = dict(ticks=args.ticks, commands=commands, seed=args.seed, gpu=args.gpu, full=args.full,
                  players=args.players, join_every=args.join_every)
    if args.cmd == "run":
        run(args.bin, args.out, detail=args.detail, **common)
        return
    work = Path(args.work)
    lock = hold(work)  # held until the process ends
    bins = (args.bin, args.bin2 or args.bin)
    first = run(bins[0], work / "run-a", detail=args.detail, **common)
    for k in range(args.repeat):
        digests = [first, run(bins[1], work / "run-b", detail=args.detail, **common)]
        diff = compare(*digests)
        if diff is not None:
            break
        print(f"== identical: {args.ticks} ticks (run {k + 1} of {args.repeat})")
    else:
        return
    tick, parts, fa, fb = diff
    print(f"== first difference at tick {tick}: {', '.join(parts)}")
    for k in parts:
        print(f"   {k}: A {fa.get(k)}  B {fb.get(k)}")
    if not args.detail and "length" not in parts:
        print(f"== again with the rows of tick {tick}")
        common["ticks"] = max(tick, 1)
        digests = [run(b, work / f"run-{k}-detail", detail=f"{tick}-{tick}", **common)
                   for k, b in zip("ab", bins)]
        detail_diff(*digests)
    sys.exit(1)


if __name__ == "__main__":
    main()
