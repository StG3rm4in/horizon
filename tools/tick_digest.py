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
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / ".local/digest-target/release"
DEFAULT_COMMANDS = "1 forceload add -128 -128 127 127\n"
GPU_STAGES = ("DENSITY", "SURFACE", "ORES", "CAVES", "LIGHT", "SECTIONS", "BIOMES")


def server_properties(seed):
    return "\n".join([
        f"level-seed={seed}", "view-distance=10", "simulation-distance=10", "online-mode=false",
        "white-list=false", "gamemode=creative", "force-gamemode=true", "spawn-protection=0",
        "max-players=200", "difficulty=easy", "sync-chunk-writes=true", "server-port=25599",
        # no pause while nobody is on, no watchdog: a fixture tick waits for its chunk work
        "pause-when-empty-seconds=0", "max-tick-time=-1",
    ]) + "\n"


def run(bin_dir, out, ticks, commands, seed, gpu, full, players=0, join_every=10, detail=None,
        timeout=None):
    """One fixture run into OUT (a fresh server directory); returns the digest's path."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    hs = Path(bin_dir) / "horizon-server"
    (out / "server.properties").write_text(server_properties(seed))
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
    env = dict(os.environ)
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
    # Generous: the startup's generation, then the ticks (a player's flight adds chunk work).
    timeout = timeout or 300 + ticks * (1 + players // 10)
    started = time.monotonic()
    count = lambda: len(digest.read_text().splitlines()) if digest.exists() else 0
    with open(out / "run.log", "w") as log:
        # The supervisor stays once the game has stopped: it is stopped here when the game (its
        # child) is gone, after the last digest line or after a crash.
        supervisor = subprocess.Popen([hs, "--dir", out, "run"], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        game_seen = False
        while supervisor.poll() is None:
            children = subprocess.run(["pgrep", "-P", str(supervisor.pid)], capture_output=True,
                                      text=True).stdout.split()
            game_seen = game_seen or bool(children)
            if game_seen and not children and (count() >= ticks or "crash" in log_tail(out)):
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
        sys.exit(f"{out}: {len(lines)} digest lines for {ticks} ticks (see {out}/logs/latest.log)")
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


def detail_diff(a, b, limit=20):
    """The first rows of two detail files that differ, by kind of row."""
    ra = Path(str(a) + ".detail").read_text().splitlines()
    rb = Path(str(b) + ".detail").read_text().splitlines()
    shown = 0
    for kind in "wsepcmaf":
        xa = [r for r in ra if r.startswith(kind + " ")]
        xb = [r for r in rb if r.startswith(kind + " ")]
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
        p.add_argument("--detail", help="rows behind the hashes of ticks FROM-TO")
        p.add_argument("--work", default="/data/tick-digest", help="where the run directories go")
    sub.choices["check"].add_argument("--bin2", help="second build (default: --bin again)")
    sub.choices["check"].add_argument("--repeat", type=int, default=1,
                                      help="runs of the second build, each compared with the first")
    sub.choices["run"].add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args()

    if args.cmd == "compare":
        diff = compare(args.a, args.b)
        print("identical" if diff is None else f"first difference at tick {diff[0]}: {', '.join(diff[1])}")
        sys.exit(diff is not None)

    commands = Path(args.commands).read_text() if args.commands else DEFAULT_COMMANDS
    common = dict(ticks=args.ticks, commands=commands, seed=args.seed, gpu=args.gpu, full=args.full,
                  players=args.players, join_every=args.join_every)
    if args.cmd == "run":
        run(args.bin, args.out, detail=args.detail, **common)
        return
    work = Path(args.work)
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
