#!/usr/bin/env python3
"""The GPU world generation of this fork (crates/horizon-gpu): checks and benchmarks anyone can rerun.

    python3 tools/gpu_worldgen.py probe
    python3 tools/gpu_worldgen.py verify --lint --tests horizon-world horizon-core --report report.md
    python3 tools/gpu_worldgen.py pregen --accept-eula --variants cpu,gpu --dims overworld,the_nether,the_end
    python3 tools/gpu_worldgen.py compare-worlds WORLD_A WORLD_B -100 -100 100 100

`probe` dumps chunks with `world_probe chunk_dump` (the probe of tools/test_chunk_generation_parity.py) on
nine zones of seed 12345 in the three dimensions and compares every dump with a reference: by default the
same build on the CPU, which is enough for a change that leaves the CPU path alone; otherwise dumps made by
another build (`--refs DIR`; `--refs-probe` makes them, for example with a probe of pure upstream). The
overworld runs on the CPU (against foreign references), with everything on the GPU and in the GPU test modes;
the Nether and the End with everything on the GPU. Every GPU run must show the GPU at work (a
`GPU final_density: N chunks` line with N > 0): a stage that fell back to the CPU would compare equal and
prove nothing. The test modes compare each chunk with the CPU inside the probe; their mismatches must be 0.

`verify` is the check before a commit: optionally rustfmt and clippy -D warnings (the upstream CI's two
commands, and horizon-gpu without CUDA), then `probe`, the vanilla parity of
tools/test_chunk_generation_parity.py with everything on the GPU, and the unit tests named by `--tests`
(`crate` or `crate:filter`). A filter that matches no test fails; a test that fails in its suite but passes
alone is reported as flaky. `--report` writes the results as Markdown.

`pregen` measures `/pregen` on fresh servers. Each pass creates a server folder, starts the server,
pregenerates the same zones with the same radius and stops it, so nothing carries over between passes. It
compares builds (`--bin NAME=DIR`: folders holding horizon-server and horizon-client) and settings
(`--variants`: cpu, gpu, or NAME:VAR=VALUE+VAR=VALUE), per dimension, and alternates their order on every
repeat. Each pass gives the server's chunks a second, the CPU time per chunk and the cores busy (Linux,
/proc), and the GPU's use and memory (nvidia-smi, when there is one). `--keep` keeps the worlds, for
`compare-worlds`.

`compare-worlds` compares the block states and biomes two worlds saved, chunk by chunk (zlib, gzip,
uncompressed and LZ4 regions).

Prerequisites, as for tools/test_chunk_generation_parity.py: the game files imported and prepared
(`python3 tools/lab.py prepare ...`, then horizon-prepare). An NVIDIA GPU and its driver for the GPU runs;
no CUDA toolkit is needed (the libraries are loaded at run time). Linux for the CPU accounting of `pregen`.
"""
import argparse
import concurrent.futures
import filecmp
import gzip
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 12345
EXE = ".exe" if os.name == "nt" else ""
# The switches of the GPU stages (crates/horizon-gpu/src/lib.rs); settings such as HORIZON_GPU_DEVICE,
# HORIZON_GPU_MEMORY_MB or HORIZON_GPU_CACHE pass through untouched.
STAGES = ["HORIZON_GPU_DENSITY", "HORIZON_GPU_SURFACE", "HORIZON_GPU_SURFACE_TEST", "HORIZON_GPU_ORES",
          "HORIZON_GPU_CAVES", "HORIZON_GPU_LIGHT", "HORIZON_GPU_SECTIONS", "HORIZON_GPU_BIOMES"]
PROBE_GPU = {"HORIZON_GPU_DENSITY": "wait", "HORIZON_GPU_SURFACE": "1", "HORIZON_GPU_ORES": "1",
             "HORIZON_GPU_CAVES": "1", "HORIZON_GPU_SECTIONS": "1", "HORIZON_GPU_BIOMES": "1"}
PROBE_TEST = dict(PROBE_GPU, HORIZON_GPU_ORES="test", HORIZON_GPU_CAVES="test",
                  HORIZON_GPU_SECTIONS="test", HORIZON_GPU_BIOMES="test")
SERVER_GPU = {"HORIZON_GPU_DENSITY": "1", "HORIZON_GPU_SURFACE": "1", "HORIZON_GPU_ORES": "1",
              "HORIZON_GPU_CAVES": "1", "HORIZON_GPU_LIGHT": "1", "HORIZON_GPU_SECTIONS": "1",
              "HORIZON_GPU_BIOMES": "1"}
ZONES = [("overworld", "0,0,15,15"), ("overworld", "-1250,940,-1235,955"),
         ("overworld", "-320,-295,-305,-280"), ("overworld", "-3008,2492,-2993,2507"),
         ("the_nether", "0,0,15,15"), ("the_nether", "-300,180,-285,195"),
         ("the_end", "0,0,7,7"), ("the_end", "1000,-1200,1015,-1185"), ("the_end", "-2600,300,-2585,315")]
NOTICES = re.compile(r"disabled|not added last|without CUDA|NVRTC library|driver library|refused|panicked|^Error",
                     re.M)


def env_with(extra):
    """This process's environment without the GPU stage switches, plus `extra`."""
    env = {k: v for k, v in os.environ.items() if k not in STAGES}
    env.update(extra)
    return env


def resources(args):
    path = Path(args.resources) if args.resources else ROOT / ".local" / "horizon-client" / "26.3"
    if not (path / "facts.json").exists():
        sys.exit(f"no prepared resources in {path}: run tools/lab.py prepare, then horizon-prepare")
    return path


# ---------------------------------------------------------------------------------------------- probe

def build_probe():
    """The probe, built as tools/test_chunk_generation_parity.py builds it (so one build serves both)."""
    print("== building the probe: cargo build --release -p horizon-world --example world_probe", flush=True)
    code, outp = run_capture(["cargo", "build", "--release", "-q", "-p", "horizon-world", "--example",
                              "world_probe"])
    if code != 0:
        sys.exit("the probe does not build:\n" + "\n".join(outp.splitlines()[-40:]))
    meta = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"],
                                              cwd=ROOT, text=True))
    return Path(meta["target_directory"]) / "release" / "examples" / f"world_probe{EXE}"


def dump(probe, res, dim, zone, env, out):
    """One probe run: the dump in `out`, the stderr returned."""
    with open(out, "wb") as f:
        r = subprocess.run([str(probe), "chunk_dump", str(res), str(SEED), dim, zone], stdout=f,
                           stderr=subprocess.PIPE, env=env, text=False)
    err = r.stderr.decode("utf-8", "replace")
    if r.returncode != 0:
        err += f"\nError: the probe exited with {r.returncode}"
    return err


def ref_name(dim, zone):
    return f"{dim}.{zone}.txt"


def make_refs(probe, res, folder, jobs):
    folder.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        futures = [pool.submit(dump, probe, res, d, z, env_with({}), folder / ref_name(d, z)) for d, z in ZONES]
        errors = [f.result() for f in futures]
    bad = [e for e in errors if "Error:" in e]
    if bad:
        sys.exit("making the references failed:\n" + bad[0][-2000:])


def gpu_problems(err, mode):
    """What a GPU run's stderr says went wrong: the GPU idle, test-mode mismatches, notices."""
    problems = []
    counts = [int(n) for n in re.findall(r"GPU final_density: (\d+) chunks", err)]
    if mode in ("gpu", "test") and not any(n > 0 for n in counts):
        problems.append("the GPU did no final_density work")
    if mode == "test":
        for line in err.splitlines():
            if line.startswith("GPU ") and re.search(r"\b[1-9]\d* (?:mismatch|mismatches|differ)", line):
                problems.append(line.strip()[:200])
    problems += [line.strip()[:200] for line in err.splitlines() if NOTICES.search(line)]
    return problems


def probe(args, built=None):
    """The nine zones against the references; returns (ok, report lines)."""
    res = resources(args)
    p = Path(args.probe) if args.probe else (built or build_probe())
    work = Path(args.work) if args.work else ROOT / ".local" / "gpu-probe"
    out = work / "run"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    own_refs = args.refs == "cpu"
    if own_refs:
        refs = work / "refs-cpu"
        shutil.rmtree(refs, ignore_errors=True)
        maker = Path(args.refs_probe) if args.refs_probe else p
        print(f"== references: {maker} on the CPU", flush=True)
        make_refs(maker, res, refs, args.jobs)
        own_refs = not args.refs_probe
    else:
        refs = Path(args.refs)
    missing = [ref_name(d, z) for d, z in ZONES if not (refs / ref_name(d, z)).is_file()]
    if missing:
        sys.exit(f"missing references in {refs}: {', '.join(missing)}")
    runs = []
    for dim, zone in ZONES:
        modes = ["gpu"] if dim != "overworld" else ["gpu", "test"]
        if not own_refs:
            modes = ["cpu"] + modes
        runs += [(dim, zone, m) for m in modes]
    envs = {"cpu": env_with({}), "gpu": env_with(PROBE_GPU), "test": env_with(PROBE_TEST)}

    def one(run):
        dim, zone, mode = run
        target = out / f"{dim}.{zone}.{mode}.txt"
        err = dump(p, res, dim, zone, envs[mode], target)
        same = filecmp.cmp(target, refs / ref_name(dim, zone), shallow=False)
        target.unlink()
        (out / f"{dim}.{zone}.{mode}.err").write_text(err)
        return run, same, gpu_problems(err, mode) if mode != "cpu" else [
            line for line in err.splitlines() if "panicked" in line or line.startswith("Error")]

    t = time.time()
    # One GPU run first: it fills the PTX cache that the parallel runs then read
    results = [one(runs[0])]
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        results += list(pool.map(one, runs[1:]))
    ok, lines = True, []
    for dim, zone in ZONES:
        cells, notes = [], []
        for (d, z, mode), same, problems in results:
            if (d, z) != (dim, zone):
                continue
            cells.append(f"{mode} {'IDENTICAL' if same else 'DIFFERENT'}")
            ok &= same and not problems
            notes += [f"   !! {mode}: {problem}" for problem in problems]
        lines += [f"== {dim} {zone}: {', '.join(cells)}", *notes]
    stats = sorted({line.strip() for f in out.glob("*.gpu.err") for line in f.read_text().splitlines()
                    if line.startswith("GPU final_density: ")})
    lines.append(f"== probe: {len(runs)} runs in {time.time() - t:.0f} s, {args.jobs} at a time")
    lines += [f"   {s[:220]}" for s in stats[:3]]
    for line in lines:
        print(line, flush=True)
    return ok, lines


# --------------------------------------------------------------------------------------------- verify

def run_capture(cmd, env=None, cwd=ROOT):
    r = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       errors="replace")
    return r.returncode, r.stdout


def lint():
    ok, lines = True, []
    code, outp = run_capture(["cargo", "fmt", "--all", "--", "--check", "-l"])
    files = [x for x in outp.splitlines() if x.strip().endswith(".rs")]
    lines.append(f"== rustfmt: {len(files)} files to reformat")
    ok &= not files
    for args in (["--workspace", "--all-targets"], ["-p", "horizon-client", "--all-targets", "--no-default-features"],
                 ["-p", "horizon-gpu", "--all-targets"]):
        code, outp = run_capture(["cargo", "clippy", *args, "--locked", "--", "-D", "warnings"])
        lines.append(f"== clippy {' '.join(args)} -D warnings: {'clean' if code == 0 else f'exit {code}'}")
        if code != 0:
            ok = False
            lines += ["   " + x for x in outp.splitlines() if re.match(r"^(error|warning)(\[|:)", x)
                      and "lints" not in x][:15]
    for line in lines:
        print(line, flush=True)
    return ok, lines


def tests(specs, res):
    env = env_with({"RUST_MIN_STACK": "16777216", "HORIZON_TEST_RESOURCES": str(res)})
    ok, lines = True, []
    for spec in specs:
        crate, _, filt = spec.partition(":")
        code, outp = run_capture(["cargo", "test", "-q", "-p", crate, "--lib", *([filt] if filt else [])], env=env)
        passed = sum(int(n) for n in re.findall(r"^test result: \w+\. (\d+) passed", outp, re.M))
        failed_n = sum(int(n) for n in re.findall(r"; (\d+) failed", outp))
        ignored = sum(int(n) for n in re.findall(r"; (\d+) ignored", outp))
        line = f"== test {crate} {filt or '(all)'}: {passed} passed, {failed_n} failed, {ignored} ignored"
        if re.search(r"^error(\[E\d+\])?: could not compile|^error\[E", outp, re.M):
            ok = False
            lines.append(line + " -- does not compile")
            lines += ["   " + x for x in outp.splitlines() if x.startswith("error")][:10]
            continue
        if filt and passed + failed_n == 0:
            ok = False
            lines.append(line + " -- the filter matched no test (wrong crate? save.rs and most of "
                         "horizon-client/src are compiled in horizon-client-lib)")
            continue
        failures = re.findall(r"^---- (\S+) stdout ----", outp, re.M)
        flaky = []
        for name in failures:
            c, o = run_capture(["cargo", "test", "-q", "-p", crate, "--lib", name, "--", "--exact"], env=env)
            if c == 0 and re.search(r"^test result: ok\. 1 passed", o, re.M):
                flaky.append(name)
        real = [f for f in failures if f not in flaky]
        if code != 0 and (real or not failures):
            ok = False
        lines.append(line + (f" ({len(flaky)} flaky: passes alone)" if flaky else ""))
        lines += [f"   !! fails: {f}" for f in real] + [f"   flaky: {f}" for f in flaky]
    for line in lines:
        print(line, flush=True)
    return ok, lines


def parity():
    print("== vanilla parity, everything on the GPU: tools/test_chunk_generation_parity.py --seeds 1", flush=True)
    code, outp = run_capture([sys.executable, "tools/test_chunk_generation_parity.py", "--seeds", "1",
                              "--summary"], env=env_with(PROBE_GPU))
    lines = [x for x in outp.splitlines() if x.startswith(("Compared", "Differences", "Traceback"))]
    ok = code == 0 and any(re.match(r"Differences: 0 chunks, 0 blocks, 0 biome cells, 0 block entities", x)
                           for x in lines)
    lines = [f"== {x}" for x in lines] or ["== parity: no result"]
    for line in lines:
        print(line, flush=True)
    return ok, lines


def git_head():
    code, head = run_capture(["git", "rev-parse", "--short=10", "HEAD"])
    dirty = run_capture(["git", "diff", "HEAD", "--quiet"])[0] != 0
    return head.strip() + (" with uncommitted changes" if dirty else "")


def verify(args):
    t = time.time()
    sections = []
    verdict = True
    if args.lint:
        ok, lines = lint()
        verdict &= ok
        sections.append(("rustfmt and clippy", ok, lines))
    built = None if args.probe else build_probe()
    ok, lines = probe(args, built)
    verdict &= ok
    sections.append(("probe: 9 zones, 3 dimensions", ok, lines))
    ok, lines = parity()
    verdict &= ok
    sections.append(("vanilla parity, everything on the GPU", ok, lines))
    if args.tests:
        ok, lines = tests(args.tests, resources(args))
        verdict &= ok
        sections.append(("unit tests", ok, lines))
    summary = f"== verdict: {'OK' if verdict else 'FAILED'} in {time.time() - t:.0f} s at {git_head()}"
    print(summary, flush=True)
    if args.report:
        md = [f"# GPU worldgen verification", "",
              f"{time.strftime('%Y-%m-%d %H:%M')}, commit {git_head()}: **{'OK' if verdict else 'FAILED'}**.", ""]
        for title, ok, lines in sections:
            md += [f"## {title}: {'OK' if ok else 'FAILED'}", "", "```", *lines, "```", ""]
        Path(args.report).write_text("\n".join(md))
        print(f"== report: {args.report}")
    return 0 if verdict else 1


# --------------------------------------------------------------------------------------------- pregen

def variant(spec):
    if spec == "cpu":
        return "cpu", {}
    if spec == "gpu":
        return "gpu", dict(SERVER_GPU)
    name, _, assignments = spec.partition(":")
    env = {}
    for a in filter(None, assignments.split("+")):
        k, _, v = a.partition("=")
        env[k] = v
    return name, env


def edit_toml(path, section, key, value):
    """Sets `key = value` in `[section]` of the server.toml that `import` wrote (comments kept)."""
    text = path.read_text()
    line = f"{key} = {value}"
    pattern = re.compile(rf"^{re.escape(key)} = .*$", re.M)
    header = f"[{section}]\n"
    if header in text:
        start = text.index(header) + len(header)
        end = text.find("\n[", start)
        end = len(text) if end < 0 else end + 1
        body = text[start:end]
        body = pattern.sub(line, body, count=1) if pattern.search(body) else line + "\n" + body
        text = text[:start] + body + text[end:]
    else:
        text = text.rstrip("\n") + f"\n\n{header}{line}\n"
    path.write_text(text)


def game_pid(folder, strict=False):
    """The game process of the server in `folder` (Linux); without `strict`, else the only one running."""
    found = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if (d / "comm").read_text().strip() == "horizon-client":
                found.append((os.readlink(d / "cwd"), int(d.name)))
        except OSError:
            continue
    mine = [pid for cwd, pid in found if Path(cwd).resolve() == folder.resolve()]
    if strict:
        return mine[0] if mine else None
    return (mine or ([found[0][1]] if len(found) == 1 else []) or [None])[0]


def cpu_seconds(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def server(binary, folder, *cmd, check=True, env=None):
    r = subprocess.run([str(binary / f"horizon-server{EXE}"), "--dir", str(folder), *cmd], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"horizon-server {' '.join(cmd)}: {r.stdout[-1500:]}")
    return r


def pregen_pass(args, binary, env, dim, folder):
    """One fresh server: start, pregenerate each zone, stop. One record per zone."""
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "server.properties").write_text("\n".join([
        f"level-seed={SEED}", "online-mode=false", "white-list=false", "difficulty=easy", "gamemode=survival",
        "view-distance=10", "simulation-distance=10", "spawn-protection=0", ""]))
    (folder / "eula.txt").write_text("eula=true\n")
    server(binary, folder, "import", str(folder / "server.properties"))
    toml = folder / "server.toml"
    edit_toml(toml, "properties", "horizon-pregen", "true")
    if args.compression:
        edit_toml(toml, "properties", "region-file-compression", f'"{args.compression}"')
    edit_toml(toml, "panel", "bind", '""')
    if args.workers is not None:
        edit_toml(toml, "engine", "worker_threads", str(args.workers))
    # The game inherits the environment of `start`: the variant's GPU switches only
    server(binary, folder, "start", env=env_with(env))
    records = []
    try:
        for _ in range(300):
            if server(binary, folder, "cmd", "list", check=False).returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("the server did not answer in 5 minutes")
        pid = game_pid(folder)
        log = folder / "logs" / "latest.log"
        for x, z in args.zones:
            seen = len(log.read_text(errors="replace").splitlines())
            smi = None
            if shutil.which("nvidia-smi"):
                smi_out = tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv")
                smi = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits", "-lms", "1000"], stdout=smi_out,
                                       stderr=subprocess.DEVNULL)
            c0 = cpu_seconds(pid) if pid and os.path.exists("/proc") else None
            t0 = time.time()
            command = f"pregen start {args.radius} square {x} {z}"
            if dim != "overworld":
                command = f"execute in minecraft:{dim} run {command}"
            server(binary, folder, "cmd", command)
            done = None
            while done is None:
                time.sleep(0.5)
                for line in log.read_text(errors="replace").splitlines()[seen:]:
                    m = re.search(r"Done\. Pregenerating \S+: (\d+) of \d+ chunks \([\d.]+%\), ([\d.]+) chunks/s", line)
                    if m:
                        done = (int(m.group(1)), float(m.group(2)))
                if time.time() - t0 > args.timeout:
                    raise RuntimeError(f"/pregen took more than {args.timeout} s")
            wall = time.time() - t0
            cpu = cpu_seconds(pid) - c0 if c0 is not None else None
            gpu_use = gpu_mem = None
            if smi:
                smi.terminate()
                smi.wait()
                smi_out.close()
                rows = [r.split(",") for r in Path(smi_out.name).read_text().splitlines() if "," in r]
                os.unlink(smi_out.name)
                if rows:
                    gpu_use = sum(float(r[0]) for r in rows) / len(rows)
                    gpu_mem = max(float(r[1]) for r in rows)
            n, rate = done
            records.append({"dim": dim, "zone": f"{x},{z}", "chunks": n, "chunks_per_s": rate, "wall_s": wall,
                            "cpu_ms_per_chunk": cpu * 1000 / n if cpu is not None else None,
                            "cores": cpu / wall if cpu is not None else None,
                            "gpu_use_pct": gpu_use, "gpu_mem_mib": gpu_mem})
    finally:
        server(binary, folder, "stop", check=False)
        for _ in range(60):
            if game_pid(folder, strict=True) is None:
                break
            time.sleep(1)
    return records


def machine():
    info = {"commit": git_head(), "date": time.strftime("%Y-%m-%d %H:%M"), "cpu": None, "gpu": None,
            "cores": os.cpu_count()}
    try:
        info["cpu"] = re.search(r"^model name\s*: (.*)$", Path("/proc/cpuinfo").read_text(), re.M).group(1)
    except (OSError, AttributeError):
        pass
    if shutil.which("nvidia-smi"):
        info["gpu"] = run_capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                   "--format=csv,noheader"])[1].strip()
    return info


def pregen(args):
    if not args.accept_eula:
        sys.exit("pregen creates Minecraft servers: pass --accept-eula to accept the Minecraft EULA "
                 "(https://aka.ms/MinecraftEULA) for them")
    bins = []
    for spec in args.bin or [f"current={ROOT / 'target' / 'release'}"]:
        name, _, folder = spec.partition("=")
        folder = Path(folder).expanduser().resolve()
        if not (folder / f"horizon-server{EXE}").exists() or not (folder / f"horizon-client{EXE}").exists():
            sys.exit(f"{folder}: no horizon-server and horizon-client (cargo build --release -p horizon-game "
                     "--bin horizon-client -p horizon-server)")
        bins.append((name, folder))
    variants = [variant(v) for v in args.variants.split(",")]
    dims = args.dims.split(",")
    work = Path(args.work) if args.work else ROOT / ".local" / "gpu-bench"
    configs = [(b, v) for b in bins for v in variants]
    info = machine()
    print(f"== {info['date']}, {info['commit']}; CPU {info['cpu']} ({info['cores']} threads); GPU {info['gpu']}")
    print(f"== zones {args.zones}, radius {args.radius} blocks, {args.repeat} repeat(s)", flush=True)
    records = []
    for rep in range(args.repeat):
        order = configs if rep % 2 == 0 else list(reversed(configs))
        for (bname, bdir), (vname, venv) in order:
            for dim in dims:
                folder = work / f"{bname}-{vname}-{dim}-{rep}"
                for r in pregen_pass(args, bdir, venv, dim, folder):
                    r.update({"bin": bname, "variant": vname, "repeat": rep})
                    records.append(r)
                    cpu = f"{r['cpu_ms_per_chunk']:.2f} ms CPU/chunk, {r['cores']:.2f} cores" if r["cores"] else ""
                    gpu = f", GPU {r['gpu_use_pct']:.0f} % {r['gpu_mem_mib']:.0f} MiB" if r["gpu_use_pct"] is not None else ""
                    print(f"{bname:>10} {vname:>6} {dim:>11} {r['zone']:>18} pass {rep}: "
                          f"{r['chunks_per_s']:7.1f} chunks/s, {cpu}{gpu}", flush=True)
                if not args.keep:
                    shutil.rmtree(folder, ignore_errors=True)
    print("\n== means")
    base = {}
    for (bname, _), (vname, _) in configs:
        for dim in dims:
            rs = [r for r in records if (r["bin"], r["variant"], r["dim"]) == (bname, vname, dim)]
            rate = sum(r["chunks_per_s"] for r in rs) / len(rs)
            cpus = [r["cpu_ms_per_chunk"] for r in rs if r["cpu_ms_per_chunk"] is not None]
            cpu = sum(cpus) / len(cpus) if cpus else None
            ref = base.setdefault(dim, (rate, cpu))
            rel = f" ({(rate / ref[0] - 1) * 100:+.1f} %)" if ref[0] != rate else ""
            cpu_text = f", {cpu:.2f} ms CPU/chunk" if cpu is not None else ""
            print(f"{bname:>10} {vname:>6} {dim:>11}: {rate:7.1f} chunks/s{rel}{cpu_text} over {len(rs)} zone passes")
    if args.json:
        Path(args.json).write_text(json.dumps({"machine": info, "args": vars(args), "passes": records}, indent=1,
                                              default=str))
        print(f"== {args.json}")
    return 0


# ------------------------------------------------------------------------------------- compare-worlds

def nbt(b, i, t):
    """An NBT payload of type t at i: (value, next index). Compounds become dicts, lists lists."""
    if t == 1:
        return b[i], i + 1
    if t == 2:
        return struct.unpack_from(">h", b, i)[0], i + 2
    if t == 3:
        return struct.unpack_from(">i", b, i)[0], i + 4
    if t == 4:
        return struct.unpack_from(">q", b, i)[0], i + 8
    if t == 5:
        return struct.unpack_from(">f", b, i)[0], i + 4
    if t == 6:
        return struct.unpack_from(">d", b, i)[0], i + 8
    if t in (7, 11, 12):
        n = struct.unpack_from(">i", b, i)[0] * {7: 1, 11: 4, 12: 8}[t]
        return bytes(b[i + 4:i + 4 + n]), i + 4 + n
    if t == 8:
        n = struct.unpack_from(">H", b, i)[0]
        return b[i + 2:i + 2 + n].decode("utf-8", "replace"), i + 2 + n
    if t == 9:
        et, n = b[i], struct.unpack_from(">i", b, i + 1)[0]
        i += 5
        out = []
        for _ in range(n):
            v, i = nbt(b, i, et)
            out.append(v)
        return out, i
    if t == 10:
        d = {}
        while True:
            et = b[i]
            i += 1
            if et == 0:
                return d, i
            n = struct.unpack_from(">H", b, i)[0]
            name = b[i + 2:i + 2 + n].decode("utf-8", "replace")
            d[name], i = nbt(b, i + 2 + n, et)
    raise ValueError(f"NBT type {t}")


def lz4_block(src, size):
    """An LZ4 block (lz4-java's LZ4BlockOutputStream writes them) decompressed."""
    out, i, n = bytearray(), 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        out += src[i:i + lit]
        i += lit
        if i >= n:
            break
        off = src[i] | src[i + 1] << 8
        i += 2
        length = token & 15
        if length == 15:
            while True:
                b = src[i]
                i += 1
                length += b
                if b != 255:
                    break
        length += 4
        start = len(out) - off
        if off >= length:
            out += out[start:start + length]
        else:
            for k in range(length):
                out.append(out[start + k])
    return bytes(out[:size])


def lz4_java(data):
    """lz4-java's block stream: blocks of `LZ4Block`, a token, sizes and a checksum, then the bytes."""
    out, i = bytearray(), 0
    while i + 21 <= len(data):
        if data[i:i + 8] != b"LZ4Block":
            raise ValueError("not an LZ4Block stream")
        method = data[i + 8] & 0xF0
        compressed, raw = struct.unpack_from("<ii", data, i + 9)
        i += 21
        if raw == 0:
            break
        block = data[i:i + compressed]
        i += compressed
        out += block if method == 0x10 else lz4_block(block, raw)
    return bytes(out)


def region_chunks(path):
    data = path.read_bytes()
    out = {}
    for k in range(1024):
        loc = struct.unpack_from(">I", data, 4 * k)[0]
        if loc == 0:
            continue
        o = (loc >> 8) * 4096
        length = struct.unpack_from(">I", data, o)[0]
        kind, payload = data[o + 4], data[o + 5:o + 4 + length]
        if kind & 128:
            out[k] = None  # stored in a separate .mcc file: not compared
            continue
        out[k] = {1: gzip.decompress, 2: zlib.decompress, 3: bytes, 4: lz4_java}.get(kind, lambda _: None)(payload)
    return out


def sections(body):
    root, _ = nbt(body, 3 + struct.unpack_from(">H", body, 1)[0], body[0])
    return ([repr(s.get("block_states")) for s in root.get("sections", [])],
            [repr(s.get("biomes")) for s in root.get("sections", [])])


def region_folder(path, dim):
    path = Path(path)
    for candidate in [path / "dimensions" / "minecraft" / dim / "region", path / "region", path]:
        if candidate.is_dir() and any(candidate.glob("r.*.mca")):
            return candidate
    for candidate in path.glob(f"*/dimensions/minecraft/{dim}/region"):
        return candidate
    sys.exit(f"no region files of {dim} under {path}")


def compare_worlds(args):
    a, b = region_folder(args.a, args.dim), region_folder(args.b, args.dim)
    total = blocks = biomes = skipped = 0
    examples = []
    for name in sorted({p.name for p in a.glob("r.*.mca")} & {p.name for p in b.glob("r.*.mca")}):
        _, rx, rz, _ = name.split(".")
        rx, rz = int(rx), int(rz)
        if not (args.x0 >> 5 <= rx <= args.x1 >> 5 and args.z0 >> 5 <= rz <= args.z1 >> 5):
            continue
        ca, cb = region_chunks(a / name), region_chunks(b / name)
        for k in sorted(set(ca) & set(cb)):
            cx, cz = rx * 32 + (k & 31), rz * 32 + (k >> 5)
            if not (args.x0 <= cx <= args.x1 and args.z0 <= cz <= args.z1):
                continue
            if ca[k] is None or cb[k] is None:
                skipped += 1
                continue
            total += 1
            (ba, ma), (bb, mb) = sections(ca[k]), sections(cb[k])
            if ba != bb:
                blocks += 1
                if len(examples) < 10:
                    examples.append((cx, cz))
            biomes += ma != mb
    print(f"{total} chunks compared ({args.dim}): {blocks} with different blocks, {biomes} with different biomes"
          + (f", {skipped} not read" if skipped else "") + (f"; first: {examples}" if examples else ""))
    return 0 if blocks == biomes == 0 else 1


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def probe_args(p):
        p.add_argument("--refs", default="cpu", help="cpu (this build on the CPU, default) or a folder of dumps")
        p.add_argument("--refs-probe", help="with --refs cpu: make the references with this probe instead")
        p.add_argument("--probe", help="a world_probe binary (default: build it)")
        p.add_argument("--resources", help="the prepared resources (default .local/horizon-client/26.3)")
        p.add_argument("--jobs", type=int, default=5, help="probe runs at a time (the probe is single-threaded)")
        p.add_argument("--work", help="where the dumps go (default .local/gpu-probe)")

    p = sub.add_parser("probe", help="the 9 zones against references")
    probe_args(p)
    p = sub.add_parser("verify", help="lint, probe, parity and tests, with a verdict")
    probe_args(p)
    p.add_argument("--lint", action="store_true", help="rustfmt and clippy -D warnings first")
    p.add_argument("--tests", nargs="*", default=[], metavar="CRATE[:FILTER]")
    p.add_argument("--report", help="write the results as Markdown")
    p = sub.add_parser("pregen", help="/pregen on fresh servers: builds and settings compared")
    p.add_argument("--accept-eula", action="store_true", help="accept the Minecraft EULA for the servers made")
    p.add_argument("--bin", action="append", metavar="NAME=DIR", help="a build to measure (default: target/release)")
    p.add_argument("--variants", default="cpu,gpu", help="cpu, gpu or NAME:VAR=VALUE+VAR=VALUE, comma-separated")
    p.add_argument("--dims", default="overworld", help="overworld, the_nether, the_end, comma-separated")
    p.add_argument("--zones", default="7000000,-7020000;-7030000,7040000", help="block centres X,Z;X,Z")
    p.add_argument("--radius", type=int, default=1600, help="blocks around each centre (1600: 40,401 chunks)")
    p.add_argument("--repeat", type=int, default=1, help="passes of every configuration, order alternated")
    p.add_argument("--workers", type=int, help="[engine] worker_threads (default: the server's)")
    p.add_argument("--compression", help="region-file-compression (deflate, lz4, none...; default: the server's)")
    p.add_argument("--timeout", type=int, default=1800, help="seconds allowed for one zone")
    p.add_argument("--work", help="where the servers are made (default .local/gpu-bench)")
    p.add_argument("--keep", action="store_true", help="keep the worlds (for compare-worlds)")
    p.add_argument("--json", help="write every pass as JSON")
    p = sub.add_parser("compare-worlds", help="blocks and biomes of two saved worlds, chunk by chunk")
    p.add_argument("a")
    p.add_argument("b")
    for name in ("x0", "z0", "x1", "z1"):
        p.add_argument(name, type=int, help="chunk coordinates of the compared area")
    p.add_argument("--dim", default="overworld")
    args = ap.parse_args()
    if args.command == "pregen":
        args.zones = [tuple(int(v) for v in z.split(",")) for z in args.zones.split(";")]
    if args.command == "probe":
        return 0 if probe(args)[0] else 1
    if args.command == "verify":
        return verify(args)
    if args.command == "pregen":
        return pregen(args)
    return compare_worlds(args)


if __name__ == "__main__":
    sys.exit(main())
