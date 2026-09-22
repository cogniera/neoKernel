"""Terminal entry points. Remote work occurs only after an explicit command."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from agent.package import package
from .guard import check
from .accounting import GPU_TIMEOUT_S, SpendLedger, SpendLimit, record_stop, estimate_usd
from .calibration import equivalent_estimates, load_calibration, refresh_host_factors
from .judge import OFFICIAL
from .schema import PUBLIC, select_workloads
from .storage import (ROOT, RESULTS, Budget, append_log, git_sha, read_log, save_run, seed_native,
                      timestamp, write_json)


def table(title: str, columns: list[str], rows: list[list]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        grid = Table(title=title)
        for column in columns:
            grid.add_column(column)
        for row in rows:
            grid.add_row(*(str(value) for value in row))
        Console().print(grid)
    except ImportError:
        print(title)
        print(" | ".join(columns))
        for row in rows:
            print(" | ".join(map(str, row)))


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def report(result: dict, correctness_only=False):
    print(f"{'PASS' if result['eligible'] else 'FAIL'}: geomean={fmt(result.get('geomean_tps'))} tok/s; GPU seconds={result.get('gpu_seconds', 0):.2f}")
    if correctness_only:
        table("Correctness", ["workload", "pass", "first bad [sample,seq,token]", "margin", "near ties", "failure"],
              [[r["name"], r["passed"], r["correctness"]["first_bad_position"], r["correctness"]["margin"],
                r["correctness"]["near_tie_count"], r["failure_code"]] for r in result["workloads"]])
    else:
        if result.get("cpu_diagnostics"):
            print("CPU diagnostics: " + json.dumps(result["cpu_diagnostics"]))
        table("Pipe and child CPU diagnostics (ms; CPU excludes prefill)",
              ["workload", "pipe mean", "pipe max", "CPU mean", "CPU max", "order"],
              [[r["name"], fmt(r.get("pipe_overhead_ms", {}).get("mean"), 4),
                fmt(r.get("pipe_overhead_ms", {}).get("max"), 4),
                fmt(r.get("cpu_step_ms", {}).get("mean"), 4), fmt(r.get("cpu_step_ms", {}).get("max"), 4),
                r.get("measurement_order", "")] for r in result["workloads"]])
        if result.get("native"):
            table("Local native reference; host correction = native / official", ["workload", "native tok/s", "host factor", "native TTFT ms", "native TPOT ms"],
                  [[name, fmt(n["tps"]), fmt(n["tps"]/OFFICIAL[name]["tps"], 4) if name in OFFICIAL else "n/a",
                    fmt(n["ttft_median"]*1000), fmt(n["tpot_median"]*1000)] for name, n in result["native"].items()])
        calibration = load_calibration() if result.get("gpu_tier") == "H100" else None
        if calibration:
            estimates = [(r, equivalent_estimates(r, calibration)) for r in result["workloads"]]
            table("Dryft-equivalent estimates (baseline ratios; not gate inputs)",
                  ["workload", "local tok/s", "Dryft-equivalent tok/s (estimate)"],
                  [[r["name"], fmt(r["tps"]), fmt(e["tps"] if e else None)] for r, e in estimates])
        table("Benchmark", ["workload", "tok/s", "TTFT ratio", "TPOT ratio", "spread", "memory", "gate"],
              [[r["name"], fmt(r["tps"]), fmt(r["ttft_ratio"]), fmt(r["tpot_ratio"]),
                fmt(r["spread"]), fmt(r["peak_mem_frac"]), r["failure_code"] or "pass"] for r in result["workloads"]])
        table("Physics floors (estimates)", ["workload", "step ms", "prefill ms", "decode ms", "floor tok/s", "measured/floor", "speculative"],
              [[r["name"], fmt(r["floors"].get("step_floor_s", 0)*1000), fmt(r["floors"].get("prefill_floor_s", 0)*1000),
                fmt(r["floors"].get("decode_floor_s", 0)*1000), fmt(r["floors"].get("floor_tps")),
                fmt(r["floors"].get("measured_to_floor_ratio")), r["floors"].get("speculative_relaxed", False)] for r in result["workloads"]])


def show_profile(result: dict):
    print(f"Profile: step={result['step_wall_ms']:.3f} ms; kernels={result['sum_kernel_ms']:.3f} ms; gap={result['gap_ms']:.3f} ms")
    table("CUDA kernels", ["kernel", "calls", "duration ms", "bytes", "HBM fraction"],
          [[r["name"], r["calls"], fmt(r["duration_ms"], 4), r["bytes_moved"], fmt(r["hbm_peak_fraction"])] for r in result["kernels"]])
    print(result["note"])
    print('Physics floors (estimates): ' + json.dumps(result['floors']))


def gauge(result: dict, width=70):
    from rich.console import Console
    from rich.text import Text
    wall = result["step_wall_ms"]
    floor = result["floors"]["step_floor_s"]*1000
    scale = max(wall, floor, .001)
    bar = Text()
    colors = ["cyan", "green", "magenta", "yellow", "blue", "red"]
    used = 0
    for i, row in enumerate(result["kernels"]):
        cells = min(width-used, round(row["duration_ms"] / scale * width))
        bar.append("█"*max(0, cells), style=colors[i % len(colors)])
        used += cells
    bar.append("█" * max(0, width-used), style="grey50")
    Console().print(bar)
    print(" " * min(width-1, int(floor/scale*width)) + "| weight floor")
    print(f"step {wall:.3f} ms | sum kernels {result['sum_kernel_ms']:.3f} ms | gap {result['gap_ms']:.3f} ms | floor {floor:.3f} ms")


class Remote:
    """One Modal app context can serve the complete sweep or agent session."""
    def __init__(self, budget: Budget | None = None, *, smoke_check=False):
        self.budget = budget
        self.native = seed_native()
        self.spend = SpendLedger()
        self.smoke_check = smoke_check

    def __enter__(self):
        try:
            from . import modal_app
        except ImportError as e:
            raise RuntimeError("Remote commands require: pip install -r neokernel/requirements.txt; then configure Modal") from e
        self.api = modal_app
        self.context = modal_app.app.run()
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        return self.context.__exit__(*args)

    def reserve(self, seconds):
        if self.budget:
            self.budget.reserve(seconds)

    def charge(self, seconds):
        if self.budget:
            self.budget.charge(seconds)

    def dispatch(self, tier, operation, call, **record_options):
        """Run one remote call against the GPU budget; a call that raises is billed at its timeout."""
        timeout_s = record_options.get("timeout_s", GPU_TIMEOUT_S)
        self.reserve(timeout_s)
        started = time.perf_counter()
        try:
            result = call()
        except BaseException as exc:
            self.charge(timeout_s)
            # A run that dies writes no result, so the newest file under runs/
            # would still be the previous attempt and read as this one's. Leave
            # a record of the attempt so freshness can never be assumed.
            try:
                write_json(RESULTS / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
                                               + f"_{git_sha()[:12]}.failed.json"),
                           {"ts": timestamp(), "sha": git_sha(), "tier": tier, "operation": operation,
                            "eligible": False, "failed": True, "workloads": [],
                            "error_type": type(exc).__name__, "error": str(exc)[:2000],
                            "elapsed_s": time.perf_counter() - started})
            except Exception:
                pass  # Never mask the original failure with a bookkeeping error.
            try:
                self.spend.record(tier, started, None, operation, False, **record_options)
            except SpendLimit:
                pass  # Preserve the original harness failure and its traceback.
            raise
        self.charge(result.get("gpu_seconds", 0))
        return started, result

    def bench(self, payload, workloads, samples=3, correctness_only=False, refresh_native=False,
              transport=None, prompt_seed=None):
        smoke = self.smoke_check and correctness_only
        if smoke and list(workloads) != [PUBLIC[0]]:
            raise ValueError('Smoke check accepts only public-0')
        tier = "L4" if correctness_only else "H100"
        operation = "check" if correctness_only else "bench"
        record_options = {'idle_s': 0, 'timeout_s': 60} if smoke else {}
        if smoke:
            self.spend.reserve_amount('L4', estimate_usd('L4', 120), 'bounded public-0 check')
        else:
            self.spend.reserve(tier)
        shapes = [asdict(w) for w in workloads]
        if correctness_only:
            endpoint = self.api.check_smoke_remote if smoke else self.api.check_remote
            call = ((lambda: endpoint.remote(payload, shapes, self.native)) if smoke else
                    (lambda: endpoint.remote(payload, shapes, self.native, prompt_seed)))
        else:
            call = lambda: self.api.judge_remote.remote(
                payload, shapes, samples, refresh_native, self.native, False,
                transport=transport or (load_calibration() or {}).get("transport", "json"),
                prompt_seed=prompt_seed)
        started, result = self.dispatch(tier, operation, call, **record_options)
        self.native = result["native"]
        if result.get("prompt_seed") is not None:
            print(f"Prompt seed {result['prompt_seed']}; replay with --prompt-seed {result['prompt_seed']}", flush=True)
        save_run(result, payload)
        self.spend.record(tier, started, result["gpu_seconds"], operation, result["eligible"], **record_options)
        if not correctness_only:
            refresh_host_factors(result)
        return result

    def unit_tests(self, sources):
        """Repair-stage kernel and handoff tests on L4; the result is diagnostic, never a keep input."""
        self.spend.reserve("L4")
        started, result = self.dispatch("L4", "unit_tests", lambda: self.api.unit_test_remote.remote(sources))
        self.spend.record("L4", started, result.get("gpu_seconds", 0), "unit_tests", result["passed"])
        return result

    def profile(self, payload, workload):
        self.spend.reserve("L4")
        started, result = self.dispatch("L4", "profile", lambda: self.api.profile_remote.remote(payload, asdict(workload)))
        write_json(RESULTS / "profile.json", result)
        self.spend.record("L4", started, result.get("gpu_seconds", 0), "profile", True)
        return result


def engine_at(ref: str, destination: Path) -> Path:
    if ref == "current":
        shutil.copytree(ROOT / "engine", destination, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        resolved = subprocess.run(["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip()
        names = subprocess.run(["git", "ls-tree", "-r", "--name-only", resolved, "--", "engine/"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.splitlines()
        for name in names:
            path = destination / Path(name).relative_to("engine")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.run(["git", "show", f"{resolved}:{name}"], cwd=ROOT,
                                            capture_output=True, check=True).stdout)
    check(destination)
    return destination


def race(remote, a, b, workload):
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    streams = ["", ""]
    elapsed = [0.0, 0.0]
    profiles = {}
    print("Race: identical prompts, sequential GPU runs to avoid resource contention; displaying sequence 0.")
    with tempfile.TemporaryDirectory() as tmp:
        payloads = [package(engine_at(ref, Path(tmp)/str(i))) for i, ref in enumerate([a, b])]
        with Live(refresh_per_second=12) as live:
            for event in remote.api.race_remote.remote_gen(payloads, asdict(workload)):
                if event["kind"] == "error":
                    raise RuntimeError(event["message"])
                if event["kind"] == "step":
                    side = event["side"]
                    streams[side] += event["text"][0]
                    elapsed[side] = event["elapsed_ms"]
                elif event["kind"] == "profile":
                    profiles[event["side"]] = event["profile"]
                elif event["kind"] == "done":
                    print(f"GPU seconds: {event['gpu_seconds']:.2f}")
                grid = Table()
                grid.add_column(f"A: {a} ({elapsed[0]:.1f} ms)")
                grid.add_column(f"B: {b} ({elapsed[1]:.1f} ms)")
                grid.add_row(Text(streams[0][-2000:]), Text(streams[1][-2000:]))
                live.update(grid)
        for side, profile in profiles.items():
            print(f"{'A' if side == 0 else 'B'} decode breakdown")
            gauge(profile)


def parser():
    root = argparse.ArgumentParser(description="neoKernel v1: local research harness, explicit remote commands")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("report", help="generate docs/RESULTS.md and docs/log.md from saved local evidence, offline")
    commands.add_parser("guard", help="lint only, offline")
    commands.add_parser("download-weights", help="explicit one-time Modal checkpoint download")
    for name in ["check", "bench", "freeze", "auto", "sweep"]:
        cmd = commands.add_parser(name)
        cmd.add_argument("--workloads", default="public-0,public-2" if name == 'sweep' else "all" if name == 'auto' else "public")
        if name in {"check", "bench"}:
            cmd.add_argument("--prompt-seed", type=int, default=None,
                             help="replay a previous run's prompts; the seed is printed and saved with every run")
        if name == "bench":
            cmd.add_argument("--samples", type=int, default=3)
            cmd.add_argument("--refresh-native", action="store_true")
            cmd.add_argument("--transport", choices=["json", "binary"], default=None)
        if name == "freeze":
            cmd.add_argument("--out", type=Path, required=True)
        if name in {"auto", "sweep"}:
            cmd.add_argument("--max-gpu-minutes", type=float, default=10000)
            cmd.add_argument("--steps", type=int, default=1 if name == "auto" else 10)
            cmd.add_argument('--resume', action='store_true')
        if name == "auto":
            cmd.add_argument("--model", default="zai-org/GLM-5.2")
            cmd.add_argument("--items")
        if name == "sweep":
            cmd.add_argument("--random", action="store_true")
            cmd.add_argument("--seed", type=int, default=0)
            cmd.add_argument("--wire-rmsnorm", action="store_true", help="stage a measured RMSNorm integration before sweeping")
    for name in ["profile", "gauge", "race"]:
        cmd = commands.add_parser(name)
        cmd.add_argument("--workload", default="public-0")
        if name == "race":
            cmd.add_argument("--a", required=True)
            cmd.add_argument("--b", required=True)
    cmd = commands.add_parser("log")
    cmd.add_argument("--last", type=int, default=15)
    cmd.add_argument("--kept", action="store_true")
    return root


def main(argv=None) -> int:
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parser().parse_args(argv)
    try:
        if args.command == "report":
            from .report import generate
            for path in generate():
                print(f"Generated {path}")
            return 0
        if args.command == "log":
            records = [r for r in read_log() if not args.kept or r["kept"]]
            print(f"Log: {len(records)} matching experiments")
            table("Experiments", ["id", "time", "sha", "proposer", "item", "geomean", "delta %", "kept", "note"],
                  [[r["id"], r["ts"], r["sha"][:8], r["proposer"], r["item"], fmt(r["geomean_tps"]),
                    fmt(r["delta_pct"]), r["kept"], r["note"]] for r in (records[-args.last:] if args.last > 0 else [])])
            return 0
        if args.command not in {'auto', 'sweep'}:
            check(ROOT / "engine")
        if args.command == "guard":
            print("PASS: engine source guard")
            return 0
        if hasattr(args, "steps") and args.steps < 1:
            raise ValueError("steps must be positive")
        if hasattr(args, "samples") and args.samples < 1:
            raise ValueError("samples must be positive")
        if args.command == "freeze" and args.out.exists():
            raise ValueError("freeze destination must not already exist")
        if args.command == "freeze" and args.out.resolve().is_relative_to((ROOT / "engine").resolve()):
            raise ValueError("freeze destination cannot be inside the source engine directory")
        if args.command in {"auto", "sweep"}:
            from .loop import run_loop
            from .sweep import run_sweep
            from .transaction import Transaction, crash
            from .night import exclusive, night_watch
            with exclusive():
                # Recovery precedes guard and imports/remote dispatch.
                Transaction(ROOT, RESULTS).startup(args.resume)
                with night_watch():
                    try:
                        check(ROOT / 'engine')
                        with Remote(Budget(args.max_gpu_minutes)) as remote:
                            return run_loop(args, remote) if args.command == 'auto' else run_sweep(args, remote)
                    except SpendLimit as exc:
                        record_stop(exc)
                        print(str(exc), flush=True)
                        return 0
                    except BaseException:
                        crash(RESULTS)
                        raise
        workloads = select_workloads(getattr(args, "workloads", "public"))
        payload = package(ROOT / "engine")
        with Remote(smoke_check=args.command == 'check' and workloads == [PUBLIC[0]]) as remote:
            if args.command == "download-weights":
                print(remote.api.download_weights.remote())
                return 0
            if args.command in {"profile", "gauge", "race"}:
                chosen = select_workloads(args.workload)
                if len(chosen) != 1:
                    raise ValueError("select exactly one workload")
                if args.command == "race":
                    race(remote, args.a, args.b, chosen[0])
                else:
                    result = remote.profile(payload, chosen[0])
                    show_profile(result) if args.command == "profile" else gauge(result)
                return 0
            if args.command == "freeze":
                checked = remote.bench(payload, PUBLIC, 1, correctness_only=True)
                measured = remote.bench(payload, workloads, 5) if checked["eligible"] else checked
                report(measured)
                if not checked["eligible"] or not measured["eligible"]:
                    return 1
                if package(ROOT / "engine") != payload:
                    raise ValueError("engine changed during freeze validation")
                args.out.mkdir(parents=True)
                shutil.copytree(ROOT / "engine", args.out / "engine", ignore=shutil.ignore_patterns("__pycache__"))
                write_json(args.out / "FREEZE.json", {"sha": git_sha(), "check": checked, "bench": measured})
                print(f"Frozen to {args.out.resolve()}")
                return 0
            result = remote.bench(payload, workloads, getattr(args, "samples", 1), args.command == "check",
                                  getattr(args, "refresh_native", False), getattr(args, "transport", None),
                                  getattr(args, "prompt_seed", None))
            report(result, args.command == "check")
            if args.command == "bench":
                append_log(result, note="manual benchmark; no keep decision")
            return 0 if result["eligible"] else 1
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; experiment preserved. Use auto --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
