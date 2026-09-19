"""Record the native H100 baseline and label ratio-adjusted estimates explicitly."""

import json
from pathlib import Path

from .judge import OFFICIAL
from .storage import RESULTS, write_json


def build_calibration(run: dict, min_samples=3) -> dict:
    if run.get("gpu_tier") != "H100":
        raise ValueError("calibration requires an H100 benchmark")
    by_name = {w["name"]: w for w in run["workloads"]}
    rows = {}
    for name, official in OFFICIAL.items():
        if min_samples < 3 and name not in run.get('native', {}):
            continue
        candidate = by_name[name]
        measured = run.get("native", {}).get(name)
        if not measured or measured.get("source") != "modal-container-native":
            raise ValueError("calibration requires native measured in the same container")
        if measured.get("sample_count", 0) < min_samples or not measured["correctness"]["passed"]:
            raise ValueError("calibration requires three correct local-native samples on every public workload")
        ratios = {"tps": measured["tps"] / official["tps"],
                  "ttft": measured["ttft_median"] / official["ttft_median"],
                  "tpot": measured["tpot_median"] / official["tpot_median"]}
        rows[name] = {"local_tps": measured["tps"], "official_tps": official["tps"],
                      "host_factor": round(ratios['tps'], 4), "sample_count": measured['sample_count'],
                      "local_ttft_ms": measured["ttft_median"] * 1000,
                      "official_ttft_ms": official["ttft_median"] * 1000,
                      "local_tpot_ms": measured["tpot_median"] * 1000,
                      "official_tpot_ms": official["tpot_median"] * 1000,
                      "candidate_tps": candidate["tps"],
                      "pipe_overhead_ms": measured.get("pipe_overhead_ms", {}),
                      "cpu_step_ms": measured.get("cpu_step_ms", {}),
                      "child_diagnostics": measured.get("child_diagnostics", {}),
                      "local_to_official_ratio": ratios, "within_15_percent": abs(ratios["tps"] - 1) <= .15}
    return {"source_run_ts": run.get("ts"), "source_sha": run.get("sha"),
            "engine_sha256": run.get("engine_sha256"), "official_run": "4877cddd",
            "gpu_tier": "H100", "workloads": rows,
            "transport": run.get("transport", "json"), "cpu_diagnostics": run.get("cpu_diagnostics", {}),
            "within_15_percent": all(r["within_15_percent"] for r in rows.values()),
            "benchmark_gates_passed": run["eligible"],
            "note": "Dryft-equivalent values are baseline-ratio estimates, not official measurements. Never used for gates."}


def equivalent_estimates(workload: dict, calibration: dict | None) -> dict | None:
    if not calibration or not (calibration.get("within_15_percent") or calibration.get("accepted")):
        return None
    entry = calibration["workloads"].get(workload["name"])
    if not entry:
        return None
    ratios = entry["local_to_official_ratio"]
    return {"tps": workload["tps"] / entry.get('host_factor', ratios["tps"]),
            "ttft_ms": workload["ttft_median"] * 1000 / ratios["ttft"],
            "tpot_ms": workload["tpot_median"] * 1000 / ratios["tpot"]}


def load_calibration(directory: Path = RESULTS) -> dict | None:
    path = directory / "calibration.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def refresh_host_factors(run: dict, directory: Path = RESULTS) -> dict | None:
    """Refresh measured public factors after every native run, including fast samples."""
    if run.get('gpu_tier') != 'H100' or not any(name in OFFICIAL for name in run.get('native', {})):
        return load_calibration(directory)
    previous = load_calibration(directory) or {}
    updated = build_calibration(run, min_samples=1)
    updated['workloads'] = {**previous.get('workloads', {}), **updated['workloads']}
    updated['accepted'] = previous.get('accepted', False)
    updated['acceptance_reason'] = previous.get('acceptance_reason', '')
    updated['within_15_percent'] = all(row['within_15_percent'] for row in updated['workloads'].values())
    write_json(directory/'calibration.json', updated)
    return updated


def write_calibration(run: dict, document: Path, directory: Path = RESULTS) -> dict:
    record = build_calibration(run)
    write_json(directory / "calibration.json", record)
    lines = ["# Calibration", "", "Official reference: Dryft run 4877cddd. Local-native measurement: unchanged starter, H100, three samples per public workload, paired with the candidate in alternating order.",
             "", f"Source run: {run.get('ts')}. Existing Git SHA: {run.get('sha')}. Engine archive SHA256: {run.get('engine_sha256')}.", "",
             "| Workload | Native tok/s | Official tok/s | Host factor | Native TTFT ms | Official TTFT ms | TTFT factor | Native TPOT ms | Official TPOT ms | TPOT factor |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in record["workloads"].items():
        ratios = row["local_to_official_ratio"]
        lines.append(f"| {name} | {row['local_tps']:.2f} | {row['official_tps']:.2f} | {ratios['tps']:.4f} | {row['local_ttft_ms']:.2f} | {row['official_ttft_ms']:.2f} | {ratios['ttft']:.4f} | {row['local_tpot_ms']:.2f} | {row['official_tpot_ms']:.2f} | {ratios['tpot']:.4f} |")
    lines.extend(["", f"Every throughput ratio within 15 percent: {record['within_15_percent']}. All benchmark gates passed: {record['benchmark_gates_passed']}.",
                  "", "Ratios are local divided by official. The table reports residual differences without normalizing the measurements. "
                  "The CLI divides later H100 public measurements by these baseline ratios to display Dryft-equivalent estimates only when the 15 percent criterion passes. "
                  "Those estimates do not affect correctness, gates, or keep decisions and do not predict hidden-workload leaderboard scores. Latency gates use the paired local native medians, never the official values.", "",
                  f"Parent CPU diagnostics: `{json.dumps(record['cpu_diagnostics'], sort_keys=True)}`. Transport: {record['transport']}.", "",
                  "| Workload | Native pipe mean ms | Native pipe max ms | Native CPU/decode mean ms | Native CPU/decode max ms | Child cgroup cpu.max |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"])
    for name, row in record["workloads"].items():
        pipe, cpu = row["pipe_overhead_ms"], row["cpu_step_ms"]
        lines.append(f"| {name} | {pipe.get('mean')} | {pipe.get('max')} | {cpu.get('mean')} | {cpu.get('max')} | {row['child_diagnostics'].get('cpu_max')} |")
    lines.extend(["",
                  "The judge uses random non-special token IDs rather than Dryft's private corpus. Its parent owns arrival timestamps and replay; "
                  "the candidate loads and warms up once per workload, sees distinct prompts, and reports lifetime GPU peak memory after samples. "
                  "The independent native model stays resident in the parent. These differences and uncontrolled GPU timing mean calibration is approximate.", ""])
    document.write_text("\n".join(lines), encoding="utf-8")
    return record
