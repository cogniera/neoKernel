"""Warm decode profiling and transparent gap calculations."""

import secrets
import time
from collections import defaultdict
from pathlib import Path

from .guard import physics_floors
from .judge import Child, CandidateError, declares_speculative, make_prompt
from .schema import Workload


def aggregate_events(events: list[dict], step_wall_ms: float, byte_estimates: dict | None = None) -> dict:
    """Sum kernel durations, and also expose interval union when kernels overlap."""
    grouped = defaultdict(lambda: {"duration_ms": 0.0, "calls": 0})
    intervals = []
    for event in events:
        start, end = event["start_us"], event["end_us"]
        if end < start:
            raise ValueError("negative kernel duration")
        grouped[event["name"]]["duration_ms"] += (end-start)/1000
        grouped[event["name"]]["calls"] += 1
        intervals.append((start, end))
    covered_us = 0.0
    right = float("-inf")
    for start, end in sorted(intervals):
        covered_us += max(0, end - max(start, right))
        right = max(right, end)
    rows = []
    for name, stats in grouped.items():
        moved = (byte_estimates or {}).get(name)
        rows.append(dict(name=name, **stats, bytes_moved=moved,
                         hbm_peak_fraction=moved/(stats["duration_ms"]*.001*3.35e12) if moved and stats["duration_ms"] else None))
    rows.sort(key=lambda row: row["duration_ms"], reverse=True)
    summed = sum(row["duration_ms"] for row in rows)
    return {"kernels": rows, "step_wall_ms": step_wall_ms, "sum_kernel_ms": summed,
            "gap_ms": step_wall_ms-summed, "kernel_union_ms": covered_us/1000,
            "uncovered_ms": max(0, step_wall_ms-covered_us/1000),
            "note": "Profiler overhead is included. Overlapping kernels can make gap_ms negative; uncovered_ms uses the interval union. Unknown byte counts are null."}


def profile_engine(engine_dir: Path, model_path: str, w: Workload, model, tokenizer) -> dict:
    if w.N < 2:
        raise ValueError("profiling needs at least two output tokens")
    start = time.perf_counter()
    child = Child(engine_dir, model_path, mode='profile')
    try:
        _, ready = child.receive(start+300)
        if ready["kind"] != "ready":
            raise CandidateError("missing ready message")
        warm = make_prompt(w, model.config.vocab_size, tokenizer.all_special_ids, secrets.randbits(63))
        child.sample(warm, w, model.config.vocab_size, start+300)
        prompt = make_prompt(w, model.config.vocab_size, tokenizer.all_special_ids, secrets.randbits(63))
        child.send({"kind": "profile", "prompt": prompt, "N": w.N})
        _, raw = child.receive(time.perf_counter()+300)
        if raw["kind"] != "profile":
            raise CandidateError("unexpected profile response")
    finally:
        child.close()
    result = aggregate_events(raw["events"], raw["step_wall_ms"], raw.get("byte_estimates"))
    for key in ['kernel_count', 'graph_launch_count', 'sync_calls', 'runtime_calls', 'trace']:
        result[key] = raw.get(key)
    params = sum(p.numel() for p in model.parameters())
    weights = sum(p.numel()*p.element_size() for p in model.parameters())
    result["floors"] = physics_floors(weights, params, w.batch, w.S, w.N, declares_speculative(engine_dir))
    result["weight_bytes"] = weights
    result["workload"] = w.name
    return result
