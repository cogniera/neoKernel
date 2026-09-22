"""Parent-clock evaluation and native teacher-forced replay.

PyTorch is imported lazily so guard, CLI help, and arithmetic tests work on CPU
hosts without the GPU runtime. The wire protocol is JSON, never pickle.
"""

import ast
import json
import math
import os
import queue
import secrets
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .guard import physics_floors
from .schema import Correctness, RunResult, Sample, Workload, WorkloadResult

OFFICIAL = {
    "public-0": {"tps": 57.0, "median_total": .56165, "ttft_median": .02201, "tpot_median": .01741, "peak_mem_gib": 9.57},
    "public-1": {"tps": 150.0, "median_total": .85337, "ttft_median": .20279, "tpot_median": .02101, "peak_mem_gib": 11.44},
    "public-2": {"tps": 671.2, "median_total": 3.05141, "ttft_median": .19263, "tpot_median": .02251, "peak_mem_gib": 11.44},
}


# The judge's tie budget, calibrated on native against itself. One name so the
# gate, the native-reference check and their messages cannot drift apart.
MARGIN = 2.0


class CandidateError(RuntimeError):
    pass


def validate_step(step, batch: int, vocab_size: int) -> None:
    if type(step) is not list or len(step) != batch or any(type(t) is not int or not 0 <= t < vocab_size for t in step):
        raise CandidateError("each yield must be a list of batch Python ints inside the vocabulary")


def validate_tokens(steps, batch: int, N: int, vocab_size: int) -> None:
    if len(steps) != N:
        raise CandidateError(f"expected {N} yields, received {len(steps)}")
    for step in steps:
        validate_step(step, batch, vocab_size)


def check_logits(logits, emitted, margin_limit: float = MARGIN) -> Correctness:
    """Compare [B,N,V] logits with [B,N] emitted IDs, on CPU or GPU."""
    import torch
    chosen = torch.as_tensor(emitted, device=logits.device, dtype=torch.long)
    maximum, argmax = logits.float().max(dim=-1)
    margins = maximum - logits.float().gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
    finite = torch.isfinite(logits).all(dim=-1)
    bad = (margins > margin_limit) | ~finite
    near = int(((argmax != chosen) & ~bad).sum().item())
    if bad.any():
        b, n = bad.nonzero()[0].tolist()
        value = float(margins[b, n])
        return Correctness(False, [b, n], value if math.isfinite(value) else None, near)
    return Correctness(near_tie_count=near)


def replay(model, prompt: list[list[int]], steps: list[list[int]]) -> Correctness:
    """Replay each sequence on its emitted prefix after the candidate has exited."""
    import torch
    device = next(model.parameters()).device
    result = Correctness()
    with torch.inference_mode():
        for b, row in enumerate(prompt):
            emitted = [step[b] for step in steps]
            ids = torch.tensor([row + emitted], dtype=torch.long, device=device)
            logits = model(input_ids=ids, use_cache=False).logits[:, len(row) - 1:len(row) + len(emitted) - 1]
            current = check_logits(logits, [emitted])
            result.near_tie_count += current.near_tie_count
            if not current.passed and result.passed:
                result.passed = False
                result.first_bad_position = [b, current.first_bad_position[1]]
                result.margin = current.margin
    return result


def make_prompt(workload: Workload, vocab: int, specials: list[int], seed: int) -> list[list[int]]:
    import torch
    excluded = set(specials)
    allowed = torch.tensor([i for i in range(vocab) if i not in excluded], dtype=torch.long)
    if not len(allowed):
        raise ValueError("vocabulary has no non-special tokens")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return allowed[torch.randint(len(allowed), (workload.batch, workload.S), generator=generator)].tolist()


def summarize(w: Workload, samples: list[Sample], correctness: Correctness, load_s: float,
              warmup_s: float, device_bytes: int, native: dict | None, floors: dict,
              correctness_only: bool = False) -> WorkloadResult:
    if not samples or any(s.total_s <= 0 or not math.isfinite(s.total_s) for s in samples):
        raise ValueError("nonempty, finite positive measurements required")
    total = statistics.median(s.total_s for s in samples)
    ttft = statistics.median(s.ttft_s for s in samples)
    tpot = statistics.median(s.tpot_s for s in samples)
    spread = (max(s.total_s for s in samples) - min(s.total_s for s in samples)) / total
    tr = ttft / native["ttft_median"] if native else None
    pr = tpot / native["tpot_median"] if native and w.N > 1 else (1.0 if native else None)
    memory = max(max(s.peak_mem_bytes, s.lifetime_peak_mem_bytes) for s in samples) / device_bytes
    gates = {"correctness": correctness.passed, "load_budget": load_s + warmup_s <= 300,
             "timeout": all(s.total_s <= 300 for s in samples), "memory_limit": memory <= .9}
    if not correctness_only:
        gates.update({"latency_limit": tr is not None and pr is not None and tr <= 1.05 and pr <= 1.05,
                      "unstable_timing": spread <= .25,
                      "physics_violation": all(s.total_s >= floors["decode_floor_s"] and
                                               s.ttft_s >= floors["prefill_floor_s"] for s in samples)})
    failure = next(("incorrect_output" if k == "correctness" else k for k, ok in gates.items() if not ok), None)
    enriched = dict(floors, measured_to_floor_ratio=(w.batch * w.N / total) / floors["floor_tps"] if floors["floor_tps"] else 0)
    return WorkloadResult(**asdict(w), samples=samples, median_total=total, spread=spread,
                          tps=w.batch*w.N/total, ttft_median=ttft, tpot_median=tpot,
                          ttft_ratio=tr, tpot_ratio=pr, peak_mem_frac=memory,
                          correctness=correctness, load_seconds=load_s, warmup_s=warmup_s,
                          gates=gates, passed=failure is None, failure_code=failure, floors=enriched,
                          pipe_overhead_ms=describe([v for s in samples for v in s.pipe_overhead_ms]),
                          cpu_step_ms=describe([v for s in samples for v in s.child_cpu_ms[1:]]))


def describe(values: list[float]) -> dict:
    return {"mean": statistics.mean(values) if values else None, "max": max(values) if values else None,
            "count": len(values)}


def read_exact(stream, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count-len(chunks))
        if not chunk:
            raise EOFError("candidate pipe closed mid-frame")
        chunks.extend(chunk)
    return bytes(chunks)


def read_binary_message(stream):
    magic = read_exact(stream, 4)
    if magic == b"NKST":
        count, sent_s, cpu_ms = struct.unpack("<Idd", read_exact(stream, 20))
        if not 0 < count <= 65536:
            raise ValueError("invalid binary batch size")
        payload = read_exact(stream, count*8)
        stamp = time.perf_counter()
        value = {"kind": "step", "tokens": list(struct.unpack(f"<{count}q", payload)),
                 "child_sent_s": sent_s, "child_cpu_ms": cpu_ms}
    elif magic == b"NKJS":
        count, = struct.unpack("<I", read_exact(stream, 4))
        if count > 1024*1024:
            raise ValueError("oversized control frame")
        payload = read_exact(stream, count)
        stamp = time.perf_counter()
        value = json.loads(payload)
    else:
        raise ValueError("invalid binary frame marker")
    return stamp, value


def aggregate(results: list[WorkloadResult], **kwargs) -> RunResult:
    passed = bool(results) and all(r.passed for r in results)
    score = math.exp(sum(math.log(r.tps) for r in results) / len(results)) if passed else None
    return RunResult(results, score, passed, **kwargs)


def child_launch_options(platform: str) -> dict:
    """Use a killable session on Linux and a hidden subprocess on Windows."""
    return ({"start_new_session": False, "creationflags": 0x08000000} if platform == "nt"
            else {"start_new_session": True})


class Child:
    """A fresh interpreter with a private JSON pipe and no harness import path."""
    def __init__(self, engine_dir: Path, model_path: str, mode: str = "engine", transport: str = "json"):
        if transport not in {"json", "binary"}:
            raise ValueError("unknown transport")
        self.transport = transport
        self.max_message_bytes = 64 * 1024 * 1024 if mode == 'profile' else 1024 * 1024
        runner = Path(__file__).with_name("worker.py").read_text(encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k.upper() in {
            "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME"}}
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        self.stderr = tempfile.TemporaryFile()
        try:
            pipe_options = {"text": True, "encoding": "utf-8", "bufsize": 1} if transport == "json" else {"bufsize": 0}
            self.process = subprocess.Popen([sys.executable, "-I", "-u", "-c", runner, str(engine_dir.resolve()), model_path, mode, transport],
                                            cwd=engine_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.stderr, env=env, **pipe_options,
                                            **child_launch_options(os.name))
        except BaseException:
            self.stderr.close()
            raise
        self.messages = queue.Queue(maxsize=512)
        self.closed = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while not self.closed.is_set():
                if self.transport == "binary":
                    try:
                        stamp, value = read_binary_message(self.process.stdout)
                        line = True
                    except (EOFError, ValueError) as exc:
                        stamp, value, line = time.perf_counter(), {"kind": "error", "message": str(exc)}, False
                else:
                    line = self.process.stdout.readline(self.max_message_bytes)
                    stamp = time.perf_counter()
                    if not line:
                        value = {"kind": "error", "message": "candidate pipe closed"}
                    elif not line.endswith("\n"):
                        value = {"kind": "error", "message": "oversized wire message"}
                    else:
                        try:
                            value = json.loads(line)
                        except ValueError:
                            value = {"kind": "error", "message": "invalid wire JSON"}
                while not self.closed.is_set():
                    try:
                        self.messages.put((stamp, value), timeout=.1)
                        break
                    except queue.Full:
                        continue
                if not line:
                    break
        except (OSError, ValueError):
            pass

    def receive(self, deadline: float):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("candidate deadline exceeded")
        try:
            stamp, value = self.messages.get(timeout=remaining)
        except queue.Empty as e:
            raise TimeoutError("candidate deadline exceeded") from e
        if not isinstance(value, dict) or not isinstance(value.get("kind"), str) or value["kind"] == "error":
            raise CandidateError(str(value)[:2000])
        return stamp, value

    def send(self, value):
        payload = json.dumps(value) + "\n"
        if self.transport == "binary":
            payload = payload.encode("utf-8")
        offset = 0
        while offset < len(payload):
            offset += self.process.stdin.write(payload[offset:])
        self.process.stdin.flush()

    def sample(self, prompt, w, vocab, deadline=None, on_step=None):
        start = time.perf_counter()
        deadline = min(deadline, start + 300) if deadline else start + 300
        self.send({"kind": "generate", "prompt": prompt, "N": w.N})
        steps, times, pipe_ms, cpu_ms = [], [], [], []
        while True:
            stamp, message = self.receive(deadline)
            if message["kind"] == "done":
                peak = message.get("peak_mem_bytes")
                if type(peak) is not int or peak < 0:
                    raise CandidateError("invalid memory report")
                break
            if message["kind"] != "step" or len(steps) >= w.N:
                raise CandidateError("unexpected or excessive yield")
            validate_step(message.get("tokens"), w.batch, vocab)
            steps.append(message["tokens"])
            times.append(stamp - start)
            sent = message.get("child_sent_s")
            cpu = message.get("child_cpu_ms")
            if isinstance(sent, (float, int)) and math.isfinite(sent):
                pipe_ms.append((stamp-sent)*1000)
            if isinstance(cpu, (float, int)) and math.isfinite(cpu):
                cpu_ms.append(cpu)
            if on_step:
                on_step(message["tokens"], stamp - start)
        validate_tokens(steps, w.batch, w.N, vocab)
        return Sample(prompt, steps, times[0], (times[-1] - times[0]) / (w.N - 1) if w.N > 1 else 0,
                      times[-1], peak, times, pipe_overhead_ms=pipe_ms, child_cpu_ms=cpu_ms)

    def close(self):
        self.closed.set()
        if os.name != "nt":
            import signal
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self.reader.join(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()


def measure(engine_dir: Path, model_path: str, w: Workload, samples: int, vocab: int,
            specials: list[int], prompt_seed: int | None = None, on_step=None, transport="json"):
    start = time.perf_counter()
    child = Child(engine_dir, model_path, transport=transport)
    try:
        _, ready = child.receive(start + 300)
        if ready["kind"] != "ready":
            raise CandidateError("missing ready message")
        loaded = time.perf_counter()
        warm = make_prompt(w, vocab, specials, secrets.randbits(63))
        child.sample(warm, w, vocab, start + 300)
        warmup_s = time.perf_counter() - loaded
        seen = {tuple(map(tuple, warm))}
        measured = []
        for i in range(samples):
            while True:
                prompt = make_prompt(w, vocab, specials, prompt_seed + i if prompt_seed is not None else secrets.randbits(63))
                identity = tuple(map(tuple, prompt))
                if identity not in seen:
                    seen.add(identity)
                    break
                if prompt_seed is not None:
                    raise ValueError("duplicate seeded prompt")
            sample = child.sample(prompt, w, vocab, on_step=on_step)
            sample.child_diagnostics = ready.get("diagnostics", {})
            measured.append(sample)
        # Ask only after all samples are complete. This includes transient load
        # and warmup allocations that per-sample reset_peak_memory_stats loses.
        child.send({"kind": "memory"})
        _, memory = child.receive(time.perf_counter() + 10)
        peak = memory.get("peak_mem_bytes")
        if memory["kind"] != "memory" or type(peak) is not int or peak < 0:
            raise CandidateError("invalid final lifetime memory report")
        measured[-1].lifetime_peak_mem_bytes = peak
        return measured, loaded - start, warmup_s
    except TimeoutError as e:
        if 'warmup_s' not in locals():
            raise CandidateError("load_budget") from e
        raise
    finally:
        child.close()


def declares_speculative(engine_dir: Path) -> bool:
    tree = ast.parse((engine_dir / "engine.py").read_text())
    return any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPECULATIVE" for t in n.targets)
               and isinstance(n.value, ast.Constant) and n.value.value is True for n in tree.body)


def replay_samples(model, records: list[Sample]) -> Correctness:
    result = Correctness()
    for sample_index, sample in enumerate(records):
        tested = replay(model, sample.prompt, sample.tokens)
        result.near_tie_count += tested.near_tie_count
        if not tested.passed and result.passed:
            result.passed = False
            result.first_bad_position = [sample_index] + tested.first_bad_position
            result.margin = tested.margin
    return result


def pair_measurements(index: int, candidate_call, native_call):
    """Alternate sequential processes; no replay or other GPU work between them."""
    if index % 2 == 0:
        native = native_call()
        candidate = candidate_call()
        order = "native,candidate"
    else:
        candidate = candidate_call()
        native = native_call()
        order = "candidate,native"
    return candidate, native, order


class NativeReplayError(RuntimeError):
    """Native's own output missed the tie budget the judge enforces on candidates.

    The contract calibrates the 2.0 margin on native against itself and puts the
    resulting noise at up to 0.75 logits, so native exceeding 2.0 means the local
    oracle, not the candidate, is out of specification: a cached decode path and
    one full forward have diverged further than the rule allows for. Observed at
    margin 2.0625 on H100 across the six-workload set on 2026-09-22. This is an
    infrastructure condition to retry and calibrate, never a candidate verdict,
    so it carries the evidence instead of reaching the caller as a bare error.
    """

    def __init__(self, workload: str, correctness):
        self.workload, self.correctness = workload, correctness
        super().__init__(f"local native replay failed on {workload}: {asdict(correctness)}")


def native_record(w: Workload, measurement, model, transport: str) -> dict:
    records, load_s, warmup_s = measurement
    correctness = replay_samples(model, records)
    if not correctness.passed:
        raise NativeReplayError(w.name, correctness)
    total_s = statistics.median(s.total_s for s in records)
    return {"source": "modal-container-native", "protocol": "paired-local-native/2", "transport": transport,
            "sample_count": len(records), "shape": asdict(w), "correctness": asdict(correctness),
            "tps": w.batch*w.N/total_s, "median_total": total_s,
            "ttft_median": statistics.median(s.ttft_s for s in records),
            "tpot_median": statistics.median(s.tpot_s for s in records),
            "load_s": load_s, "warmup_s": warmup_s,
            "pipe_overhead_ms": describe([v for s in records for v in s.pipe_overhead_ms]),
            "cpu_step_ms": describe([v for s in records for v in s.child_cpu_ms[1:]]),
            "child_diagnostics": records[0].child_diagnostics}


def keep_decision(result: dict, baseline: dict) -> tuple[bool, float | None]:
    """Only the judge grants keep; compare identical measured workload sets."""
    allowed = {'incorrect_output', 'candidate_error', 'timeout', 'load_budget',
               'latency_limit', 'memory_limit', 'unstable_timing', 'physics_violation'}
    for run in (baseline, result):
        for w in run.get('workloads', []):
            if w.get('failure_code') and w['failure_code'] not in allowed:
                raise RuntimeError('Harness failure: ' + str(w['failure_code']))
    score, previous = result.get('geomean_tps'), baseline.get('geomean_tps')
    signature = lambda run: sorted((w['name'], w.get('batch'), w.get('S'), w.get('N')) for w in run.get('workloads', []))
    passed = bool(result.get('eligible') and baseline.get('eligible') and result.get('workloads')
                  and signature(result) == signature(baseline)
                  and all(w.get('passed') and w.get('gates') and all(w['gates'].values()) for w in result['workloads']))
    delta = (score / previous - 1) * 100 if score and previous and math.isfinite(score) and math.isfinite(previous) and previous > 0 else None
    # A geomean win that trades one shape for another is not kept (experiment #37).
    before = {w['name']: w.get('tps') for w in baseline.get('workloads', [])}
    regressed = any(before.get(w['name']) and w.get('tps', 0) < before[w['name']] * 0.97 for w in result.get('workloads', []))
    return bool(passed and not regressed and delta is not None and delta > 1.0 and score > previous * 1.01), delta


def evaluate(engine_dir: Path, model_path: str, workloads: list[Workload], samples: int,
             model, tokenizer, native: dict, correctness_only=False, prompt_seed=None,
             baseline_dir: Path | None = None, transport="json", refresh_native=False) -> RunResult:
    if samples < 1:
        raise ValueError("samples must be positive")
    import torch
    params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Native parameter_count={params} weight_bytes={weight_bytes}", flush=True)
    speculative = declares_speculative(engine_dir)
    results, used_native = [], {}
    started = time.perf_counter()
    for index, w in enumerate(workloads):
        floors = physics_floors(weight_bytes, params, w.batch, w.S, w.N, speculative)
        try:
            def candidate_call():
                return measure(engine_dir, model_path, w, samples, model.config.vocab_size,
                               tokenizer.all_special_ids, prompt_seed, transport=transport)
            key = (w.name, w.batch, w.S, w.N, samples, transport)
            if correctness_only:
                candidate_measurement = candidate_call()
                order = "candidate-only"
            elif key in native and not refresh_native:
                candidate_measurement = candidate_call()
                used_native[w.name] = native[key]
                order = "container-cached-native,candidate"
            else:
                if baseline_dir is None:
                    raise ValueError("benchmark requires the unchanged local starter")
                def native_call():
                    try:
                        return measure(baseline_dir, model_path, w, samples, model.config.vocab_size,
                                       tokenizer.all_special_ids, prompt_seed, transport=transport)
                    except (CandidateError, TimeoutError, BrokenPipeError) as exc:
                        raise RuntimeError(f"native reference failed: {exc}") from exc
                candidate_measurement, native_measurement, order = pair_measurements(index, candidate_call, native_call)
                # Both children have exited before any teacher-forced replay.
                native[key] = native_record(w, native_measurement, model, transport)
                used_native[w.name] = native[key]
            records, load_s, warmup_s = candidate_measurement
            correctness = replay_samples(model, records)
            result = summarize(w, records, correctness, load_s, warmup_s,
                               torch.cuda.get_device_properties(0).total_memory, used_native.get(w.name), floors, correctness_only)
            result.measurement_order = order
        except NativeReplayError as e:
            # Not the candidate's fault, and not a result: the local oracle missed
            # its own gate, so this workload carries no verdict. 'harness_error' is
            # outside keep_decision's allowed set, so the run cannot be kept and is
            # raised to the operator to retry rather than charged to the engine.
            note = (f"native reference missed the {MARGIN} tie budget by "
                    f"{e.correctness.margin} at {e.correctness.first_bad_position}; "
                    "retry, and recalibrate the budget if it recurs")
            print(f"HARNESS: {w.name}: {note}", flush=True)
            result = WorkloadResult(**asdict(w), failure_code="harness_error", note=note,
                                    gates={"harness_error": False}, floors=floors)
        except (CandidateError, TimeoutError, BrokenPipeError) as e:
            code = "timeout" if isinstance(e, TimeoutError) else "load_budget" if str(e) == "load_budget" else "candidate_error"
            result = WorkloadResult(**asdict(w), failure_code=code, note=str(e), gates={code: False}, floors=floors)
        results.append(result)
    return aggregate(results, gpu_seconds=time.perf_counter()-started, weight_bytes=weight_bytes,
                     parameter_count=params, native=used_native)
