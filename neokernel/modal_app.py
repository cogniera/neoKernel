"""Opt-in Modal entry points. Importing the local CLI never contacts Modal."""

import importlib.metadata
import json
import os
import tempfile
import time
from pathlib import Path

import modal

from .accounting import CPU_CORES, GPU_IDLE_S, GPU_TIMEOUT_S, MEMORY_GIB
from .guard import extract
from .judge import Child, evaluate, measure
from .schema import Correctness, Workload, WorkloadResult

PINS = {"torch": "2.5.1", "triton": "3.1.0", "transformers": "4.51.3",
        "safetensors": "0.5.3", "tokenizers": "0.21.1"}
MODEL_REPO = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_PATH = "/weights/qwen3-4b"
app = modal.App("neokernel-v1")
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install(*(f"{name}=={version}" for name, version in PINS.items()),
                      "huggingface_hub", "rich", extra_index_url="https://download.pytorch.org/whl/cu124")
         .add_local_python_source("neokernel", ignore=["**/results/**", "**/results_backup/**", "**/__pycache__/**"]))
volume = modal.Volume.from_name("neokernel-weights", create_if_missing=True)
_reference = None
_native_cache = {}
_started = time.perf_counter()


def verify_runtime():
    import sys
    import torch
    if sys.version_info[:2] != (3, 11) or torch.version.cuda != "12.4":
        raise RuntimeError("requires Python 3.11 and CUDA 12.4")
    for name, expected in PINS.items():
        actual = importlib.metadata.version(name).split("+")[0]
        if actual != expected:
            raise RuntimeError(f"runtime mismatch: {name} {actual}, expected {expected}")


if not modal.is_local():
    verify_runtime()


@app.function(image=image, volumes={"/weights": volume}, cpu=1, memory=8192, timeout=900, scaledown_window=2)
def download_weights():
    """Explicit setup operation; generation functions never download checkpoints."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_REPO, revision=REVISION, local_dir=MODEL_PATH)
    volume.commit()
    return {"model_path": MODEL_PATH, "revision": REVISION}


@app.function(image=image, cpu=1, memory=2048, timeout=120, scaledown_window=2)
def runtime_probe() -> dict:
    """Verify pinned packages and Linux subprocess startup without allocating a GPU."""
    import json
    import subprocess
    import sys
    import torch
    from .judge import child_launch_options
    verify_runtime()
    versions = {name: importlib.metadata.version(name) for name in PINS}
    child = subprocess.run([sys.executable, "-I", "-u", "-c", "import os; print(os.name)"],
                           capture_output=True, text=True, encoding="utf-8", timeout=30,
                           check=True, **child_launch_options("posix"))
    result = {"versions": versions, "python": sys.version.split()[0], "cuda": torch.version.cuda,
              "child_platform": child.stdout.strip(), "gpu_allocated": False}
    print(json.dumps(result, indent=2), flush=True)
    return result


def reference():
    global _reference
    if _reference is None:
        verify_runtime()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        started = time.perf_counter()
        print(f"Container startup elapsed_s={started - _started:.3f}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                    attn_implementation="sdpa", local_files_only=True).eval().to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        _reference = model, tokenizer
        print(f"Native load_s={time.perf_counter()-started:.3f}; cold=True", flush=True)
    else:
        print("Native cached; cold=False", flush=True)
    return _reference


def cpu_diagnostics():
    import torch
    before = torch.get_num_threads()
    torch.set_num_threads(8)
    quota = Path("/sys/fs/cgroup/cpu.max")
    diagnostics = {"os_cpu_count": os.cpu_count(), "cpu_max": quota.read_text().strip() if quota.exists() else None,
                   "torch_threads_before": before, "torch_num_threads": torch.get_num_threads()}
    diagnostics["cgroup_v1"] = {str(path): path.read_text().strip() for path in [
        Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'), Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us'),
        Path('/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us'), Path('/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us')]
        if path.exists()}
    print(json.dumps({"parent_cpu": diagnostics}), flush=True)
    return diagnostics


def run_one(engine_tar, workloads, samples, refresh_native=False, native=None, correctness_only=False, transport="json"):
    started = time.perf_counter()
    diagnostics = cpu_diagnostics()
    model, tokenizer = reference()
    selected = [Workload(**w) for w in workloads]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        candidate = extract(engine_tar, root / "candidate")
        baseline = root / "native"
        baseline.mkdir()
        baseline.joinpath("engine.py").write_bytes(Path(__file__).with_name("native_engine.py").read_bytes())
        # Client-side files and official numbers never supply latency gates.
        result = evaluate(candidate, MODEL_PATH, selected, samples, model, tokenizer, _native_cache,
                          correctness_only, baseline_dir=baseline, transport=transport,
                          refresh_native=refresh_native).to_dict()
    result["gpu_seconds"] = time.perf_counter() - started
    result["cpu_diagnostics"] = diagnostics
    result["transport"] = transport
    print(f"Estimated GPU seconds consumed: {result['gpu_seconds']:.2f}", flush=True)
    return result


REMOTE = dict(image=image, gpu="H100", volumes={"/weights": volume}, cpu=CPU_CORES,
              memory=MEMORY_GIB * 1024, timeout=GPU_TIMEOUT_S, scaledown_window=GPU_IDLE_S)
L4_REMOTE = dict(REMOTE, gpu="L4")


@app.function(**L4_REMOTE)
def check_remote(engine_tar: bytes, workloads: list, native: dict | None = None) -> dict:
    result = run_one(engine_tar, workloads, 1, False, native, True)
    result["gpu_tier"] = "L4"
    return result


@app.function(**REMOTE)
def judge_remote(engine_tar: bytes, workloads: list, samples: int = 3, refresh_native: bool = False,
                 native: dict | None = None, correctness_only: bool = False, transport: str = "json") -> dict:
    result = run_one(engine_tar, workloads, samples, refresh_native, native, correctness_only, transport)
    result["gpu_tier"] = "H100"
    return result


@app.function(**REMOTE)
def judge_many(engine_tars: list[bytes], workloads: list, samples: int = 2,
               refresh_native: bool = False, native: dict | None = None,
               check_workloads: list | None = None, validate_rmsnorm: bool = False) -> list[dict]:
    results = []
    for payload in engine_tars:
        validation_s = 0.0
        if validate_rmsnorm:
            started = time.perf_counter()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    root = extract(payload, Path(tmp))
                    child = Child(root, MODEL_PATH, mode="validate_rmsnorm")
                    try:
                        _, message = child.receive(time.perf_counter()+300)
                        if message["kind"] != "validated":
                            raise RuntimeError("missing RMSNorm validation result")
                    finally:
                        child.close()
            except (RuntimeError, TimeoutError) as exc:
                from dataclasses import asdict
                validation_s = time.perf_counter() - started
                results.append({"eligible": False, "geomean_tps": None, "gpu_seconds": validation_s,
                                "native": native or {}, "workloads": [asdict(WorkloadResult(
                                    **w, correctness=Correctness(False), gates={"kernel_comparison": False},
                                    failure_code="incorrect_output", note=f"RMSNorm unit comparison: {exc}")) for w in workloads]})
                print(f"RMSNorm validation failed; GPU seconds={validation_s:.2f}", flush=True)
                continue
            validation_s = time.perf_counter() - started
        check_result = run_one(payload, check_workloads or workloads[:1], 1, False, native, True)
        native = check_result["native"]
        if check_result["eligible"]:
            result = run_one(payload, workloads, samples, refresh_native, native)
            result["gpu_seconds"] += check_result["gpu_seconds"]
        else:
            result = check_result
        result["gpu_seconds"] += validation_s
        results.append(result)
        native = result["native"]
        refresh_native = False
    return results


@app.function(**L4_REMOTE)
def profile_remote(engine_tar: bytes, workload: dict) -> dict:
    from .profile import profile_engine
    started = time.perf_counter()
    model, tokenizer = reference()
    with tempfile.TemporaryDirectory() as tmp:
        engine = extract(engine_tar, Path(tmp))
        result = profile_engine(engine, MODEL_PATH, Workload(**workload), model, tokenizer)
    result["gpu_seconds"] = time.perf_counter() - started
    result["gpu_tier"] = "L4"
    print(f"Estimated GPU seconds consumed: {result['gpu_seconds']:.2f}", flush=True)
    return result


@app.function(**REMOTE)
def profile_h100_remote(engine_tar: bytes, workload: dict) -> dict:
    from .profile import profile_engine
    started = time.perf_counter()
    cpu_diagnostics()
    model, tokenizer = reference()
    with tempfile.TemporaryDirectory() as tmp:
        engine = extract(engine_tar, Path(tmp))
        result = profile_engine(engine, MODEL_PATH, Workload(**workload), model, tokenizer)
    result['gpu_seconds'] = time.perf_counter()-started
    result['gpu_tier'] = 'H100'
    return result


@app.function(**REMOTE)
def race_remote(engine_tars: list[bytes], workload: dict):
    """Stream sequential, same-prompt runs; avoid two candidates contending on H100."""
    import queue
    import secrets
    import threading
    from .profile import profile_engine
    model, tokenizer = reference()
    w = Workload(**workload)
    seed = secrets.randbits(62)
    started = time.perf_counter()
    for index, payload in enumerate(engine_tars):
        with tempfile.TemporaryDirectory() as tmp:
            engine = extract(payload, Path(tmp))
            events = queue.Queue()

            def run():
                try:
                    measure(engine, MODEL_PATH, w, 1, model.config.vocab_size, tokenizer.all_special_ids, seed,
                            lambda tokens, elapsed: events.put({"kind": "step", "side": index,
                                                               "tokens": tokens, "elapsed_ms": elapsed*1000,
                                                               "text": [tokenizer.decode([t]) for t in tokens]}))
                except Exception as e:
                    events.put({"kind": "error", "message": str(e)})
                finally:
                    events.put(None)

            thread = threading.Thread(target=run)
            thread.start()
            while True:
                event = events.get()
                if event is None:
                    break
                yield event
            thread.join()
            yield {"kind": "profile", "side": index,
                   "profile": profile_engine(engine, MODEL_PATH, w, model, tokenizer)}
    seconds = time.perf_counter() - started
    print(f"Estimated GPU seconds consumed: {seconds:.2f}", flush=True)
    yield {"kind": "done", "gpu_seconds": seconds}
