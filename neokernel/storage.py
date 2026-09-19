"""Local run records, baseline provenance, and non-Git experiment snapshots."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict

from .schema import LogLine

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "neokernel" / "results"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unversioned"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_native(directory: Path = RESULTS) -> dict:
    """Read local reports for display only; remote containers measure their own native."""
    native = {}
    for path in directory.glob("native_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("source") == "modal-container-native":
            native[path.stem.removeprefix("native_")] = record
    return native


def save_run(result: dict, payload: bytes, directory: Path = RESULTS) -> Path:
    result["ts"] = timestamp()
    result["sha"] = git_sha()
    result["engine_sha256"] = hashlib.sha256(payload).hexdigest()
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    path = directory / "runs" / f"{name}_{result['sha'][:12]}.json"
    write_json(path, result)
    for workload, record in result.get("native", {}).items():
        write_json(directory / f"native_{workload}.json", record)
    return path


def read_log(directory: Path = RESULTS) -> list[dict]:
    path = directory / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_log(result: dict | None, *, proposer="human", item="manual", hypothesis="",
               kept=False, note="", files_changed=None, guard="pass", patch_sha256=None, proposal_sha256=None, diff='', implemented_items=None,
               directory: Path = RESULTS) -> dict:
    records = read_log(directory)
    previous = next((r for r in reversed(records) if r.get("kept") and r.get("geomean_tps")), None)
    score = result.get("geomean_tps") if result else None
    row = {"id": max((r["id"] for r in records), default=0) + 1, "ts": timestamp(),
           "sha": git_sha(), "parent_sha": previous["sha"] if previous else None,
           "proposer": proposer, "item": item, "hypothesis": hypothesis,
           "files_changed": files_changed or [], "guard": guard,
           "workloads": [{k: v for k, v in w.items() if k != "samples"}
                         for w in result.get("workloads", [])] if result else [], "geomean_tps": score,
           "delta_pct": (score / previous["geomean_tps"] - 1) * 100 if score and previous else None,
           "kept": kept, "gpu_seconds": result.get("gpu_seconds", 0) if result else 0, "note": note,
           "engine_sha256": result.get("engine_sha256") if result else None,
           "patch_sha256": patch_sha256, "proposal_sha256": proposal_sha256, "diff": diff,
           "implemented_items": implemented_items or []}
    row = asdict(LogLine(**row))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    return row


def snapshot(engine_dir: Path) -> dict[str, bytes]:
    return {p.relative_to(engine_dir).as_posix(): p.read_bytes() for p in engine_dir.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def restore(engine_dir: Path, state: dict[str, bytes]) -> None:
    """Restore only files in the explicitly scoped engine directory."""
    root = engine_dir.resolve()
    for p in engine_dir.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            if not p.resolve().is_relative_to(root):
                raise ValueError("snapshot target escapes engine directory")
            if p.relative_to(engine_dir).as_posix() not in state:
                p.unlink()
    for name, content in state.items():
        path = engine_dir / name
        if not path.resolve().is_relative_to(root):
            raise ValueError("snapshot path escapes engine directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


class Budget:
    """Reserve worst-case GPU seconds before dispatch; charge returned usage."""
    def __init__(self, minutes: float):
        if minutes <= 0:
            raise ValueError("GPU budget must be positive")
        self.limit_s = minutes * 60
        self.used_s = 0.0

    def reserve(self, seconds: float) -> None:
        if self.used_s + seconds > self.limit_s:
            raise ValueError(f"GPU budget exceeded: {self.used_s:.1f}s used, {seconds:.1f}s reserved, {self.limit_s:.1f}s limit")

    def charge(self, seconds: float) -> None:
        self.used_s += seconds
