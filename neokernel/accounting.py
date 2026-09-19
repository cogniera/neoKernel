"""Conservative per-tier estimates and the attended session's dollar stop limit."""

import json
import time
from pathlib import Path

from .storage import RESULTS, timestamp, write_json

# Published Modal task rates checked at https://modal.com/pricing.
GPU_RATES_USD_S = {"L4": .000222, "H100": .001097}
CPU_RATE_USD_S = .0000131
MEMORY_RATE_USD_GIB_S = .00000222
GPU_TIMEOUT_S = 900
GPU_IDLE_S = 2
CPU_CORES = 8.0
MEMORY_GIB = 32
STOP_USD = 3.0


def estimate_usd(tier: str, seconds: float) -> float:
    return seconds * (GPU_RATES_USD_S[tier] + CPU_CORES * CPU_RATE_USD_S + MEMORY_GIB * MEMORY_RATE_USD_GIB_S)


class SpendLedger:
    def __init__(self, directory: Path = RESULTS):
        self.directory = directory
        self.path = directory / "spend.json"

    def read(self) -> dict:
        result = json.loads(self.path.read_text()) if self.path.exists() else {"calls": []}
        setup = self.directory / "setup.json"
        result["setup_estimated_usd"] = json.loads(setup.read_text()).get("estimated_usd", 0) if setup.exists() else 0
        result["estimated_usd"] = result["setup_estimated_usd"] + sum(c["estimated_usd"] for c in result["calls"])
        return result

    def reserve(self, tier: str):
        # Include a minute of cold startup and the short idle window. The remote
        # function itself has a fixed timeout, independent of this estimate.
        reservation = estimate_usd(tier, GPU_TIMEOUT_S + 60 + GPU_IDLE_S)
        current = self.read()["estimated_usd"]
        if current + reservation > STOP_USD:
            raise ValueError(f"Spend stop: ${current:.4f} estimated so far; ${reservation:.4f} reservation would exceed ${STOP_USD:.2f}")
        return reservation

    def record(self, tier: str, started: float, reported_s: float | None, operation: str, passed: bool) -> dict:
        wall_s = time.perf_counter() - started
        billed_s = max(wall_s, reported_s or 0) + GPU_IDLE_S
        if reported_s is None:
            billed_s = max(billed_s, GPU_TIMEOUT_S + 60 + GPU_IDLE_S)
        result = self.read()
        result["calls"].append({"ts": timestamp(), "operation": operation, "tier": tier,
                                "reported_gpu_s": reported_s, "estimated_allocation_s": billed_s,
                                "estimated_usd": estimate_usd(tier, billed_s), "passed": passed})
        result["estimated_usd"] = result["setup_estimated_usd"] + sum(c["estimated_usd"] for c in result["calls"])
        result["rates_source"] = "https://modal.com/pricing"
        write_json(self.path, result)
        print(f"Spend: {tier} estimated allocation {billed_s:.2f}s; cumulative estimated ${result['estimated_usd']:.4f}")
        if result["estimated_usd"] >= STOP_USD:
            raise ValueError(f"Spend stop: estimated cumulative cost is ${result['estimated_usd']:.4f}")
        return result
