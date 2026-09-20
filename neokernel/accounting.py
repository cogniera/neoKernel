"""Conservative per-tier estimates and the attended session's dollar stop limit."""

import json
import time
import uuid
from pathlib import Path

from .storage import RESULTS, timestamp, write_json, read_log

# Published Modal task rates checked at https://modal.com/pricing.
GPU_RATES_USD_S = {"L4": .000222, "H100": .001097}
CPU_RATE_USD_S = .0000131
MEMORY_RATE_USD_GIB_S = .00000222
GPU_TIMEOUT_S = 900
GPU_IDLE_S = 60
CPU_CORES = 8.0
MEMORY_GIB = 32
STOP_USD = 3.0


class SpendLimit(ValueError):
    """A normal, clean search stop; never a candidate or harness failure."""


def start_night(directory=RESULTS):
    path = directory / 'night_budget.json'
    if not path.exists():
        ledger = SpendLedger(directory)
        write_json(path, {'start_estimated_usd': ledger.read()['estimated_usd'], 'limit_usd': 8.0,
                          'start_log_id': max((r['id'] for r in read_log(directory)), default=0),
                          'ts': timestamp()})
    return json.loads(path.read_text())


def estimate_usd(tier: str, seconds: float) -> float:
    return seconds * (GPU_RATES_USD_S[tier] + CPU_CORES * CPU_RATE_USD_S + MEMORY_GIB * MEMORY_RATE_USD_GIB_S)


class SpendLedger:
    def __init__(self, directory: Path = RESULTS):
        self.directory = directory
        self.path = directory / "spend.json"

    def ceiling(self):
        night = self.directory / 'night_budget.json'
        if night.exists():
            task = json.loads(night.read_text())
            return float(task['start_estimated_usd']) + min(8.0, float(task['limit_usd']))
        task_path = self.directory / "hand_rolled_budget.json"
        if task_path.exists():
            task = json.loads(task_path.read_text())
            # This attended task has an explicit incremental budget. Include
            # historical spending without charging it against that allowance.
            return float(task["start_estimated_usd"]) + float(task["limit_usd"])
        # A baseline marked kept is not an optimization experiment.
        return 6.0 if any(r.get('proposer') in {'agent', 'codex'} and r.get('kept') and r.get('files_changed')
                          and r.get('guard') == 'pass' for r in read_log(self.directory)) else STOP_USD

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
        self.reserve_amount(tier, reservation, 'pending GPU call')
        return reservation

    def reserve_amount(self, tier, reservation, operation):
        result = self.read()
        current = result['estimated_usd']
        limit = self.ceiling()
        if current + reservation > limit:
            raise SpendLimit(f"Spend stop: ${current:.4f} charged/reserved; ${reservation:.4f} reservation would exceed ${limit:.2f}")
        self.reservation_id = uuid.uuid4().hex
        result['calls'].append(dict(id=self.reservation_id, ts=timestamp(), tier=tier, operation=operation,
                                    estimated_usd=reservation, status='reserved', estimated_allocation_s=0))
        result['estimated_usd'] = current + reservation
        write_json(self.path, result)
        return self.reservation_id

    def settle(self, amount, **details):
        result = self.read()
        reservation_id = getattr(self, 'reservation_id', None)
        row = next((r for r in result['calls'] if r.get('id') == reservation_id), None)
        if row is None:
            raise RuntimeError('Missing spend reservation')
        row.update(estimated_usd=amount, status='settled', **details)
        result['estimated_usd'] = result['setup_estimated_usd'] + sum(c['estimated_usd'] for c in result['calls'])
        write_json(self.path, result)
        self.reservation_id = None
        return result

    def record(self, tier: str, started: float, reported_s: float | None, operation: str, passed: bool) -> dict:
        wall_s = time.perf_counter() - started
        billed_s = max(wall_s, reported_s or 0) + GPU_IDLE_S
        if reported_s is None:
            billed_s = max(billed_s, GPU_TIMEOUT_S + 60 + GPU_IDLE_S)
        if not getattr(self, 'reservation_id', None):
            self.reserve(tier)
        result = self.settle(estimate_usd(tier, billed_s), operation=operation, tier=tier,
                             reported_gpu_s=reported_s, estimated_allocation_s=billed_s, passed=passed)
        result["estimated_usd"] = result["setup_estimated_usd"] + sum(c["estimated_usd"] for c in result["calls"])
        result["rates_source"] = "https://modal.com/pricing"
        write_json(self.path, result)
        print(f"Spend: {tier} estimated allocation {billed_s:.2f}s; cumulative estimated ${result['estimated_usd']:.4f}")
        if result["estimated_usd"] >= self.ceiling():
            raise SpendLimit(f"Spend stop: estimated cumulative cost is ${result['estimated_usd']:.4f}")
        return result
