import argparse
import hashlib
import json
import tempfile
import unittest
import time
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
from unittest.mock import patch

from neokernel.guard import GuardError, check
from neokernel.loop import apply_proposal, validate_files, generated_diff, history_summary, validate_proposal, retry_call
from neokernel.profile import aggregate_events
from neokernel.schema import Proposal, select_workloads
from neokernel.storage import Budget, append_log, read_log, restore, seed_native, snapshot
from neokernel.sweep import points, set_tunables, tunables, wire_rmsnorm
from neokernel.accounting import SpendLedger, estimate_usd


class WorkflowTests(unittest.TestCase):
    def test_incremental_task_budget_preserves_reservation_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'hand_rolled_budget.json').write_text(json.dumps(
                {'start_estimated_usd': 3.7, 'limit_usd': 4.0}))
            (directory / 'setup.json').write_text(json.dumps({'estimated_usd': 4.9}))
            ledger = SpendLedger(directory)
            self.assertAlmostEqual(ledger.ceiling(), 7.7)
            ledger.reserve('H100')
            (directory / 'setup.json').write_text(json.dumps({'estimated_usd': 7.0}))
            with self.assertRaises(ValueError):
                ledger.reserve('H100')

    def test_ceiling_increases_only_after_kept_optimization(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ledger = SpendLedger(directory)
            baseline = {'proposer': 'agent', 'kept': True, 'files_changed': [], 'guard': 'pass'}
            (directory/'log.jsonl').write_text(json.dumps(baseline)+'\n')
            self.assertEqual(ledger.ceiling(), 3)
            baseline['files_changed'] = ['engine/engine.py']
            baseline['kept'] = False
            (directory/'log.jsonl').write_text(json.dumps(baseline)+'\n')
            self.assertEqual(ledger.ceiling(), 3)
            baseline['kept'] = True
            (directory/'log.jsonl').write_text(json.dumps(baseline)+'\n')
            self.assertEqual(ledger.ceiling(), 6)

    def test_whole_file_schema_and_full_context(self):
        from neokernel.loop import context
        data = dict(item='static_kv_cache', hypothesis='faster', expected_effect='speed', risk='latency',
                    reasoning='one sentence', files={'engine/kernels/x.py': ''})
        self.assertEqual(Proposal.parse(data).files, data['files'])
        for invalid in ['diff', {}, {'engine/engine.py': None}]:
            with self.assertRaises(ValueError):
                Proposal.parse(dict(data, files=invalid))
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp)/'engine'
            restore(engine, {'engine.py': b'complete main', 'kernels/x.py': b'complete kernel', 'notes.txt': b'notes'})
            messages = context(engine, {'trace': {'traceEvents': []}, 'graph_launch_count': 1}, [])
            self.assertEqual(json.loads(messages[1]['content'])['profile'], {'graph_launch_count': 1})
            self.assertEqual(json.loads(messages[1]['content'])['current_files'],
                             {'engine/engine.py': 'complete main', 'engine/kernels/x.py': 'complete kernel'})

    def test_spend_reservation_and_tier_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = SpendLedger(Path(tmp))
            self.assertLess(ledger.reserve("L4"), ledger.reserve("H100"))
            record = ledger.record("L4", time.perf_counter(), 10, "test", True)
            self.assertAlmostEqual(record["calls"][0]["estimated_usd"], estimate_usd("L4", 12))
            (Path(tmp)/"setup.json").write_text(json.dumps({"estimated_usd": 2.9}))
            with self.assertRaises(ValueError):
                ledger.reserve("L4")

    def test_correctness_routes_only_to_l4(self):
        from neokernel.cli import Remote
        remote = Remote()
        result = {"native": {}, "gpu_seconds": 1, "eligible": True, "workloads": [], "geomean_tps": 1}
        remote.api = SimpleNamespace(check_remote=Mock(), judge_remote=Mock())
        remote.api.check_remote.remote.return_value = result
        remote.spend = Mock()
        with patch("neokernel.cli.save_run"):
            remote.bench(b"source", select_workloads("public"), 1, correctness_only=True)
        remote.api.check_remote.remote.assert_called_once()
        remote.api.judge_remote.remote.assert_not_called()
        remote.spend.reserve.assert_called_once_with("L4")

    def test_patch_scope(self):
        for target in ["neokernel/judge.py", "engine/../secret.py", "engine/kernels/../../x.py", "engine/x.py"]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_files({target: 'value = 2\n'})

    def test_patch_apply_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp)/"engine"
            restore(engine, {"engine.py": b"value = 1\n"})
            before = snapshot(engine)
            files = {'engine/engine.py': 'value = 2\n', 'engine/kernels/new.py': 'x = 1\n'}
            self.assertEqual(apply_proposal(engine, files), ['engine/engine.py', 'engine/kernels/new.py'])
            self.assertEqual((engine/'engine.py').read_text(), 'value = 2\n')
            self.assertIn('new file mode', generated_diff(before, snapshot(engine)))
            apply_proposal(engine, {'engine/kernels/new.py': ''})
            self.assertFalse((engine/'kernels/new.py').exists())
            self.assertEqual((engine/'engine.py').read_text(), 'value = 2\n')
            for files in [{'engine/engine.py': ''}, {'engine/engine.py': 'changed', 'engine/../x.py': 'bad'}]:
                saved = snapshot(engine)
                with self.assertRaises(ValueError):
                    apply_proposal(engine, files)
                self.assertEqual(snapshot(engine), saved)
            restore(engine, before)
            self.assertEqual(snapshot(engine), before)

    def test_guard_staged_rmsnorm(self):
        root = Path(__file__).resolve().parents[1]
        live_original = snapshot(root/"engine")
        # This exercises the explicit starter-only adapter, not the current
        # candidate's independently evolving kernel package and tunables.
        original = {
            'engine.py': (root/'neokernel/native_engine.py').read_bytes(),
            'kernels/__init__.py': b'"""Starter kernels."""\n',
            'kernels/rmsnorm.py': (root/'tests/fixtures/starter_rmsnorm.py').read_bytes(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp)/"engine"
            restore(staged, original)
            wire_rmsnorm(staged)
            check(staged)
            init = staged/"kernels"/"__init__.py"
            self.assertEqual(tunables(init.read_text())["rmsnorm.num_warps"], [4, 8, 16])
            init.write_text(set_tunables(init.read_text(), {"rmsnorm.num_warps": 4}))
            self.assertIn("TUNABLES", (staged/"kernels"/"rmsnorm.py").read_text())
        self.assertEqual(snapshot(root/"engine"), live_original)

    def test_tunable_preservation_and_randomness(self):
        source = '"""helpers"""\nTUNABLES = {"a": [1, 2]}\ndef helper(): return 4\n'
        updated = set_tunables(source, {"a": 2})
        self.assertIn("def helper", updated)
        self.assertEqual(tunables(updated), {"a": [2]})
        space = {"a": [1, 2], "b": [3, 4, 5]}
        self.assertEqual(len(list(points(space, 20, False, 0))), 6)
        chosen = list(points(space, 4, True, 17))
        self.assertEqual(chosen, list(points(space, 4, True, 17)))
        self.assertEqual(len({tuple(sorted(p.items())) for p in chosen}), 4)

    def test_profile_overlap(self):
        result = aggregate_events([{"name": "x", "start_us": 0, "end_us": 2000},
                                   {"name": "x", "start_us": 1000, "end_us": 3000}], 5, {"x": 1000})
        self.assertEqual(result["sum_kernel_ms"], 4)
        self.assertEqual(result["gap_ms"], 1)
        self.assertEqual(result["kernel_union_ms"], 3)
        self.assertEqual(result["uncovered_ms"], 2)
        self.assertIsNotNone(result["kernels"][0]["hbm_peak_fraction"])

    def test_native_provenance_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = seed_native(root)
            self.assertEqual(first, {})
            path = root/"native_public-0.json"
            path.write_text(json.dumps({"source": "dryft-official", "tps": 57}))
            self.assertEqual(seed_native(root), {})
            changed = dict(source="modal-container-native", tps=60)
            path.write_text(json.dumps(changed))
            self.assertEqual(seed_native(root)["public-0"]["tps"], 60)
            with patch("neokernel.storage.git_sha", return_value="test"):
                append_log({"geomean_tps": 100}, kept=True, directory=root)
                append_log({"geomean_tps": 102}, kept=True, directory=root)
            self.assertAlmostEqual(read_log(root)[-1]["delta_pct"], 2)

    def test_budget(self):
        budget = Budget(60)
        budget.reserve(3600)
        budget.charge(1)
        with self.assertRaises(ValueError):
            budget.reserve(3600)

    def test_move_on(self):
        self.assertEqual(history_summary([{'item': 'cuda_graph_decode', 'proposer': 'codex', 'kept': True,
                                          'implemented_items': ['static_kv_cache', 'bypass_wrapper']}])['kept'],
                         ['bypass_wrapper', 'cuda_graph_decode', 'static_kv_cache'])
        records = [{"item": "static_kv_cache", "proposer": "agent", "kept": False}]*5
        self.assertEqual(history_summary(records)["reverts"]["static_kv_cache"], 5)
        proposal = Proposal("static_kv_cache", "faster", "1%", {"engine/engine.py": "content"}, "latency", "one sentence")
        with self.assertRaises(ValueError):
            validate_proposal(proposal, ["static_kv_cache"], records)

    def test_workload_selection_and_proposal_schema(self):
        self.assertEqual(len(select_workloads("all")), 6)
        with self.assertRaises(ValueError):
            select_workloads("private-official")
        with self.assertRaises(ValueError):
            Proposal.parse({"patch": "anything"})

    def test_retry_backoff(self):
        class RateLimit(Exception):
            status_code = 429
        attempts = []
        def call():
            attempts.append(1)
            if len(attempts) < 3:
                raise RateLimit()
            return 7
        with patch("neokernel.loop.time.sleep"):
            self.assertEqual(retry_call(call), 7)
        self.assertEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()
