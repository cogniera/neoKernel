"""Exercise full orchestration with a fake remote, never provider services."""

import argparse
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from neokernel.loop import run_loop
from neokernel.schema import Proposal
from neokernel.storage import append_log, read_log, restore, snapshot

ENGINE = b'class Engine:\n    def __init__(self, model_path): pass\n    def generate(self, input_ids, max_new_tokens):\n        yield [1]\n'
PATCH = '--- a/engine/engine.py\n+++ b/engine/engine.py\n@@ -4 +4 @@\n-        yield [1]\n+        yield [2]\n'


class FakeRemote:
    def __init__(self, score=102, interrupt=False):
        self.calls = 0
        self.score = score
        self.interrupt = interrupt

    def reserve(self, seconds):
        pass

    def profile(self, payload, workload):
        return {"step_wall_ms": 3, "gap_ms": 1, "kernels": [], "gpu_seconds": 1}

    def bench(self, payload, workloads, samples, correctness_only=False):
        self.calls += 1
        if self.interrupt and self.calls == 3:
            raise KeyboardInterrupt()
        return {"eligible": True, "geomean_tps": 100 if self.calls == 1 else self.score,
                "workloads": [], "gpu_seconds": 1}


class FakeProposer:
    def __init__(self, patch_text=PATCH):
        self.patch_text = patch_text

    def propose(self, messages):
        return Proposal("static_kv_cache", "test hypothesis", "2 percent", self.patch_text, "latency", "test reasoning")


class LoopTests(unittest.TestCase):
    def run_case(self, score=102, interrupt=False, bad_guard=False):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            engine = root/"engine"
            results = root/"results"
            restore(engine, {"engine.py": ENGINE})
            original = snapshot(engine)
            stack.enter_context(patch("neokernel.loop.ROOT", root))
            stack.enter_context(patch("neokernel.loop.RESULTS", results))
            stack.enter_context(patch("neokernel.loop.read_log", side_effect=lambda: read_log(results)))
            stack.enter_context(patch("neokernel.loop.append_log", side_effect=lambda *a, **kw: append_log(*a, **kw, directory=results)))
            stack.enter_context(patch("neokernel.storage.git_sha", return_value="unchanged-head"))
            args = argparse.Namespace(items=None, workloads="public", model="fake", steps=1, attended=False)
            remote = FakeRemote(score, interrupt)
            proposer = FakeProposer(PATCH.replace("yield [2]", "eval('1')") if bad_guard else PATCH)
            if interrupt:
                with self.assertRaises(KeyboardInterrupt):
                    run_loop(args, remote, proposer)
            else:
                self.assertEqual(run_loop(args, remote, proposer), 0)
            records = read_log(results)
            if score > 101 and not interrupt and not bad_guard:
                self.assertTrue(records[-1]["kept"])
                self.assertNotEqual(snapshot(engine), original)
            else:
                self.assertFalse(records[-1]["kept"])
                self.assertEqual(snapshot(engine), original)
            if bad_guard:
                self.assertEqual(remote.calls, 1)
                self.assertEqual(records[-1]["guard"], "fail")

    def test_keep(self):
        self.run_case()

    def test_revert_small_improvement(self):
        self.run_case(score=101)

    def test_revert_interrupt(self):
        self.run_case(interrupt=True)

    def test_revert_guard_failure(self):
        self.run_case(bad_guard=True)


if __name__ == "__main__":
    unittest.main()
