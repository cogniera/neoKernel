import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from neokernel.guard import physics_floors
from neokernel.judge import CandidateError, aggregate, check_logits, make_prompt, measure, replay, summarize, validate_tokens
from neokernel.schema import Correctness, Sample, Workload


class JudgeTests(unittest.TestCase):
    def test_paired_native_order_cache_and_local_gates(self):
        from neokernel.judge import evaluate
        from pathlib import Path
        calls = []
        def measured(path, model_path, w, *args, **kwargs):
            calls.append((str(path), w.name))
            return [Sample([[1]], [[1], [1]], .1, .1, .2, 1)], 1, 1
        model = SimpleNamespace(parameters=lambda: [], config=SimpleNamespace(vocab_size=8))
        shapes = [Workload('public-0', 1, 1, 2), Workload('public-1', 1, 1, 2)]
        cache = {}
        with patch('neokernel.judge.declares_speculative', return_value=False), \
                patch('neokernel.judge.measure', side_effect=measured), \
                patch('neokernel.judge.replay_samples', return_value=Correctness()), \
                patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=100)):
            def run():
                return evaluate(Path('candidate'), 'model', shapes, 1, model,
                                SimpleNamespace(all_special_ids=[]), cache, baseline_dir=Path('native'))
            first = run()
            self.assertTrue(first.eligible)  # .1s is far above official public-0 TTFT, but equals local native.
            self.assertEqual(calls, [('native', 'public-0'), ('candidate', 'public-0'),
                                     ('candidate', 'public-1'), ('native', 'public-1')])
            calls.clear()
            self.assertTrue(run().eligible)
            self.assertEqual(calls, [('candidate', 'public-0'), ('candidate', 'public-1')])

    def test_pipe_and_decode_cpu_diagnostics(self):
        from neokernel.judge import native_record
        sample = Sample([[1]], [[1], [1]], .1, .1, .2, 1,
                        pipe_overhead_ms=[.2, .4], child_cpu_ms=[100, 7])
        with patch('neokernel.judge.replay_samples', return_value=Correctness()):
            record = native_record(Workload('tiny', 1, 1, 2), ([sample], 1, 1), None, 'json')
        self.assertAlmostEqual(record['pipe_overhead_ms']['mean'], .3)
        self.assertEqual(record['pipe_overhead_ms']['max'], .4)
        self.assertEqual(record['cpu_step_ms'], {'mean': 7, 'max': 7, 'count': 1})

    def test_load_and_warmup_peak_is_gated(self):
        w = Workload("tiny", 1, 1, 2)
        samples = [Sample([[1]], [[1], [1]], .1, .1, .2, 80, lifetime_peak_mem_bytes=91)]
        result = summarize(w, samples, Correctness(), 1, 1, 100,
                           {"ttft_median": .1, "tpot_median": .1}, physics_floors(1, 1, 1, 1, 2))
        self.assertEqual(result.failure_code, "memory_limit")
        self.assertEqual(result.peak_mem_frac, .91)

    def test_distinct_prompts_and_final_memory_request(self):
        calls = []
        class FakeChild:
            def __init__(self, *args, **kwargs):
                self.ready = False
            def receive(self, deadline):
                if not self.ready:
                    self.ready = True
                    return 0, {"kind": "ready"}
                return 0, {"kind": "memory", "peak_mem_bytes": 99}
            def sample(self, prompt, *args, **kwargs):
                calls.append(prompt)
                return Sample(prompt, [[0]], .1, 0, .1, 10)
            def send(self, message):
                self_request.append(message)
            def close(self):
                closed.append(True)
        self_request, closed = [], []
        with patch("neokernel.judge.Child", FakeChild), patch("neokernel.judge.make_prompt", side_effect=[
                [[1]], [[1]], [[2]], [[2]], [[3]]]):
            records, _, _ = measure(None, "unused", Workload("tiny", 1, 1, 1), 2, 4, [])
        self.assertEqual(calls, [[[1]], [[2]], [[3]]])
        self.assertEqual(self_request, [{"kind": "memory"}])
        self.assertEqual(records[-1].lifetime_peak_mem_bytes, 99)
        self.assertEqual(closed, [True])

    def test_yields(self):
        validate_tokens([[1, 2], [2, 3]], 2, 2, 4)
        for steps in [[], [[1, 2]], [[1, 2]] * 3, [[True, 1], [1, 2]], [(1, 2), [1, 2]],
                      [[1], [2]], [[-1, 1], [1, 2]], [[4, 1], [1, 2]], [[1., 2], [1, 2]]]:
            with self.subTest(steps=steps), self.assertRaises(CandidateError):
                validate_tokens(steps, 2, 2, 4)

    def test_gates(self):
        w = Workload("test", 1, 2, 3)
        records = [Sample([[1, 2]], [[1], [2], [3]], .1, .1, .3, 80) for _ in range(3)]
        floors = physics_floors(1, 1, 1, 2, 3)
        native = {"ttft_median": .1, "tpot_median": .1}
        result = summarize(w, records, Correctness(), 2, 1, 100, native, floors)
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.tps, 10)
        self.assertAlmostEqual(aggregate([result]).geomean_tps, 10)
        for field, value, code in [("ttft_s", .106, "latency_limit"), ("tpot_s", .106, "latency_limit"),
                                   ("peak_mem_bytes", 91, "memory_limit")]:
            with self.subTest(field=field):
                old = getattr(records[0], field)
                for s in records:
                    setattr(s, field, value)
                bad = summarize(w, records, Correctness(), 2, 1, 100, native, floors)
                self.assertEqual(bad.failure_code, code)
                self.assertIsNone(aggregate([bad]).geomean_tps)
                for s in records:
                    setattr(s, field, old)
        records[0].total_s = .5
        self.assertEqual(summarize(w, records, Correctness(), 2, 1, 100, native, floors).failure_code, "unstable_timing")
        self.assertEqual(summarize(w, records, Correctness(False), 2, 1, 100, native, floors).failure_code, "incorrect_output")
        self.assertEqual(summarize(w, records, Correctness(), 300, 1, 100, native, floors).failure_code, "load_budget")

    def test_physics_rejection_and_missing_native(self):
        w = Workload("test", 1, 1, 2)
        records = [Sample([[1]], [[1], [1]], .1, .1, .2, 1)]
        floor = physics_floors(int(3.35e12), 1, 1, 1, 2)
        native = {"ttft_median": .1, "tpot_median": .1}
        self.assertEqual(summarize(w, records, Correctness(), 1, 1, 100, native, floor).failure_code, "physics_violation")
        self.assertFalse(summarize(w, records, Correctness(), 1, 1, 100, None, floor).passed)
        self.assertTrue(summarize(w, records, Correctness(), 1, 1, 100, None, floor, True).passed)


@unittest.skipUnless(importlib.util.find_spec("torch"), "install CPU torch for tensor replay tests")
class TensorReplayTests(unittest.TestCase):
    def test_exact_ties_use_lowest_index(self):
        import torch
        logits = torch.tensor([[[5., 5., 1.]]])
        self.assertEqual(check_logits(logits, [[0]]).near_tie_count, 0)
        self.assertEqual(check_logits(logits, [[1]]).near_tie_count, 1)
        self.assertTrue(check_logits(logits, [[1]]).passed)

    def test_replay_positions_and_emitted_prefix(self):
        import torch
        class PositionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.received = []
            def forward(self, input_ids, use_cache):
                self.received.append((input_ids.tolist(), use_cache))
                length = input_ids.shape[1]
                logits = torch.full((1, length, 16), -10.)
                for i in range(length):
                    logits[0, i, i] = 10.
                return SimpleNamespace(logits=logits)
        model = PositionModel()
        result = replay(model, [[8, 9, 10]], [[2], [3], [4]])
        self.assertTrue(result.passed)
        self.assertEqual(model.received, [([[8, 9, 10, 2, 3, 4]], False)])
        self.assertEqual(replay(model, [[8, 9, 10]], [[3], [4], [5]]).first_bad_position, [0, 0])

    def test_prompt_rng_is_private_and_specials_excluded(self):
        import torch
        before = torch.random.get_rng_state().clone()
        w = Workload("tiny", 2, 64, 2)
        a = make_prompt(w, 8, [0, 3, 7], 123)
        self.assertEqual(a, make_prompt(w, 8, [0, 3, 7], 123))
        self.assertNotEqual(a, make_prompt(w, 8, [0, 3, 7], 124))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertFalse({0, 3, 7} & {token for row in a for token in row})

    def test_margin_boundary_nonfinite_and_near_ties(self):
        import torch
        logits = torch.tensor([[[4., 2., 1.], [1., 4., 1.]]])
        self.assertTrue(check_logits(logits, [[1, 1]]).passed)
        self.assertEqual(check_logits(logits, [[1, 1]]).near_tie_count, 1)
        self.assertEqual(check_logits(logits, [[2, 1]]).first_bad_position, [0, 0])
        logits[0, 0, 0] = float("nan")
        self.assertFalse(check_logits(logits, [[1, 1]]).passed)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "install transformers==4.51.3 for tiny Qwen3 replay")
    def test_tiny_qwen3_replay_own_prefix(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128,
                                           num_hidden_layers=2, num_attention_heads=4,
                                           num_key_value_heads=2, head_dim=16)).eval()
        prompt = [[3, 5, 8], [4, 6, 9]]
        current = torch.tensor(prompt)
        steps = []
        with torch.inference_mode():
            for _ in range(4):
                next_ids = model(current).logits[:, -1].argmax(-1)
                steps.append(next_ids.tolist())
                current = torch.cat([current, next_ids[:, None]], dim=1)
        self.assertTrue(replay(model, prompt, steps).passed)


if __name__ == "__main__":
    unittest.main()
