import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neokernel.report import generate, outcome


class ReportTests(unittest.TestCase):
    def test_report_is_offline_before_guard_or_remote(self):
        from neokernel.cli import main
        with patch('neokernel.report.generate', return_value=[]) as generate_report, \
                patch('neokernel.cli.check', side_effect=AssertionError('guard called')), \
                patch('neokernel.cli.Remote', side_effect=AssertionError('GPU called')):
            self.assertEqual(main(['report']), 0)
            generate_report.assert_called_once_with()

    def test_evidence_units_missing_values_and_spend_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results, docs = root / 'results', root / 'docs'
            (results / 'runs').mkdir(parents=True)
            def save(name, value):
                (results / name).write_text(json.dumps(value), encoding='utf-8')
            save('dryft_runs.json', {'runs': [{'label': 'native', 'run_id': 'abc', 'score': 217.3,
                 'rank': 12, 'source': 'run page', 'workloads': {'public-0': {'ttft_ms': 22.01, 'tpot_ms': 17.41, 'tps': 57}}}]})
            row = {'id': 18, 'proposer': 'codex', 'item': 'decode', 'guard': 'pass', 'kept': False,
                   'geomean_tps': None, 'gpu_seconds': 0, 'note': 'failed | latency\nfresh prompt',
                   'workloads': [{'name': 'public-0', 'failure_code': 'latency_limit'}]}
            (results / 'log.jsonl').write_text(json.dumps(row) + '\n')
            w = {'name': 'public-0', 'batch': 1, 'S': 512, 'N': 32, 'tps': 200,
                 'floors': {'step_floor_s': .0024, 'prefill_floor_s': .0068, 'decode_floor_s': .0744,
                            'floor_tps': 400, 'measured_to_floor_ratio': .5}}
            save('runs/old.json', {'gpu_tier': 'H100', 'eligible': True, 'ts': '2026-01', 'workloads': [w]})
            save('runs/new_failed.json', {'gpu_tier': 'H100', 'eligible': False, 'ts': '2026-02', 'workloads': [{**w, 'tps': 999}]})
            save('profile.json', {'workload': 'public-0', 'gpu_tier': 'L4', 'sum_kernel_ms': 3,
                                 'gap_ms': 2, 'kernels': [{'calls': 5}, {'calls': 7}]})
            save('profile_trace.json', {'traceEvents': []})
            save('spend.json', {'setup_estimated_usd': .1, 'calls': [
                {'tier': 'H100', 'estimated_allocation_s': 20, 'estimated_usd': .2},
                {'tier': 'L4', 'estimated_allocation_s': 10, 'estimated_usd': .03}]})
            save('setup.json', {'estimated_usd': .1, 'attempts': [{}]})
            paths = generate(results, docs)
            report, log = [p.read_text(encoding='utf-8') for p in paths]
            self.assertIn('22.01 | 17.41 | 57.0', report)
            self.assertIn('not recorded', report)
            self.assertIn('2.400 | 6.800 | 74.400 | 400.000 | 200.000 | 0.5000', report)
            self.assertIn('12 | 3.000 | 2.000', report)
            self.assertNotIn('profile_trace.json', report)
            self.assertIn('0.330000', report)
            self.assertIn('failed: latency_limit', log)
            self.assertIn('failed \\| latency<br>fresh prompt', log)
            self.assertEqual([p.read_text(encoding='utf-8') for p in generate(results, docs)], [report, log])

    def test_not_kept_is_not_automatically_failure(self):
        self.assertEqual(outcome({'guard': 'pass', 'kept': False, 'geomean_tps': 5}), 'measured, not kept')
        self.assertEqual(outcome({'guard': 'patch_failed', 'kept': False, 'note': 'patch_failed: corrupt patch'}), 'patch rejected')
        self.assertEqual(outcome({'guard': 'pass', 'kept': False, 'note': 'failed after 2 repairs', 'repairs': 2}),
                         'failed: unit tests after 2 repairs')
        self.assertEqual(outcome({'guard': 'not_run', 'kept': False, 'note': 'interrupted; recovered without merge'}),
                         'interrupted / recovered')


if __name__ == '__main__':
    unittest.main()
