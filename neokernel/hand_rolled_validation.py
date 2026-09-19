"""Explicit attended L4 unit validation using the existing pinned Modal funnel.

Does not change the judge, guard, gates, engine, or existing Modal entry points.
Run only with operator approval: py -3.11 -m neokernel.hand_rolled_validation
"""

import io
import argparse
from pathlib import Path
import tempfile
import time

from . import modal_app
from .accounting import SpendLedger, estimate_usd, GPU_TIMEOUT_S, GPU_IDLE_S
from .guard import lint_source, check
from .storage import ROOT, RESULTS, write_json


@modal_app.app.function(**modal_app.L4_REMOTE)
def validate_remote(sources: dict[str, str]) -> dict:
    import contextlib
    import unittest
    started = time.perf_counter()
    modal_app.verify_runtime()
    modal_app.cpu_diagnostics()
    output = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, source in sources.items():
            if name not in {
                'engine/engine.py',
                'engine/kernels/__init__.py',
                'engine/kernels/elementwise.py',
                'engine/kernels/attention.py',
                'engine/kernels/decode.py',
                'engine/kernels/rmsnorm.py',
                'engine/kernels/weights.py',
                'tests/test_hand_rolled_kernels.py',
                'tests/test_hand_rolled_handoff.py',
            }:
                raise ValueError('unexpected validation source')
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(source, encoding='utf-8')
        suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_hand_rolled*.py')
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    return dict(passed=result.wasSuccessful() and not result.skipped,
                tests=result.testsRun, skipped=len(result.skipped),
                output=output.getvalue(), gpu_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration', action='store_true',
                        help='validate full-model handoff, then run the existing public L4 check if tests pass')
    args = parser.parse_args()
    check(ROOT / 'engine')
    sources = {}
    for path in sorted((ROOT / 'engine/kernels').glob('*.py')):
        source = path.read_text(encoding='utf-8')
        lint_source(source, path.name, {'kernels'})
        sources[path.relative_to(ROOT).as_posix()] = source
    test = ROOT / 'tests/test_hand_rolled_kernels.py'
    sources[test.relative_to(ROOT).as_posix()] = test.read_text(encoding='utf-8')
    if args.integration:
        for name in ('engine/engine.py', 'tests/test_hand_rolled_handoff.py'):
            sources[name] = (ROOT / name).read_text(encoding='utf-8')
    ledger = SpendLedger()
    task_path = RESULTS / 'hand_rolled_budget.json'
    import json
    if task_path.exists():
        task = json.loads(task_path.read_text())
    else:
        task = dict(start_estimated_usd=ledger.read()['estimated_usd'], limit_usd=4.0)
        write_json(task_path, task)
    spent = ledger.read()['estimated_usd'] - task['start_estimated_usd']
    reservation = estimate_usd('L4', GPU_TIMEOUT_S + 60 + GPU_IDLE_S)
    if args.integration:
        reservation *= 2
    if spent + reservation > task['limit_usd']:
        raise RuntimeError(f'Task budget: ${spent:.4f} spent; ${reservation:.4f} reservation exceeds $4')
    ledger.reserve('L4')
    result = None
    started = time.perf_counter()
    try:
        with modal_app.app.run():
            result = validate_remote.remote(sources)
        write_json(RESULTS / ('hand_rolled_integration_tests.json' if args.integration else 'hand_rolled_unit_tests.json'), result)
        print(result['output'])
    finally:
        ledger.record('L4', started, result['gpu_seconds'] if result else None,
                      'hand_rolled_unit_tests', result['passed'] if result else False)
    spent = ledger.read()['estimated_usd'] - task['start_estimated_usd']
    print(f'Hand-rolled task estimated spend: ${spent:.4f} / $4.00')
    if spent >= task['limit_usd']:
        raise RuntimeError('Task budget reached; stop')
    if result['passed'] and args.integration:
        from agent.package import package
        from .cli import Remote, report
        from .schema import PUBLIC
        if spent + estimate_usd('L4', GPU_TIMEOUT_S + 60 + GPU_IDLE_S) > task['limit_usd']:
            raise RuntimeError('Task budget cannot reserve public correctness check')
        with Remote() as remote:
            checked = remote.bench(package(ROOT / 'engine'), PUBLIC, 1, correctness_only=True)
        write_json(RESULTS / 'hand_rolled_public_check.json', checked)
        report(checked, correctness_only=True)
        spent = ledger.read()['estimated_usd'] - task['start_estimated_usd']
        print(f'Hand-rolled task estimated spend: ${spent:.4f} / $4.00')
        if spent >= task['limit_usd']:
            raise RuntimeError('Task budget reached; stop')
        return 0 if checked['eligible'] else 1
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
