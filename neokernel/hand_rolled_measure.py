"""Attended H100 measurements through the existing judge and profiler.

Each invocation requires operator approval. No gate or reference is replaced.
compare: three public samples per attention implementation, choose an eligible
winner by geometric-mean throughput. bench: three samples on all six workloads.
profile: before/after traces for public-0 and public-2. No automatic freeze.
"""

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import time

from agent.package import package
from .accounting import SpendLedger, estimate_usd, GPU_TIMEOUT_S, GPU_IDLE_S
from .calibration import equivalent_estimates, load_calibration
from .cli import Remote, report
from .guard import check
from .schema import PUBLIC, select_workloads
from .storage import ROOT, RESULTS, append_log, write_json


def task_spend():
    task = json.loads((RESULTS / 'hand_rolled_budget.json').read_text())
    spent = SpendLedger().read()['estimated_usd'] - task['start_estimated_usd']
    return spent, task['limit_usd']


def reserve_task(calls=1):
    spent, limit = task_spend()
    reservation = calls * estimate_usd('H100', GPU_TIMEOUT_S + 60 + GPU_IDLE_S)
    if spent + reservation > limit:
        raise RuntimeError(f'Task budget: ${spent:.4f} spent + ${reservation:.4f} reserved exceeds ${limit:.2f}')
    return reservation


def check_spend():
    spent, limit = task_spend()
    print(f'Hand-rolled task estimated spend: ${spent:.4f} / ${limit:.2f}', flush=True)
    if spent >= limit:
        raise RuntimeError('Task budget reached; stop')


def choose_attention(source, implementation):
    tree = ast.parse(source)
    node, = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in n.targets)]
    config = ast.literal_eval(node.value)
    config['attention_impl'] = implementation
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1]) + 'CONFIG = ' + repr(config) + '\n' + ''.join(lines[node.end_lineno:])


def save_estimates(name, result):
    calibration = load_calibration()
    write_json(RESULTS / name, dict(result=result, calibration=calibration,
               dryft_equivalent_estimates={w['name']: equivalent_estimates(w, calibration)
                                           for w in result['workloads']}))


def benchmark(remote, payload, workloads, label, samples=3):
    reserve_task()
    result = remote.bench(payload, workloads, samples=samples, refresh_native=True)
    save_estimates(f'hand_rolled_{label}.json', result)
    report(result)
    row = append_log(result, proposer='codex', item='hand_rolled_decode', kept=False,
                     hypothesis='Packed projections and fused decode kernels reduce graph work while preserving native BF16 semantics.',
                     note=f'H100 {label}; selection, final freeze, and measured target review still required before keep/commit.',
                     files_changed=[p.relative_to(ROOT).as_posix() for p in (ROOT / 'engine').rglob('*.py')])
    print(f'codex/hand_rolled_decode log ID: {row["id"]}')
    check_spend()
    return result


def profile(remote, payload, workload, label):
    reserve_task()
    remote.spend.reserve('H100')
    started = time.perf_counter()
    result = None
    try:
        result = remote.api.profile_h100_remote.remote(payload, asdict(workload))
        write_json(RESULTS / f'hand_rolled_profile_{label}_{workload.name}.json', result)
    finally:
        remote.spend.record('H100', started, result.get('gpu_seconds') if result else None,
                            f'hand_rolled_profile_{label}_{workload.name}', result is not None)
    print(json.dumps({k: result.get(k) for k in
                      ('workload', 'kernel_count', 'sum_kernel_ms', 'gap_ms', 'graph_launch_count', 'sync_calls')}), flush=True)
    check_spend()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('compare', 'bench', 'profile'))
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--workloads', default='all')
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError('samples must be positive')
    check(ROOT / 'engine')
    # Reserve a complete comparison before launching either side. Profile and
    # bench reserve each call and stop whenever the task cannot fund its bound.
    reserve_task(2 if args.stage == 'compare' else 1)
    original = package(ROOT / 'engine')
    with Remote() as remote:
        if args.stage == 'compare':
            outcomes = {}
            init = ROOT / 'engine/kernels/__init__.py'
            source = init.read_text(encoding='utf-8')
            with tempfile.TemporaryDirectory() as tmp:
                for implementation in ('triton', 'sdpa_grouped'):
                    staged = Path(tmp) / implementation
                    shutil.copytree(ROOT / 'engine', staged, ignore=shutil.ignore_patterns('__pycache__'))
                    (staged / 'kernels/__init__.py').write_text(choose_attention(source, implementation), encoding='utf-8')
                    check(staged)
                    outcomes[implementation] = benchmark(remote, package(staged), PUBLIC, implementation)
            eligible = {k: v for k, v in outcomes.items() if v['eligible']}
            if not eligible:
                print('No attention candidate passed all public gates; active CONFIG unchanged.')
                return 1
            winner = max(eligible, key=lambda k: eligible[k]['geomean_tps'])
            if package(ROOT / 'engine') != original:
                raise RuntimeError('Active engine changed during comparison; no selection applied')
            init.write_text(choose_attention(source, winner), encoding='utf-8')
            write_json(RESULTS / 'hand_rolled_attention_choice.json',
                       dict(winner=winner, scores={k: v['geomean_tps'] for k, v in outcomes.items()},
                            eligible={k: v['eligible'] for k, v in outcomes.items()}))
            print(f'Selected {winner}; full six-workload benchmark and freeze still required.')
        elif args.stage == 'bench':
            label = 'full_bench' if args.workloads == 'all' else 'bench_' + args.workloads.replace(',', '_')
            result = benchmark(remote, original, select_workloads(args.workloads), label, args.samples)
            return 0 if result['eligible'] else 1
        else:
            before = RESULTS / 'hand_rolled_before/engine'
            check(before)
            for label, payload in (('before', package(before)), ('after', original)):
                for workload in (PUBLIC[0], PUBLIC[2]):
                    profile(remote, payload, workload, label)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
