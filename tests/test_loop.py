"""Real Git transaction tests with fake judge JSON; no provider calls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from neokernel import loop
from neokernel.schema import Proposal
from neokernel.storage import read_log, restore, snapshot
from neokernel.transaction import Transaction, git, credential_scan
from neokernel.accounting import SpendLedger, SpendLimit, start_night, GPU_IDLE_S
from neokernel.judge import keep_decision

ENGINE = b'class Engine:\n    def __init__(self, model_path): pass\n    def generate(self, input_ids, max_new_tokens):\n        yield [1]\n'
FILES = {'engine/engine.py': ENGINE.decode().replace('yield [1]', 'yield [2]')}


def repo(root):
    restore(root/'engine', {'engine.py': ENGINE})
    (root/'.gitignore').write_text('results/\nresults_backup/\n__pycache__/\n')
    git(root, 'init', '-b', 'main')
    git(root, 'config', 'core.autocrlf', 'false')
    git(root, 'config', 'user.name', 'Test')
    git(root, 'config', 'user.email', 'test@example.invalid')
    git(root, 'add', '.')
    credential_scan(root)
    git(root, 'commit', '-m', 'initial')


def result(score=100, passed=True):
    return dict(eligible=passed, geomean_tps=score if passed else None, gpu_seconds=1,
                workloads=[dict(name='public-0', batch=1, S=512, N=32, tps=score,
                                passed=passed, gates={'correctness': passed},
                                failure_code=None if passed else 'incorrect_output')])


UNIT_FAILURE = '''test_attention (test_hand_rolled_kernels.HandRolledKernelTests.test_attention) ... ERROR
test_norm_and_residual (test_hand_rolled_kernels.HandRolledKernelTests.test_norm_and_residual) ... {"test": "norm/1/False", "max_abs_diff": 0.0}
ok

======================================================================
ERROR: test_attention (test_hand_rolled_kernels.HandRolledKernelTests.test_attention)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/x/tests/test_hand_rolled_kernels.py", line 137, in test_attention
    attention_out(q, k, v, position, buf.attn, buf.partial, buf.lse)
triton.compiler.errors.CompilationError: at 12:8: d = tl.arange(0, D)

----------------------------------------------------------------------
Ran 2 tests in 1.000s

FAILED (errors=1)
'''


class FakeRemote:
    def __init__(self, score=102, fail=False, crash=False):
        self.calls, self.score, self.fail, self.crash = 0, score, fail, crash
        self.unit_calls, self.unit_sources = 0, []

    def unit_tests(self, sources):
        # Stubbed L4 suite: a candidate passes once its engine yields [3].
        self.unit_calls += 1
        self.unit_sources.append(sources)
        if self.unit_calls == getattr(self, 'unit_crash_on', None):
            raise RuntimeError('unit test harness crash')
        passed = not getattr(self, 'unit_fail', False) or 'yield [3]' in sources['engine/engine.py']
        return dict(passed=passed, tests=2, skipped=0, returncode=0 if passed else 1, timed_out=False,
                    output='Ran 2 tests in 1.000s\n\nOK\n' if passed else UNIT_FAILURE, gpu_seconds=7, gpu_tier='L4')

    def bench(self, *a, **kw):
        self.calls += 1
        if self.crash and self.calls > 1:
            raise RuntimeError('harness test crash')
        return result(100 if self.calls == 1 else self.score, not (self.fail and self.calls > 1))


class FakeProposer:
    def __init__(self, files=None, repairs=()):
        self.files = files or FILES
        self.repairs, self.conversations = list(repairs), []
    def propose(self, messages):
        return Proposal('lm_head_argmax_tiled', 'test hypothesis', '2 percent', self.files, 'latency', 'test reasoning')
    def repair(self, conversation):
        self.conversations.append(list(conversation))
        files = self.repairs.pop(0)
        if files is None:
            raise ValueError('invalid proposal after one retry: not json')
        return Proposal('lm_head_argmax_tiled', 'test hypothesis', '2 percent', files, 'latency', 'repaired')


class LoopTests(unittest.TestCase):
    def run_case(self, score=102, fail=False, crash=False, bad_guard=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo(root)
            initial = git(root, 'rev-parse', 'HEAD')
            args = argparse.Namespace(items=None, workloads='public-0', model='fake', steps=1, attended=False, resume=False)
            with patch.object(loop, 'ROOT', root), patch.object(loop, 'RESULTS', root/'results'):
                proposer = FakeProposer({'engine/engine.py': FILES['engine/engine.py'].replace('yield [2]', "eval('1')")} if bad_guard else None)
                if crash:
                    with self.assertRaises(RuntimeError):
                        loop.run_loop(args, FakeRemote(crash=True), proposer)
                    self.assertEqual(git(root, 'branch', '--show-current'), 'exp/1')
                    self.assertEqual(git(root, 'rev-parse', 'main'), initial)
                    self.assertIn('harness test crash', (root/'results/CRASH.txt').read_text())
                    self.assertNotEqual(snapshot(root/'engine'), {'engine.py': ENGINE})
                    return
                self.assertEqual(loop.run_loop(args, FakeRemote(score, fail), proposer), 0)
            row = read_log(root/'results')[-1]
            kept = score > 101 and not fail and not bad_guard
            self.assertEqual(row['kept'], kept)
            self.assertEqual(git(root, 'branch', '--show-current'), 'main')
            self.assertFalse(git(root, 'branch', '--list', 'exp/*'))
            self.assertTrue((root/'results/snapshots/1/engine/engine.py').exists())
            self.assertTrue(row['diff'])
            self.assertEqual((row['repairs'], row['first_error']), (0, None))
            if kept:
                self.assertEqual(git(root, 'rev-parse', 'kept-1'), git(root, 'rev-parse', 'main'))
                best = json.loads((root/'results/best.json').read_text())
                self.assertEqual(best['id'], 1)
                self.assertEqual(best['per_workload_tps'], {'public-0': score})
                self.assertTrue((root/'results_backup/1/log.jsonl').exists())
                self.assertTrue((root/'results_backup/1/snapshots/1/engine/engine.py').exists())
            else:
                self.assertEqual(git(root, 'rev-parse', 'main'), initial)
                self.assertEqual(snapshot(root/'engine'), {'engine.py': ENGINE})

    def test_keep(self): self.run_case()
    def test_revert_small_improvement(self): self.run_case(score=101)
    def test_revert_slower(self): self.run_case(score=99)
    def test_revert_candidate_failure(self): self.run_case(fail=True)
    def test_guard_revert(self): self.run_case(bad_guard=True)
    def test_harness_crash_preserved(self): self.run_case(crash=True)

    def test_candidate_failure_continues_and_ceiling_stops_cleanly(self):
        for ceiling in [False, True]:
            with self.subTest(ceiling=ceiling), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); repo(root)
                args = argparse.Namespace(items=None, workloads='public-0', model='fake', steps=2, resume=False)
                remote = FakeRemote(fail=True)
                if ceiling:
                    remote.bench = lambda *a, **kw: (_ for _ in ()).throw(SpendLimit('ceiling')) if kw else result()
                with patch.object(loop, 'ROOT', root), patch.object(loop, 'RESULTS', root/'results'):
                    self.assertEqual(loop.run_loop(args, remote, FakeProposer()), 0)
                self.assertEqual(len(read_log(root/'results')), 1 if ceiling else 2)
                self.assertEqual(git(root, 'branch', '--show-current'), 'main')
                self.assertFalse((root/'results/CRASH.txt').exists())
                self.assertEqual((root/'results/STOP.json').exists(), ceiling)


    def repair_case(self, repairs, score=102, unit_crash_on=None):
        """One agent step whose proposal fails the stubbed L4 suite until an engine yields [3]."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); repo(root)
            args = argparse.Namespace(items=None, workloads='public-0', model='fake', steps=1, resume=False)
            remote = FakeRemote(score); remote.unit_fail = True; remote.unit_crash_on = unit_crash_on
            proposer = FakeProposer(repairs=repairs)
            with patch.object(loop, 'ROOT', root), patch.object(loop, 'RESULTS', root/'results'):
                if unit_crash_on:
                    with self.assertRaises(RuntimeError):
                        loop.run_loop(args, remote, proposer)
                    self.assertEqual(git(root, 'branch', '--show-current'), 'exp/1')
                    state = json.loads((root/'results/transaction.json').read_text())
                    self.assertEqual(state['repairs'], 1)
                    self.assertIn('CompilationError', state['first_error'])
                    Transaction(root, root/'results').startup(resume=True)
                    self.assertEqual(git(root, 'branch', '--show-current'), 'main')
                    self.assertFalse(git(root, 'status', '--porcelain'))
                else:
                    self.assertEqual(loop.run_loop(args, remote, proposer), 0)
            rows = read_log(root/'results')
            self.assertEqual(len(rows), 1)
            self.assertEqual(git(root, 'branch', '--show-current'), 'main')
            self.assertFalse(git(root, 'branch', '--list', 'exp/*'))
            return rows[0], remote, proposer, sorted(p.name for p in (root/'results/unit_tests').glob('*.json')), \
                sorted(p.name for p in (root/'results/proposals').glob('*.json'))

    def test_repair_fixes_then_judges_and_keeps(self):
        fixed = {'engine/engine.py': ENGINE.decode().replace('yield [1]', 'yield [3]')}
        row, remote, proposer, units, proposals = self.repair_case([fixed])
        self.assertTrue(row['kept'])
        self.assertEqual((row['repairs'], remote.unit_calls, remote.calls), (1, 2, 3))
        self.assertIn('CompilationError', row['first_error'])
        self.assertEqual(units, ['1-0.json', '1-1.json'])
        self.assertEqual(proposals, ['1-repair1.json', '1.json'])
        self.assertEqual(row['gpu_seconds'], 1 + 1 + 7 + 7)
        self.assertIn('yield [3]', row['diff'])
        conversation = proposer.conversations[0]
        self.assertEqual([m['role'] for m in conversation[-2:]], ['assistant', 'user'])
        self.assertIn('yield [2]', conversation[-2]['content'])
        self.assertIn('Repair turn 1 of 2', conversation[-1]['content'])
        self.assertIn('CompilationError', conversation[-1]['content'])
        self.assertNotIn('max_abs_diff', conversation[-1]['content'])
        self.assertIn('tests/test_hand_rolled_kernels.py', remote.unit_sources[0])
        self.assertIn('tests/test_hand_rolled_handoff.py', remote.unit_sources[0])
        self.assertNotIn('Remote', remote.unit_sources[0]['tests/test_hand_rolled_handoff.py'])

    def test_reverts_after_two_failed_repairs(self):
        still = [{'engine/engine.py': ENGINE.decode().replace('yield [1]', f'yield [{n}]')} for n in (4, 5)]
        row, remote, proposer, units, proposals = self.repair_case(still)
        self.assertFalse(row['kept'])
        self.assertEqual(row['note'], 'failed after 2 repairs')
        self.assertEqual((row['repairs'], remote.unit_calls, remote.calls), (2, 3, 1))
        self.assertEqual(row['geomean_tps'], None)
        self.assertEqual(row['gpu_seconds'], 21)
        self.assertIn('yield [5]', row['diff'])
        self.assertEqual(units, ['1-0.json', '1-1.json', '1-2.json'])
        self.assertIn('Repair turn 2 of 2', proposer.conversations[1][-1]['content'])

    def test_guard_rejected_and_invalid_repairs_consume_turns(self):
        bad = {'engine/engine.py': ENGINE.decode().replace('yield [1]', "eval('3')")}
        fixed = {'engine/engine.py': ENGINE.decode().replace('yield [1]', 'yield [3]')}
        row, remote, proposer, units, _ = self.repair_case([bad, fixed])
        self.assertTrue(row['kept'])
        self.assertEqual((row['repairs'], remote.unit_calls), (2, 2))
        self.assertIn('Guard rejected the repaired files', proposer.conversations[1][-1]['content'])
        self.assertIn('eval is forbidden', proposer.conversations[1][-1]['content'])
        self.assertNotIn('eval', remote.unit_sources[1]['engine/engine.py'])
        row, remote, proposer, units, _ = self.repair_case([None, fixed])
        self.assertTrue(row['kept'])
        self.assertEqual((row['repairs'], remote.unit_calls), (2, 2))
        self.assertIn('not valid proposal JSON', proposer.conversations[1][-1]['content'])

    def test_crash_during_retest_is_journaled_and_recoverable(self):
        fixed = {'engine/engine.py': ENGINE.decode().replace('yield [1]', 'yield [3]')}
        row, remote, proposer, units, _ = self.repair_case([fixed], unit_crash_on=2)
        self.assertFalse(row['kept'])
        self.assertEqual(row['note'], 'interrupted; recovered without merge')
        self.assertEqual(row['repairs'], 1)
        self.assertIn('CompilationError', row['first_error'])
        self.assertIn('yield [3]', row['diff'])

    def test_unit_output_summary_and_local_suite_runner(self):
        first, excerpt = loop.summarize_unit_output(UNIT_FAILURE)
        self.assertTrue(first.startswith('ERROR: test_attention'))
        self.assertIn('CompilationError', first)
        self.assertIn('FAILED (errors=1)', excerpt)
        self.assertNotIn('max_abs_diff', excerpt)
        self.assertEqual(loop.summarize_unit_output('Segmentation fault\n'), ('Segmentation fault\n', 'Segmentation fault\n'))
        from neokernel import modal_app
        for name, ok in [('engine/engine.py', True), ('engine/kernels/lm_head_argmax.py', True), ('tests/test_hand_rolled_x.py', True),
                         ('tests/test_loop.py', False), ('engine/kernels/../engine.py', False), ('neokernel/judge.py', False),
                         ('engine/kernels/x.txt', False), ('/engine/engine.py', False)]:
            self.assertEqual(modal_app.unit_source_ok(name), ok, name)
        stub = ('import sys, unittest\nfrom pathlib import Path\nclass T(unittest.TestCase):\n    def test_engine(self):\n'
                '        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))\n        import engine\n'
                '        self.assertEqual(list(engine.Engine("").generate([[1]], 1)), [[%s]])\n')
        with tempfile.TemporaryDirectory() as tmp:
            sources = {'engine/engine.py': ENGINE.decode(), 'tests/test_hand_rolled_stub.py': stub % 1}
            passing = modal_app.run_unit_suite(sources, Path(tmp) / 'a')
            self.assertEqual((passing['passed'], passing['tests'], passing['skipped']), (True, 1, 0))
            failing = modal_app.run_unit_suite({**sources, 'tests/test_hand_rolled_stub.py': stub % 2}, Path(tmp) / 'b')
            self.assertEqual((failing['passed'], failing['returncode']), (False, 1))
            self.assertIn('FAIL: test_engine', failing['output'])
            with self.assertRaises(ValueError):
                modal_app.run_unit_suite({**sources, 'tests/test_loop.py': ''}, Path(tmp) / 'c')

    def test_sweep_threshold_and_numeric_coordinates(self):
        from neokernel import sweep
        source = 'TUNABLES = {"norm.BLOCK": 4096, "attention.BLOCK": 128}\n'
        choices = sweep.numeric_candidates(source, 10)
        self.assertTrue(choices)
        self.assertTrue(all(c['norm.BLOCK'] >= 4096 for c in choices))
        for score in [101, 102]:
            with self.subTest(score=score), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); repo(root)
                restore(root/'engine/kernels', {'__init__.py': source.encode()})
                git(root, 'add', 'engine'); credential_scan(root); git(root, 'commit', '-m', 'tunables')
                args = argparse.Namespace(workloads='public-0', steps=1, random=False, seed=0, wire_rmsnorm=False, resume=False)
                with patch.object(sweep, 'ROOT', root), patch.object(sweep, 'RESULTS', root/'results'):
                    self.assertEqual(sweep.run_sweep(args, FakeRemote(score)), 0)
                self.assertEqual(read_log(root/'results')[-1]['kept'], score == 102)

    def test_startup_and_credential_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); repo(root)
            tx = Transaction(root, root/'results')
            (root/'untracked').write_text('keep me')
            with self.assertRaises(ValueError): tx.startup()
            (root/'untracked').unlink()
            git(root, 'switch', '-c', 'other')
            with self.assertRaises(ValueError): tx.startup()
            git(root, 'switch', 'main')
            (root/'untracked').write_text('dryft' + '_pat')
            git(root, 'add', 'untracked')
            with self.assertRaises(RuntimeError): credential_scan(root)

    def test_judge_alone_and_merge_assert(self):
        self.assertFalse(keep_decision(result(101), result())[0])
        self.assertFalse(keep_decision(result(999, False), result())[0])
        two = result(100); two['workloads'].append(dict(two['workloads'][0], name='public-2', tps=1000))
        mixed = result(110); mixed['workloads'].append(dict(mixed['workloads'][0], name='public-2', tps=960))
        self.assertFalse(keep_decision(mixed, two)[0])  # +10% geomean, public-2 down 4%: not kept
        mixed['workloads'][1]['tps'] = 975
        self.assertTrue(keep_decision(mixed, two)[0])
        wrong = result(999); wrong['workloads'][0]['name'] = 'other'
        self.assertFalse(keep_decision(wrong, result())[0])
        wrong['workloads'][0]['failure_code'] = 'harness_error'
        with self.assertRaises(RuntimeError): keep_decision(wrong, result())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); repo(root)
            tx = Transaction(root, root/'results'); tx.begin('test', 'agent')
            loop.apply_proposal(root/'engine', FILES)
            with self.assertRaises(AssertionError):
                tx.finish(tx.row(result(), True, 0, 'pass', 'forged keep', 'diff'))
            self.assertEqual(git(root, 'branch', '--show-current'), 'exp/1')

    def test_shared_budget_reserves_before_dispatch_and_survives_restart(self):
        self.assertEqual(GPU_IDLE_S, 60)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); start_night(directory)
            a = SpendLedger(directory); a.reserve_amount('Baseten', 1, 'pending')
            b = SpendLedger(directory); b.reserve_amount('H100', 6.9, 'pending')
            self.assertAlmostEqual(SpendLedger(directory).read()['estimated_usd'], 7.9)
            with self.assertRaises(SpendLimit): SpendLedger(directory).reserve_amount('L4', .11, 'pending')
            b.settle(2)
            self.assertEqual(a.read()['estimated_usd'], 3)
            self.assertEqual(len(a.read()['calls']), 2)

    def test_persistent_api_pause(self):
        class Failure(Exception): status_code = 503
        calls = []
        def call():
            calls.append(1)
            if len(calls) <= 5: raise Failure()
            return 'ok'
        with patch.object(loop.time, 'sleep') as sleep:
            self.assertEqual(loop.retry_call(call), 'ok')
            self.assertIn(((300,),), [(c.args,) for c in sleep.call_args_list])

    def test_killed_process_recovery_between_stages(self):
        # Actually terminate Python while it owns an experiment at each boundary.
        script = r'''
import sys,time,json
from pathlib import Path
import neokernel.transaction as m
from neokernel.loop import apply_proposal,generated_diff
from neokernel.storage import snapshot
root=Path(sys.argv[1]); stage=sys.argv[2]
def stop(name):
    if stage == name:
        (root/'results/ready').write_text(name)
        while True: time.sleep(.1)
original=m.git
original_save=m.Transaction.save
def save(self):
    original_save(self)
    stop('journal')
m.Transaction.save=save
original_put=m.put_row
def put(directory,row):
    original_put(directory,row)
    stop('revert_log')
m.put_row=put
def hooked(root,*args):
    value=original(root,*args)
    if args[0]=='commit': stop('commit')
    if args[:2]==('switch','main'):
        stop('switch'); stop('revert_switch')
    if args[:2]==('branch','-D'): stop('revert_drop')
    if args[0]=='merge': stop('merge')
    if args[0]=='tag' and len(args)>2 and args[1]!='--list': stop('tag')
    return value
m.git=hooked
tx=m.Transaction(root,root/'results'); tx.begin('test','agent'); stop('begin')
before=snapshot(root/'engine')
apply_proposal(root/'engine',{'engine/engine.py':before['engine.py'].decode().replace('yield [1]','yield [2]')}); stop('apply')
r=dict(eligible=True,geomean_tps=102,workloads=[dict(name='public-0',tps=102,passed=True,gates={'correctness':True})])
row=tx.row(r,not stage.startswith('revert_'),2,'pass','test',generated_diff(before,snapshot(root/'engine')))
if stage=='decided':
    tx.snapshot_candidate(); tx.state.update(stage='decided',row=row); tx.save(); stop('decided')
tx.finish(row)
'''
        for stage in ['journal', 'begin', 'apply', 'decided', 'commit', 'switch', 'merge', 'tag', 'revert_log', 'revert_switch', 'revert_drop']:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); repo(root)
                env = dict(os.environ, PYTHONPATH=str(Path(loop.__file__).resolve().parents[1]))
                child = subprocess.Popen([sys.executable, '-c', script, str(root), stage], env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 15
                    while not (root/'results/ready').exists() and child.poll() is None and time.monotonic() < deadline:
                        time.sleep(.05)
                    if not (root/'results/ready').exists():
                        child.kill(); out, err = child.communicate()
                        self.fail(f'child failed at {stage}: {out!r} {err!r}')
                    child.kill(); child.communicate(timeout=5)
                    tx = Transaction(root, root/'results'); tx.startup(resume=True)
                    self.assertEqual(git(root, 'branch', '--show-current'), 'main')
                    self.assertFalse(git(root, 'status', '--porcelain'))
                    rows = read_log(root/'results')
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]['kept'], stage in {'merge', 'tag'})
                    self.assertTrue((root/'results/snapshots/1/engine/engine.py').exists())
                    if stage not in {'begin', 'journal'}:
                        self.assertIn('yield [2]', (root/'results/snapshots/1/engine/engine.py').read_text())
                    self.assertEqual(tx.begin('next', 'agent'), 2)
                finally:
                    if child.poll() is None: child.kill(); child.communicate()


if __name__ == '__main__': unittest.main()
