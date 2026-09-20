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


class FakeRemote:
    def __init__(self, score=102, fail=False, crash=False):
        self.calls, self.score, self.fail, self.crash = 0, score, fail, crash

    def bench(self, *a, **kw):
        self.calls += 1
        if self.crash and self.calls > 1:
            raise RuntimeError('harness test crash')
        return result(100 if self.calls == 1 else self.score, not (self.fail and self.calls > 1))


class FakeProposer:
    def __init__(self, files=None):
        self.files = files or FILES
    def propose(self, messages):
        return Proposal('lm_head_argmax', 'test hypothesis', '2 percent', self.files, 'latency', 'test reasoning')


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

    def test_sweep_threshold_and_numeric_coordinates(self):
        from neokernel import sweep, storage
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
                with patch.object(sweep, 'ROOT', root), patch.object(storage, 'RESULTS', root/'results'):
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
