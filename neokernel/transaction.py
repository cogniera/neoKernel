"""Durable, recoverable experiment transactions. Never pushes."""

import json
import shutil
import subprocess
import traceback
from pathlib import Path

from .storage import read_log, snapshot, restore, timestamp, write_json


def git(root, *args):
    result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'git {args[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def credential_scan(root):
    # Split markers so the scanner does not flag its own source.
    result = subprocess.run(['git', 'grep', '--cached', '-l', '-I', '-e', '578O' + 'LdPX',
                             '-e', 'dryft' + '_pat', '--', '.'], cwd=root, capture_output=True)
    if result.returncode != 1:
        raise RuntimeError('Credential scan failed; commit aborted (contents suppressed)')


def put_row(directory, row):
    """Atomic, idempotent log append, including after a killed finalization."""
    records = read_log(directory)
    existing = next((r for r in records if r['id'] == row['id']), None)
    if existing:
        if existing != row:
            raise RuntimeError('Conflicting transaction log id')
        return
    path = directory / 'log.jsonl'
    tmp = path.with_suffix('.jsonl.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        for record in [*records, row]:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)


def crash(directory):
    directory.mkdir(parents=True, exist_ok=True)
    text = traceback.format_exc()
    import os
    key = os.environ.get('BASETEN_API_KEY')
    if key:
        text = text.replace(key, '[REDACTED]')
    (directory / 'CRASH.txt').write_text(timestamp() + '\n' + text, encoding='utf-8')


class Transaction:
    def __init__(self, root, directory):
        self.root, self.directory = Path(root), Path(directory)
        self.path = self.directory / 'transaction.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else None

    def save(self):
        write_json(self.path, self.state)

    def startup(self, resume=False):
        branch = git(self.root, 'branch', '--show-current')
        leftovers = git(self.root, 'branch', '--list', 'exp/*').splitlines()
        if self.state or leftovers:
            if not resume:
                raise ValueError('Unfinished experiment: use auto --resume')
            self.recover()
        if git(self.root, 'branch', '--show-current') != 'main':
            raise ValueError('Loop requires branch main')
        if git(self.root, 'status', '--porcelain', '--untracked-files=all'):
            raise ValueError('Loop requires a clean working tree')

    def begin(self, item, proposer, hypothesis=''):
        self.startup()
        ident = max((r['id'] for r in read_log(self.directory)), default=0) + 1
        self.state = dict(id=ident, branch=f'exp/{ident}', parent=git(self.root, 'rev-parse', 'main'),
                          item=item, proposer=proposer, hypothesis=hypothesis, ts=timestamp(), stage='started')
        self.save()
        git(self.root, 'switch', '-c', self.state['branch'])
        restore(self.directory / 'snapshots' / str(ident) / 'before', snapshot(self.root / 'engine'))
        return ident

    def snapshot_candidate(self):
        dest = self.directory / 'snapshots' / str(self.state['id'])
        restore(dest / 'engine', snapshot(self.root / 'engine'))
        # Also preserve all tracked/untracked changes, not only the allowed source.
        (dest / 'working-tree.diff').write_text(git(self.root, 'diff', 'HEAD', '--'), encoding='utf-8')
        return dest

    def finish(self, row):
        self.snapshot_candidate()
        self.state.update(stage='decided', row=row)
        self.save()
        if row['kept']:
            assert row['guard'] == 'pass' and row['delta_pct'] > 1.0
            assert row['workloads'] and all(w['passed'] and all(w['gates'].values()) for w in row['workloads'])
            git(self.root, 'add', '--', 'engine')
            credential_scan(self.root)
            git(self.root, 'commit', '-m', f"experiment {row['id']}: {row['item']}")
            self.state['candidate_sha'] = git(self.root, 'rev-parse', 'HEAD')
            self.save()
            git(self.root, 'switch', 'main')
            if git(self.root, 'rev-parse', 'HEAD') != self.state['parent']:
                raise RuntimeError('main changed during experiment')
            credential_scan(self.root)
            # A fast-forward merge has no additional commit that could lose its journal identity.
            git(self.root, 'merge', '--ff-only', self.state['branch'])
            self.finalize_keep()
        else:
            put_row(self.directory, row)
            self.discard()
        self.clear()

    def finalize_keep(self):
        row = self.state['row']
        assert row['kept'] and row['delta_pct'] > 1.0
        assert all(w['passed'] and all(w['gates'].values()) for w in row['workloads'])
        row['sha'] = self.state['candidate_sha']
        tag = f"kept-{row['id']}"
        existing = git(self.root, 'tag', '--list', tag)
        if not existing:
            git(self.root, 'tag', tag, row['sha'])
        elif git(self.root, 'rev-parse', tag) != row['sha']:
            raise RuntimeError('Existing keep tag conflicts')
        put_row(self.directory, row)
        write_json(self.directory / 'best.json', dict(id=row['id'], sha=row['sha'],
                   geomean_tps=row['geomean_tps'], per_workload_tps={w['name']: w['tps'] for w in row['workloads']}, ts=row['ts']))
        backup = self.directory.parent / 'results_backup' / str(row['id'])
        shutil.copytree(self.directory, backup, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('*.lock', '*.tmp'))
        if git(self.root, 'branch', '--list', self.state['branch']):
            git(self.root, 'branch', '-d', self.state['branch'])

    def discard(self):
        # Preserve unknown edits outside the experiment; never reset the whole repository.
        outside = git(self.root, 'status', '--porcelain', '--', '.', ':(exclude)engine')
        if outside:
            raise RuntimeError('Changes outside engine prevent safe recovery')
        git(self.root, 'restore', '--source=' + self.state['parent'], '--staged', '--worktree', '--', 'engine')
        git(self.root, 'clean', '-fd', '--', 'engine')
        git(self.root, 'switch', 'main')
        if git(self.root, 'branch', '--list', self.state['branch']):
            git(self.root, 'branch', '-D', self.state['branch'])

    def recover(self):
        if not self.state:
            branches = [b.strip().lstrip('*').strip() for b in git(self.root, 'branch', '--list', 'exp/*').splitlines()]
            current = git(self.root, 'branch', '--show-current')
            if len(branches) != 1 or current != branches[0] or not branches[0][4:].isdigit():
                raise RuntimeError('Ambiguous orphan experiments; preserved for inspection')
            self.state = dict(id=int(current[4:]), branch=current, parent=git(self.root, 'rev-parse', 'main'),
                              item='interrupted', proposer='agent', hypothesis='', ts=timestamp())
            self.save()
        candidate = self.state.get('candidate_sha')
        if candidate and git(self.root, 'rev-parse', 'main') == candidate:
            # The merge already happened; complete metadata rather than undo a judged keep.
            self.finalize_keep()
            self.clear()
            return
        current = git(self.root, 'branch', '--show-current')
        existing_row = next((r for r in read_log(self.directory) if r['id'] == self.state['id']), None)
        if current == 'main' and existing_row and not existing_row['kept'] and not git(self.root, 'branch', '--list', self.state['branch']):
            self.clear()
            return
        if current == self.state['branch']:
            self.snapshot_candidate()
        elif current != 'main':
            raise RuntimeError('Unrelated branch during recovery; preserved')
        elif not (self.directory / 'snapshots' / str(self.state['id']) / 'engine').exists():
            self.snapshot_candidate()
        from .loop import generated_diff
        before_path = self.directory / 'snapshots' / str(self.state['id']) / 'before'
        before = snapshot(before_path) if before_path.exists() else {}
        row = self.state.get('row')
        if row is None or row.get('kept'):
            candidate_path = self.directory / 'snapshots' / str(self.state['id']) / 'engine'
            candidate_state = snapshot(candidate_path) if candidate_path.exists() else snapshot(self.root / 'engine')
            row = self.row(None, False, None, 'not_run', 'interrupted; recovered without merge',
                           generated_diff(before, candidate_state))
        put_row(self.directory, row)
        self.discard()
        self.clear()

    def row(self, result, kept, delta, guard, note, diff, files=None):
        result = result or {}
        return dict(id=self.state['id'], ts=timestamp(), sha=git(self.root, 'rev-parse', 'HEAD'),
                    parent_sha=self.state['parent'], proposer=self.state['proposer'], item=self.state['item'],
                    hypothesis=self.state['hypothesis'], files_changed=files or [], guard=guard,
                    workloads=[{k: v for k, v in w.items() if k != 'samples'} for w in result.get('workloads', [])],
                    geomean_tps=result.get('geomean_tps'), delta_pct=delta, kept=kept,
                    gpu_seconds=result.get('gpu_seconds', 0), note=note, diff=diff,
                    engine_sha256=result.get('engine_sha256'), implemented_items=[],
                    repairs=self.state.get('repairs', 0), first_error=self.state.get('first_error'))

    def clear(self):
        self.path.unlink(missing_ok=True)
        self.state = None
