"""Local single-writer lock, hourly status, and morning report; no remote work."""

import json
import os
import threading
from contextlib import contextmanager

from .accounting import SpendLedger, start_night
from .storage import ROOT, RESULTS, read_log, timestamp, write_json, git_sha


@contextmanager
def exclusive(directory=RESULTS):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'night.lock').open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another loop owns the night lock') from exc
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def status(directory=RESULTS):
    night = json.loads((directory / 'night_budget.json').read_text())
    rows = [r for r in read_log(directory) if r['id'] > night['start_log_id']]
    ledger = SpendLedger(directory).read()
    best = json.loads((directory / 'best.json').read_text()) if (directory / 'best.json').exists() else None
    return night, rows, ledger, best


def hourly(directory=RESULTS):
    night, rows, ledger, best = status(directory)
    with (directory / 'NIGHT.md').open('a', encoding='utf-8') as stream:
        stream.write(f"{timestamp()} | steps={sum(r['proposer'] == 'agent' for r in rows)} | "
                     f"kept={[r['id'] for r in rows if r['kept']]} | best={best['geomean_tps'] if best else None} | "
                     f"spend=${ledger['estimated_usd'] - night['start_estimated_usd']:.4f}\n")


def morning(directory=RESULTS, outcome='finished'):
    night, rows, ledger, best = status(directory)
    totals = {}
    for call in ledger['calls']:
        if call['ts'] >= night['ts']:
            totals[call['tier']] = totals.get(call['tier'], 0) + call['estimated_usd']
    read = lambda name: (directory / name).read_text(encoding='utf-8') if (directory / name).exists() else 'None'
    ref = f"kept-{best['id']}" if best and (directory.parent / 'results_backup' / str(best['id'])).exists() else 'main'
    text = (f'# Morning report\n\n{timestamp()} — {outcome}\n\n'
            f'Log lines added: {len(rows)}; auto steps: {sum(r["proposer"] == "agent" for r in rows)}.\n\n'
            f'Kept ids and geomean deltas: {[(r["id"], r["geomean_tps"], r["delta_pct"]) for r in rows if r["kept"]]}\n\n'
            f'best.json:\n```json\n{json.dumps(best, indent=2)}\n```\n\n'
            f'Night spend by tier (estimated/reserved USD): {json.dumps(totals)}\n\n'
            f'Total night estimated/reserved spend: ${ledger["estimated_usd"] - night["start_estimated_usd"]:.4f} / $8.00. '
            'Provider invoices may differ; unresolved reservations remain charged.\n\n'
            f'CRASH.txt:\n```text\n{read("CRASH.txt")}\n```\n\n'
            f'Power restore:\n{read("POWERCFG_RESTORE.md")}\n\n'
            'Commands for the operator (not executed; no push performed):\n```powershell\n'
            f'git switch --detach {ref}\n'
            'py -3.11 -m neokernel freeze --out neokernel/results/morning_freeze --workloads all\n'
            'git switch main\n'
            'git push origin main\n' + (f'git push origin {ref}\n' if ref.startswith('kept-') else '') + '```\n')
    (directory / 'MORNING.md').write_text(text, encoding='utf-8')


@contextmanager
def night_watch(directory=RESULTS):
    start_night(directory)
    if not (directory / 'best.json').exists():
        latest = next((r for r in reversed(read_log(directory)) if r.get('kept') and r.get('geomean_tps')), None)
        if latest:
            write_json(directory / 'best.json', dict(id=latest['id'], sha=git_sha(), geomean_tps=latest['geomean_tps'],
                       per_workload_tps={w['name']: w['tps'] for w in latest['workloads']}, ts=latest['ts']))
    stop = threading.Event()
    def worker():
        while not stop.wait(3600):
            hourly(directory)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        hourly(directory)
        morning(directory)
