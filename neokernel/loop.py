"""Bounded proposal loop with judged, journaled Git transactions. Never pushes."""

import json
import hashlib
import os
import random
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

from agent.package import package
from .guard import GuardError, check
from .schema import PUBLIC, Proposal, PROPOSAL_SCHEMA, select_workloads
from .storage import ROOT, RESULTS, append_log, read_log, restore, snapshot, write_json
from .transaction import Transaction, crash
from .accounting import SpendLedger, SpendLimit, record_stop
from .judge import keep_decision

KEPT_ITEMS = {'static_kv_cache', 'bypass_wrapper', 'cuda_graph_decode', 'concat_qkv', 'concat_gate_up',
              'fused_rmsnorm', 'fused_qk_norm_rope_kv_write', 'fused_silu_mul', 'decode_attention_kernel'}
PLAYBOOK = ['residual_fuse_norm', 'attention_splitk', 'lm_head_argmax_tiled', 'prefill_packed_weights',
            'gemm_epilogue_residual', 'per_shape_config', 'speculative_prompt_lookup']
REPAIR_TURNS = 2
UNIT_TEST_FILES = ('tests/test_hand_rolled_kernels.py', 'tests/test_hand_rolled_handoff.py')
FIRST_ERROR_CHARS = 4000


def allowed_path(name: str) -> bool:
    if not isinstance(name, str) or any(part in {'', '.', '..'} for part in name.split('/')):
        return False
    if any(c in name for c in ['\\', ':', '\x00']) or name.startswith('/'):
        return False
    return name == 'engine/engine.py' or (name.startswith('engine/kernels/') and name.endswith('.py'))


def validate_files(files: dict[str, str]) -> set[str]:
    if not isinstance(files, dict) or not files:
        raise ValueError('files must be a nonempty mapping')
    for name, content in files.items():
        if not allowed_path(name) or not isinstance(content, str):
            raise ValueError(f'file outside write scope or invalid content: {name}')
        if name == 'engine/engine.py' and not content.strip():
            raise ValueError('engine/engine.py may never be deleted or emptied')
    if len(files) > 200 or sum(len(v.encode('utf-8')) for v in files.values()) > 2*1024*1024:
        raise ValueError('replacement files exceed source budget')
    return set(files)


def proposal_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def apply_proposal(engine_dir: Path, files: dict[str, str]) -> list[str]:
    validate_files(files)
    root = engine_dir.resolve()
    before = snapshot(engine_dir)
    targets = {}
    for name, content in files.items():
        relative = name.removeprefix('engine/')
        target = engine_dir / relative
        if not target.resolve().is_relative_to(root) or any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in [target, *target.parents] if p != root.parent):
            raise ValueError('replacement target escapes engine or traverses a link')
        targets[relative] = (target, content)
    try:
        for relative, (target, content) in targets.items():
            if content == '':
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding='utf-8', newline='\n')
        after = snapshot(engine_dir)
        changed = sorted('engine/'+name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
        if not changed:
            raise ValueError('replacement files had no effect')
        return changed
    except BaseException:
        restore(engine_dir, before)
        raise


def generated_diff(before: dict, after: dict) -> str:
    """Git compares snapshots so new, deleted, and previously uncommitted files appear."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        restore(root/'before'/'engine', before)
        restore(root/'after'/'engine', after)
        result = subprocess.run(['git', '-c', 'core.autocrlf=false', 'diff', '--no-index', '--no-ext-diff',
                                 '--src-prefix=a/', '--dst-prefix=b/', '--', 'before', 'after'],
                                cwd=root, capture_output=True, text=True, encoding='utf-8')
        if result.returncode not in (0, 1):
            raise RuntimeError('git diff failed: '+result.stderr.strip())
        return result.stdout.replace('a/before/', 'a/').replace('b/after/', 'b/').replace('a/after/', 'a/').replace('b/before/', 'b/')


def history_summary(records: list[dict]) -> dict:
    tried = Counter(r["item"] for r in records if r.get("proposer") == "agent")
    reverted = Counter(r["item"] for r in records if r.get("proposer") == "agent" and not r.get("kept"))
    kept = sorted(KEPT_ITEMS | {item for r in records if r.get('kept') for item in [r['item'], *r.get('implemented_items', [])]})
    dryft = {r['item']: {'id': r['id'], 'kept': r['kept'], 'dryft_tps': r['dryft_tps']} for r in records if r.get('dryft_tps')}
    return {"attempts": dict(tried), "reverts": dict(reverted), "kept": kept,
            "current_base": [item for item, v in dryft.items() if v['kept']], "dryft_verdicts": dryft,
            "coverage": [item for item in PLAYBOOK if not tried[item]]}


def context(engine_dir: Path, profile: dict, records: list[dict], items: list[str] | None = None) -> list[dict]:
    program = Path(__file__).with_name("program.md").read_text(encoding="utf-8")
    static = program + "\nGuard rules: only torch, triton, transformers, safetensors, math, os (read-only), typing, dataclasses, functools, itertools, collections, json, and local engine modules. No timers, CUDA events, exec/eval, importlib, subprocess, socket, writes, module mutation, or environment mutation. Literal disabling TF32 is the sole module-setting exception.\nReturn a proposal matching this schema:\n" + json.dumps(PROPOSAL_SCHEMA, sort_keys=True)
    files = {"engine/"+name: data.decode("utf-8") for name, data in snapshot(engine_dir).items() if allowed_path("engine/"+name)}
    diff = subprocess.run(["git", "diff", "--", "engine/"], cwd=ROOT, capture_output=True, text=True).stdout
    dynamic = {"selected_items": items or PLAYBOOK, "last_15_log_lines": records[-15:], "history": history_summary(records),
               "profile": {k: v for k, v in profile.items() if k != 'trace'},
               "current_files": files, "current_diff": diff}
    return [{"role": "system", "content": static}, {"role": "user", "content": json.dumps(dynamic)}]


def retry_call(call, allow_format_error=False):
    """Transient backoff; persistent API outages pause five minutes and retry."""
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if isinstance(exc, SpendLimit):
                raise
            if allow_format_error and status in {400, 422} and any(
                    term in str(exc).lower() for term in ['response_format', 'json_schema', 'structured output']):
                raise  # structured-output fallback / malformed request
            if status is None and not type(exc).__name__.startswith(('API', 'Connect', 'ReadTimeout')):
                raise
            if attempt < 4 and (status == 429 or (isinstance(status, int) and 500 <= status < 600)):
                time.sleep(min(30, 2**attempt) + random.random())
                attempt += 1
            else:
                print('Baseten unavailable; pausing five minutes before retry.', flush=True)
                time.sleep(300)
                attempt = 0


class Proposer:
    def __init__(self, model: str):
        from openai import OpenAI
        if not os.environ.get("BASETEN_API_KEY"):
            raise ValueError("BASETEN_API_KEY must be configured before auto")
        self.client = OpenAI(base_url="https://inference.baseten.co/v1", api_key=os.environ["BASETEN_API_KEY"],
                             max_retries=0, timeout=120)
        self.model = model
        self.spend = SpendLedger()
        available = retry_call(self.client.models.list)
        if model not in {m.id for m in available.data}:
            raise ValueError(f"model slug is not present in /v1/models: {model}")
        print(f'Confirmed GET /v1/models: {model}', flush=True)
        self.structured = True

    def completion(self, messages, response_format):
        # UTF-8 byte count is a conservative token upper bound, plus framing/schema.
        input_bound = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + len(json.dumps(response_format).encode()) + 4096
        max_tokens = 16384
        # Baseten GLM-5.2 pricing: $1.40/M input, $4.40/M output (2026-09-20).
        bound = (input_bound * 1.4 + max_tokens * 4.4) / 1e6
        self.spend.reserve_amount('Baseten', bound, 'GLM-5.2 proposal')
        try:
            result = self.client.chat.completions.create(model=self.model, messages=messages,
                                                        response_format=response_format, max_tokens=max_tokens)
        except Exception as exc:
            if getattr(exc, 'status_code', None) in {400, 401, 403, 404, 422, 429}:
                self.spend.settle(0, operation='rejected API request')
            # Unknown/server failures retain their reservation; they may have generated tokens.
            raise
        usage = result.usage
        amount = (usage.prompt_tokens * 1.4 + usage.completion_tokens * 4.4) / 1e6 if usage else bound
        self.spend.settle(amount, operation='GLM-5.2 proposal')
        return result

    def repair(self, conversation: list[dict]) -> Proposal:
        """Same schema and model; the conversation carries the model's own files and the test output."""
        return self.propose(conversation)

    def propose(self, messages: list[dict]) -> Proposal:
        last_error = None
        for attempt in range(2):
            response_format = ({"type": "json_schema", "json_schema": {"name": "proposal", "strict": True,
                                                                         "schema": PROPOSAL_SCHEMA}}
                               if self.structured else {"type": "json_object"})
            try:
                result = retry_call(lambda: self.completion(messages, response_format), allow_format_error=self.structured)
            except Exception as exc:
                if self.structured and getattr(exc, "status_code", None) in {400, 422} and any(
                        term in str(exc).lower() for term in ["response_format", "json_schema", "structured output"]):
                    self.structured = False
                    result = retry_call(lambda: self.completion(messages, {"type": "json_object"}))
                else:
                    raise
            try:
                return parse_proposal_text(result.choices[0].message.content)
            except (ValueError, TypeError, IndexError, AttributeError) as exc:
                last_error = str(exc)
                messages = messages + [{"role": "user", "content": f"Invalid proposal JSON: {last_error}. Return exactly the schema."}]
        raise ValueError(f"invalid proposal after one retry: {last_error}")


def parse_proposal_text(text: str) -> Proposal:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate proposal field: {key}")
            result[key] = value
        return result

    return Proposal.parse(json.loads(text, object_pairs_hook=unique))


def validate_proposal(proposal: Proposal, items: list[str], records: list[dict]) -> None:
    if proposal.item not in items:
        raise ValueError(f"item is not in the selected playbook; this run accepts only: {', '.join(items)}")
    summary = history_summary(records)
    if summary["reverts"].get(proposal.item, 0) >= 5:
        raise ValueError("move-on rule: this item already has five reverts")
    validate_files(proposal.files)
    digest = proposal_digest(proposal.files)
    if any(r.get("guard") == "fail" and r.get("proposal_sha256") == digest for r in records):
        raise ValueError("this patch already failed guard; propose a different patch")
    if len(re.findall(r"[.!?](?:\s|$)", proposal.reasoning)) > 3:
        raise ValueError("reasoning exceeds three sentences")


def unit_sources(engine_dir: Path) -> dict[str, str]:
    """Candidate engine plus the harness's own test files; a proposal never supplies tests."""
    sources = {'engine/' + name: data.decode('utf-8') for name, data in snapshot(engine_dir).items()
               if allowed_path('engine/' + name)}
    harness = Path(__file__).resolve().parents[1]
    for name in UNIT_TEST_FILES:
        sources[name] = (harness / name).read_text(encoding='utf-8')
    return sources


def summarize_unit_output(output: str, limit: int = 12000) -> tuple[str, str]:
    """First failure block, and a bounded excerpt of test headers, failure blocks and the final summary."""
    blocks = [b.strip('\n') for b in output.split('=' * 70)]
    failures = [b for b in blocks if b.startswith(('ERROR:', 'FAIL:'))]
    headers = []
    for line in output.splitlines():
        if re.match(r'^test_\w+ \(', line):
            head, _, status = line.partition(' ... ')
            headers.append(head[:160] + ' ... ' + (status if re.match(r'^(ok|ERROR|FAIL|skipped)\b', status) else ''))
        elif re.match(r'^(Ran \d|OK\b|FAILED|TIMEOUT)', line):
            headers.append(line[:200])
    first = failures[0] if failures else output[-FIRST_ERROR_CHARS:]
    excerpt = '\n'.join(headers) + '\n\n' + '\n\n'.join(failures) if failures else output[-limit:]
    if len(excerpt) > limit:
        excerpt = excerpt[:limit] + '\n[truncated]'
    return first[:FIRST_ERROR_CHARS], excerpt


def repair_request(turn: int, failure: str) -> str:
    return (f"Repair turn {turn} of {REPAIR_TURNS}. Your proposal passed the guard but failed the harness's fixed unit tests "
            f"on L4 ({', '.join(UNIT_TEST_FILES)}). They compare every kernel with the Transformers 4.51.3 module it "
            "replaces on random inputs and check the engine for prefill handoff, graph and buffer reuse, exact output "
            "counts and fresh-prompt reset on the pinned checkpoint; the interfaces they import must stay available.\n"
            f"Test output:\n{failure}\n"
            "Return corrected complete files in the same proposal schema with the same item and hypothesis. Include the "
            "full content of every file you change; omitted files stay as last applied. Fix the cause of the failure "
            "rather than removing what the tests exercise.")


def repair_stage(transaction, remote, engine_dir, proposal, proposer, messages):
    """L4 unit tests between guard and check, with up to REPAIR_TURNS corrected-file turns from the same model."""
    ident = transaction.state['id']
    conversation = [*messages, {'role': 'assistant', 'content': json.dumps(vars(proposal))}]
    summary = dict(passed=False, repairs=0, first_error=None, gpu_seconds=0.0)

    def test(turn):
        result = remote.unit_tests(unit_sources(engine_dir))
        write_json(RESULTS / 'unit_tests' / f'{ident}-{turn}.json', result)
        summary['gpu_seconds'] += result.get('gpu_seconds', 0)
        if result.get('skipped'):
            raise RuntimeError(f"unit tests skipped on L4; environment problem, not a candidate failure: {result['output'][-2000:]}")
        first, excerpt = summarize_unit_output(result['output'])
        print(f"Unit tests {ident} turn {turn}: {'passed' if result['passed'] else 'FAILED'}; "
              f"tests={result.get('tests')}; GPU seconds={result.get('gpu_seconds', 0):.2f}", flush=True)
        return result['passed'], first, excerpt

    passed, first, failure = test(0)
    if not passed:
        summary['first_error'] = first
        transaction.state.update(stage='unit_tests', first_error=first)
        transaction.save()
    while not passed and summary['repairs'] < REPAIR_TURNS and proposer is not None:
        turn = summary['repairs'] + 1
        summary['repairs'] = turn
        transaction.state.update(repairs=turn)
        transaction.save()
        conversation.append({'role': 'user', 'content': repair_request(turn, failure)})
        try:
            repaired = proposer.repair(conversation)
        except SpendLimit:
            raise
        except ValueError as exc:
            failure = f'Your repair was not valid proposal JSON: {exc}'
            print(f'Repair {ident}-{turn}: invalid proposal JSON', flush=True)
            continue
        write_json(RESULTS / 'proposals' / f'{ident}-repair{turn}.json', vars(repaired))
        conversation.append({'role': 'assistant', 'content': json.dumps(vars(repaired))})
        tested = snapshot(engine_dir)
        try:
            apply_proposal(engine_dir, repaired.files)
            check(engine_dir)
        except (GuardError, ValueError) as exc:
            restore(engine_dir, tested)
            failure = f'Guard rejected the repaired files, so they were not applied or tested: {exc}'
            print(f'Repair {ident}-{turn}: guard rejected; {exc}', flush=True)
            continue
        passed, _, failure = test(turn)
    summary['passed'] = passed
    return summary


def run_experiment(transaction, remote, baseline, workloads, files, hypothesis='', proposal=None,
                   proposer=None, messages=None):
    engine_dir = transaction.root / 'engine'
    before = snapshot(engine_dir)
    changed, result, guard_status, note, unit = [], None, 'not_run', '', None
    try:
        changed = apply_proposal(engine_dir, files)
        check(engine_dir)
        guard_status = 'pass'
    except (GuardError, ValueError) as exc:
        guard_status, note = 'fail', str(exc)
    if guard_status == 'pass' and proposal is not None:
        # Only a candidate that passes the unit tests reaches check and the judge.
        unit = repair_stage(transaction, remote, engine_dir, proposal, proposer, messages or [])
        if not unit['passed']:
            note = f"failed after {unit['repairs']} repairs" if unit['repairs'] else 'unit tests failed; no repair proposer'
    if guard_status == 'pass' and (unit is None or unit['passed']):
        checked = remote.bench(package(engine_dir), PUBLIC, 1, correctness_only=True)
        keep_decision(checked, baseline)  # Surface infrastructure failure codes even on L4.
        if not checked['eligible']:
            result, note = checked, 'correctness check failed'
        else:
            result = remote.bench(package(engine_dir), workloads, 3)
            result['gpu_seconds'] += checked.get('gpu_seconds', 0)
    kept, delta = keep_decision(result, baseline) if result else (False, None)
    note = note or ('judge kept improvement above 1 percent' if kept else 'gates failed or improvement did not exceed 1 percent')
    after = snapshot(engine_dir)
    changed = sorted('engine/' + name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    diff = generated_diff(before, after)
    if unit:
        result = result or {}
        result['gpu_seconds'] = result.get('gpu_seconds', 0) + unit['gpu_seconds']
    row = transaction.row(result, kept, delta, guard_status, note, diff, changed)
    if proposal:
        row['proposal_sha256'] = proposal_digest(proposal.files)
    transaction.finish(row)
    print(f"Experiment {row['id']}: {row['item']}: {'kept' if kept else 'reverted'}; delta={delta}; {note}", flush=True)
    return result if kept else baseline


def measured_baseline(remote, engine_dir, workloads):
    # A failing current-engine sample is a candidate failure, not a harness crash.
    while True:
        baseline = remote.bench(package(engine_dir), workloads, 3)
        keep_decision(baseline, baseline)
        if baseline['eligible']:
            return baseline
        print('Current-engine baseline failed gates; preserving main and remeasuring.', flush=True)


def run_loop(args, remote, proposer=None) -> int:
    transaction = Transaction(ROOT, RESULTS)
    transaction.startup(getattr(args, 'resume', False))
    items = args.items.split(',') if args.items else PLAYBOOK
    if not items or any(i not in PLAYBOOK for i in items):
        raise ValueError('unknown playbook item')
    workloads = select_workloads(args.workloads)
    engine_dir = ROOT / 'engine'
    try:
        proposer = proposer or Proposer(args.model)
        baseline = measured_baseline(remote, engine_dir, workloads)
        night_path = RESULTS / 'night_budget.json'
        start_id = json.loads(night_path.read_text())['start_log_id'] if night_path.exists() else 0
        done = sum(r.get('proposer') == 'agent' and r['id'] > start_id for r in read_log(RESULTS)) if getattr(args, 'resume', False) else 0
        for step in range(done, args.steps):
            records = read_log(RESULTS)
            # Use available profile data; new GPU work goes through judge calls only.
            profile_path = RESULTS / 'profile.json'
            profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
            messages = context(engine_dir, profile, records, items)
            transaction.begin('proposal_rejected', 'agent')
            proposal = None
            for retry in range(2):
                try:
                    proposed = proposer.propose(messages)
                    validate_proposal(proposed, items, records)
                    proposal = proposed
                    break
                except SpendLimit:
                    raise
                except ValueError as exc:
                    messages.append({'role': 'user', 'content': str(exc) + '; propose a valid alternative.'})
            if proposal is None:
                row = transaction.row(None, False, None, 'not_run', messages[-1]['content'], '')
                transaction.finish(row)
                continue
            transaction.state.update(item=proposal.item, hypothesis=proposal.hypothesis)
            transaction.save()
            write_json(RESULTS / 'proposals' / f"{transaction.state['id']}.json", vars(proposal))
            baseline = run_experiment(transaction, remote, baseline, workloads, proposal.files, proposal=proposal,
                                      proposer=proposer, messages=messages)
        return 0
    except SpendLimit as exc:
        record_stop(exc, RESULTS)
        if transaction.state:
            before = snapshot(RESULTS / 'snapshots' / str(transaction.state['id']) / 'before')
            row = transaction.row(None, False, None, 'not_run', str(exc), generated_diff(before, snapshot(engine_dir)))
            transaction.finish(row)
        print(str(exc), flush=True)
        return 0
    except BaseException:
        crash(RESULTS)
        raise
