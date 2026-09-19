"""Bounded proposal loop with file snapshots. Never commits, merges, or pushes."""

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

PLAYBOOK = ["static_kv_cache", "bypass_wrapper", "cuda_graph_decode", "concat_qkv", "concat_gate_up",
            "prefill_cuda_graph", "fused_rmsnorm", "fused_qk_norm_rope_kv_write", "fused_silu_mul",
            "lm_head_argmax", "decode_attention_kernel", "speculative_prompt_lookup"]


def allowed_path(name: str) -> bool:
    return ("\\" not in name and ":" not in name and ".." not in name.split("/") and
            (name == "engine/engine.py" or (name.startswith("engine/kernels/") and name.endswith(".py"))))


def patch_paths(patch: str) -> set[str]:
    """Refuse metadata operations and ambiguous paths before invoking git apply."""
    found = set()
    if len(patch.encode()) > 2*1024*1024:
        raise ValueError("patch exceeds source budget")
    for line in patch.splitlines():
        if line.startswith(("rename ", "copy ", "GIT binary patch", "Binary files", "old mode", "new mode", "new file mode 120", "index ")):
            if line.startswith("index ") and not line.endswith(" 120000"):
                continue
            raise ValueError("patch may contain only regular-file unified text edits")
        if line.startswith(("--- ", "+++ ")):
            name = line[4:]
            if name == "/dev/null":
                continue
            if name.startswith(("a/", "b/")):
                name = name[2:]
            if not allowed_path(name):
                raise ValueError(f"patch outside write scope: {name}")
            found.add(name)
    if not found:
        raise ValueError("patch contains no editable file hunks")
    return found


def apply_proposal(engine_dir: Path, patch: str) -> list[str]:
    """Validate in a temporary tree, then copy scoped changes without touching Git's index."""
    requested = patch_paths(patch)
    before = snapshot(engine_dir)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        staged = root / "engine"
        restore(staged, before)
        patch_file = root / "proposal.diff"
        patch_file.write_text(patch, encoding="utf-8", newline="\n")
        environment = dict(os.environ, GIT_CEILING_DIRECTORIES=str(root.parent))
        for extra in (["--check"], []):
            run = subprocess.run(["git", "apply", *extra, "--", str(patch_file)], cwd=root,
                                 env=environment, capture_output=True, text=True)
            if run.returncode:
                raise ValueError("patch_failed: " + run.stderr.strip())
        after = snapshot(staged)
        changed = {"engine/"+name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
        if not changed or not changed <= requested or any(not allowed_path(name) for name in changed):
            raise ValueError("patch changed unexpected paths or had no effect")
        restore(engine_dir, after)
    return sorted(changed)


def history_summary(records: list[dict]) -> dict:
    tried = Counter(r["item"] for r in records if r.get("proposer") == "agent")
    reverted = Counter(r["item"] for r in records if r.get("proposer") == "agent" and not r.get("kept"))
    kept = sorted({r["item"] for r in records if r.get("kept")})
    return {"attempts": dict(tried), "reverts": dict(reverted), "kept": kept,
            "coverage": [item for item in PLAYBOOK if not tried[item]]}


def context(engine_dir: Path, profile: dict, records: list[dict]) -> list[dict]:
    program = Path(__file__).with_name("program.md").read_text(encoding="utf-8")
    static = program + "\nGuard rules: only torch, triton, transformers, safetensors, math, os (read-only), typing, dataclasses, functools, itertools, collections, json, and local engine modules. No timers, CUDA events, exec/eval, importlib, subprocess, socket, writes, module mutation, or environment mutation. Literal disabling TF32 is the sole module-setting exception.\nReturn a proposal matching this schema:\n" + json.dumps(PROPOSAL_SCHEMA, sort_keys=True)
    files = {name: data.decode("utf-8") for name, data in snapshot(engine_dir).items() if name.endswith(".py")}
    diff = subprocess.run(["git", "diff", "--", "engine/"], cwd=ROOT, capture_output=True, text=True).stdout
    dynamic = {"last_15_log_lines": records[-15:], "history": history_summary(records),
               "profile": profile, "current_files": files, "current_diff": diff}
    return [{"role": "system", "content": static}, {"role": "user", "content": json.dumps(dynamic)}]


def retry_call(call):
    """Retry rate limits and server failures at most five times, with jitter."""
    for attempt in range(5):
        try:
            return call()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if (status != 429 and not (isinstance(status, int) and 500 <= status < 600)) or attempt == 4:
                raise
            time.sleep(min(30, 2**attempt) + random.random())


class Proposer:
    def __init__(self, model: str):
        from openai import OpenAI
        if not os.environ.get("BASETEN_API_KEY"):
            raise ValueError("BASETEN_API_KEY must be configured before auto")
        self.client = OpenAI(base_url="https://inference.baseten.co/v1", api_key=os.environ["BASETEN_API_KEY"],
                             max_retries=0, timeout=120)
        self.model = model
        available = retry_call(self.client.models.list)
        if model not in {m.id for m in available.data}:
            raise ValueError(f"model slug is not present in /v1/models: {model}")
        print(f'Confirmed GET /v1/models: {model}', flush=True)
        self.structured = True

    def propose(self, messages: list[dict]) -> Proposal:
        last_error = None
        for attempt in range(2):
            response_format = ({"type": "json_schema", "json_schema": {"name": "proposal", "strict": True,
                                                                         "schema": PROPOSAL_SCHEMA}}
                               if self.structured else {"type": "json_object"})
            try:
                result = retry_call(lambda: self.client.chat.completions.create(model=self.model, messages=messages,
                                                                                response_format=response_format))
            except Exception as exc:
                if self.structured and getattr(exc, "status_code", None) in {400, 422} and any(
                        term in str(exc).lower() for term in ["response_format", "json_schema", "structured output"]):
                    self.structured = False
                    result = retry_call(lambda: self.client.chat.completions.create(model=self.model, messages=messages,
                                                                                    response_format={"type": "json_object"}))
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
        raise ValueError("item is not in the selected playbook")
    summary = history_summary(records)
    if summary["reverts"].get(proposal.item, 0) >= 5:
        raise ValueError("move-on rule: this item already has five reverts")
    if proposal.item == "speculative_prompt_lookup" and not set(PLAYBOOK[:-1]) <= set(summary["kept"]):
        raise ValueError("speculative lookup requires all earlier items to be kept")
    patch_paths(proposal.patch)
    digest = hashlib.sha256(proposal.patch.encode()).hexdigest()
    if any(r.get("guard") == "fail" and r.get("patch_sha256") == digest for r in records):
        raise ValueError("this patch already failed guard; propose a different patch")
    if len(re.findall(r"[.!?](?:\s|$)", proposal.reasoning)) > 3:
        raise ValueError("reasoning exceeds three sentences")


def run_loop(args, remote, proposer=None) -> int:
    engine_dir = ROOT / "engine"
    items = args.items.split(",") if args.items else PLAYBOOK
    if not items or any(i not in PLAYBOOK for i in items):
        raise ValueError("unknown playbook item")
    workloads = select_workloads(args.workloads)
    remote.reserve(3600)
    proposer = proposer or Proposer(args.model)
    baseline = remote.bench(package(engine_dir), workloads, 3)
    if not baseline["eligible"]:
        append_log(baseline, proposer="agent", item="baseline", note="baseline failed gates")
        return 1
    best_score = baseline["geomean_tps"]
    append_log(baseline, proposer="agent", item="baseline", kept=True, note="measured current tree before proposals")
    for step in range(args.steps):
        records = read_log()
        profile = remote.profile(package(engine_dir), workloads[0])
        messages = context(engine_dir, profile, records)
        proposal = None
        for retry in range(2):
            try:
                proposed = proposer.propose(messages)
                validate_proposal(proposed, items, records)
                proposal = proposed
                break
            except ValueError as exc:
                messages.append({"role": "user", "content": str(exc) + "; propose a valid alternative."})
        if proposal is None:
            append_log(None, proposer="agent", item="proposal_rejected", note=messages[-1]["content"])
            continue
        before = snapshot(engine_dir)
        iteration_id = max((r["id"] for r in records), default=0) + 1
        for name, data in before.items():
            path = RESULTS / "snapshots" / str(iteration_id) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        write_json(RESULTS / "proposals" / f"{iteration_id}.json", vars(proposal))
        kept, result, files, guard_status, note = False, None, [], "not_run", ""
        try:
            files = apply_proposal(engine_dir, proposal.patch)
            check(engine_dir)
            guard_status = "pass"
            checked = remote.bench(package(engine_dir), PUBLIC, 1, correctness_only=True)
            if not checked["eligible"]:
                result, note = checked, "correctness check failed"
            else:
                result = remote.bench(package(engine_dir), workloads, 3)
                result["gpu_seconds"] += checked.get("gpu_seconds", 0) + profile.get("gpu_seconds", 0)
                kept = result["eligible"] and result["geomean_tps"] > best_score * 1.01
                if kept:
                    best_score = result["geomean_tps"]
                    note = "kept in working tree; no commit"
                else:
                    note = "gates failed or improvement did not exceed 1 percent"
        except GuardError as exc:
            guard_status, note = "fail", str(exc)
        except (ValueError, RuntimeError) as exc:
            note = str(exc)
        except KeyboardInterrupt:
            note = "interrupted; snapshot restored"
            raise
        finally:
            if not kept:
                restore(engine_dir, before)
            row = append_log(result, proposer="agent", item=proposal.item, hypothesis=proposal.hypothesis,
                             kept=kept, note=note, files_changed=files, guard=guard_status,
                             patch_sha256=hashlib.sha256(proposal.patch.encode()).hexdigest())
            print(f"Experiment {row['id']}: {proposal.item}: {'kept' if kept else 'reverted'}; {note}")
        attended = args.attended if args.attended is not None else step < 5
        if attended:
            input("Press Enter to continue: ")
    return 0
