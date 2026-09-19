"""LLM-free numeric search, staged outside the live engine until a winner is known."""

import ast
import itertools
import random
import tempfile
from pathlib import Path

from agent.package import package
from .guard import check
from .schema import select_workloads
from .storage import ROOT, append_log, restore, snapshot


def tunables(source: str) -> dict[str, list[int | float]]:
    tree = ast.parse(source)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TUNABLES" for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError("kernels/__init__.py must define exactly one literal TUNABLES dict; use --wire-rmsnorm for the starter")
    values = ast.literal_eval(assignments[0].value)
    if not isinstance(values, dict) or not values:
        raise ValueError("TUNABLES must be a nonempty dict")
    result = {}
    for name, options in values.items():
        options = options if isinstance(options, list) else [options]
        if not isinstance(name, str) or not options or any(type(v) not in {int, float} for v in options):
            raise ValueError("TUNABLES values must be numeric choices")
        result[name] = options
    return result


def set_tunables(source: str, values: dict) -> str:
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TUNABLES" for t in n.targets)]
    if len(nodes) != 1:
        raise ValueError("exactly one TUNABLES assignment required")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[:node.lineno-1]) + "TUNABLES = " + repr(values) + "\n" + "".join(lines[node.end_lineno:])


def points(space: dict, limit: int, randomized: bool, seed: int):
    if limit < 1:
        raise ValueError("sweep limit must be positive")
    keys = sorted(space)
    count = 1
    for key in keys:
        count *= len(space[key])
    indices = random.Random(seed).sample(range(count), min(limit, count)) if randomized else range(min(limit, count))
    for index in indices:
        chosen = {}
        for key in reversed(keys):
            index, digit = divmod(index, len(space[key]))
            chosen[key] = space[key][digit]
        yield chosen


def wire_rmsnorm(root: Path):
    """Stage the guide's adapter and a tunable launch hook; called only on request."""
    engine = root / "engine.py"
    source = engine.read_text(encoding="utf-8")
    if "class FusedRMSNorm" in source or "base = self.model.model" in source:
        raise ValueError("automatic RMSNorm wiring is for the unchanged starter only")
    native = Path(__file__).with_name("native_engine.py").read_text(encoding="utf-8")
    if source != native:
        raise ValueError("--wire-rmsnorm requires the unchanged starter")
    adapter = '''from kernels.rmsnorm import rms_norm


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight
        self.variance_epsilon = reference.variance_epsilon

    def forward(self, x):
        return rms_norm(x, self.weight, self.variance_epsilon)


'''
    source = source.replace("class Engine:", adapter + "class Engine:")
    integration = '''        base = self.model.model
        base.norm = FusedRMSNorm(base.norm)
        for layer in base.layers:
            layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
            layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
            layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)

'''
    source = source.replace("    def generate(", integration + "    def generate(")
    engine.write_text(source, encoding="utf-8")
    init = root / "kernels" / "__init__.py"
    init.write_text(init.read_text(encoding="utf-8") + '\nTUNABLES = {"rmsnorm.num_warps": [4, 8, 16]}\n', encoding="utf-8")
    kernel = root / "kernels" / "rmsnorm.py"
    text = kernel.read_text(encoding="utf-8").replace("import torch", "from kernels import TUNABLES\nimport torch")
    text = text.replace("num_warps=max(4, min(16, block // 256))", 'num_warps=TUNABLES["rmsnorm.num_warps"]')
    kernel.write_text(text, encoding="utf-8")


def run_sweep(args, remote) -> int:
    live = ROOT / "engine"
    original = snapshot(live)
    workloads = select_workloads(args.workloads)
    best_state = original
    baseline = remote.bench(package(live), workloads, 2)
    if not baseline["eligible"]:
        append_log(baseline, proposer="sweep", item="baseline", note="baseline failed gates")
        return 1
    best_score = baseline["geomean_tps"]
    append_log(baseline, proposer="sweep", item="baseline", kept=True, note="measured current tree")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "engine"
        restore(root, original)
        if args.wire_rmsnorm:
            wire_rmsnorm(root)
        init = root / "kernels" / "__init__.py"
        source = init.read_text(encoding="utf-8")
        candidates = list(points(tunables(source), args.steps, args.random, args.seed))
        payloads, states = [], []
        for point in candidates:
            init.write_text(set_tunables(source, point), encoding="utf-8")
            check(root)
            payloads.append(package(root))
            states.append(snapshot(root))
        # One remote invocation reuses the reference model and container for all points.
        results = remote.many(payloads, workloads, 2, validate_rmsnorm=args.wire_rmsnorm)
        winner = None
        for i, result in enumerate(results):
            if result["eligible"] and result["geomean_tps"] > best_score:
                best_score, best_state, winner = result["geomean_tps"], states[i], i
        for i, result in enumerate(results):
            append_log(result, proposer="sweep", item="numeric_tunables", hypothesis=str(candidates[i]),
                       kept=i == winner, files_changed=["engine/kernels/__init__.py"],
                       note=f"point {candidates[i]}; {'best passing improvement' if i == winner else 'not kept'}")
    if snapshot(live) != original:
        raise ValueError("live engine changed during sweep; refusing to overwrite it")
    restore(live, best_state)
    print(f"Sweep: {len(candidates)} points; best={best_score:.2f} tok/s; working tree only")
    return 0
