"""LLM-free numeric search, staged outside the live engine until a winner is known."""

import ast
import random
import tempfile
from pathlib import Path

from .accounting import SpendLimit, record_stop
from .loop import measured_baseline, run_experiment, generated_diff
from .schema import select_workloads
from .storage import ROOT, RESULTS, restore, snapshot
from .transaction import Transaction, crash


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


def numeric_candidates(source, limit, randomized=False, seed=0):
    space = tunables(source)
    if any(len(v) > 1 for v in space.values()):
        return list(points(space, limit, randomized, seed))
    base = {k: v[0] for k, v in space.items()}
    # Coordinate sweep: preserve scalar source, vary one numeric knob at a time.
    candidates = []
    keys = sorted(base, key=lambda k: (not k.startswith('attention.'), not k.startswith('merge.'), k))
    for key in keys:
        value = base[key]
        options = [value // 2, value * 2] if key.endswith('BLOCK') else ([2, 4, 8] if key.endswith('num_warps') else [1, 2, 3])
        for choice in options:
            minimum = 4096 if key == 'norm.BLOCK' else 128 if key == 'qk.BLOCK' else 1
            if choice != value and choice >= minimum:
                candidates.append({**base, key: choice})
    if randomized:
        random.Random(seed).shuffle(candidates)
    return candidates[:limit]


def run_sweep(args, remote) -> int:
    transaction = Transaction(ROOT, RESULTS)
    transaction.startup(args.resume)
    live = ROOT / 'engine'
    workloads = select_workloads(args.workloads)
    try:
        baseline = measured_baseline(remote, live, workloads)
        original = snapshot(live)
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / 'engine'
            restore(staged, original)
            if args.wire_rmsnorm:
                wire_rmsnorm(staged)
            source = (staged / 'kernels' / '__init__.py').read_text(encoding='utf-8')
            candidates = numeric_candidates(source, args.steps, args.random, args.seed)
            for point in candidates:
                (staged / 'kernels' / '__init__.py').write_text(set_tunables(source, point), encoding='utf-8')
                state = snapshot(staged)
                current = snapshot(live)
                files = {'engine/' + name: data.decode('utf-8') for name, data in state.items() if current.get(name) != data}
                if not files:
                    continue
                transaction.begin('numeric_tunables', 'sweep', str(point))
                baseline = run_experiment(transaction, remote, baseline, workloads, files)
        print(f"Sweep finished: best={baseline['geomean_tps']:.2f} tok/s", flush=True)
        return 0
    except SpendLimit as exc:
        record_stop(exc, RESULTS)
        if transaction.state:
            before = snapshot(RESULTS / 'snapshots' / str(transaction.state['id']) / 'before')
            transaction.finish(transaction.row(None, False, None, 'not_run', str(exc), generated_diff(before, snapshot(live))))
        print(str(exc), flush=True)
        return 0
    except BaseException:
        crash(RESULTS)
        raise
