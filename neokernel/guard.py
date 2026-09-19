"""Conservative source lint and bounded archive extraction, not a security sandbox."""

import ast
import io
import tarfile
from pathlib import Path, PurePosixPath

ALLOWED = {"torch", "triton", "transformers", "safetensors", "math", "os", "typing",
           "dataclasses", "functools", "itertools", "collections", "json"}
PROTECTED = {"torch", "triton", "transformers"}
MAX_BYTES = 2 * 1024 * 1024
TF32 = {"torch.backends.cuda.matmul.allow_tf32", "torch.backends.cudnn.allow_tf32"}


class GuardError(ValueError):
    pass


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    if isinstance(node, ast.Subscript):
        return dotted(node.value)
    return ""


def lint_source(source: str, filename: str, local: set[str]) -> None:
    try:
        tree = ast.parse(source, filename)
    except SyntaxError as e:
        raise GuardError(f"{filename}:{e.lineno}: {e.msg}") from e
    aliases: dict[str, str] = {}

    def canonical(node):
        value = dotted(node)
        head, _, rest = value.partition(".")
        return aliases.get(head, head) + ("." + rest if rest else "")

    def fail(node, message):
        raise GuardError(f"{filename}:{getattr(node, 'lineno', 1)}: {message}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for module in modules:
                root = module.split(".")[0]
                if root not in ALLOWED | local and not (isinstance(node, ast.ImportFrom) and node.level):
                    fail(node, f"import {module!r} is not allowed")
            for a in node.names:
                if a.name == "*":
                    fail(node, "wildcard imports hide symbol provenance")
                aliases[a.asname or a.name.split(".")[0]] = (
                    f"{node.module}.{a.name}" if isinstance(node, ast.ImportFrom) else
                    a.name if a.asname else a.name.split(".")[0])
    # Track direct aliases, including aliases of protected module attributes.
    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = canonical(node.value)
                if value:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            aliases[target.id] = value
    for node in ast.walk(tree):
        name = canonical(node)
        if isinstance(node, ast.Name) and node.id in {"exec", "eval", "__import__", "builtins", "importlib"}:
            fail(node, f"forbidden symbol {node.id}")
        if isinstance(node, ast.Attribute) and (name.startswith(("time.", "sys.modules", "torch.cuda.Event"))
                                                or node.attr in {"__globals__", "__builtins__", "__subclasses__"}):
            fail(node, f"forbidden attribute {name}")
        if isinstance(node, ast.Name) and "monkeypatch" in node.id.lower():
            fail(node, "monkeypatch is forbidden")
        if isinstance(node, ast.Call):
            call = canonical(node.func)
            if call.startswith(("time.", "torch.cuda.Event", "sys.modules")):
                fail(node, f"forbidden call {call}")
            if call in {"exec", "eval", "__import__", "setattr", "delattr"}:
                fail(node, f"{call} is forbidden")
            if call.startswith("os.") and call not in {"os.fspath", "os.getenv", "os.cpu_count"} and not call.startswith("os.path."):
                fail(node, f"unsafe OS call {call}")
            if call == "open":
                mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), ast.Constant("r"))
                if not isinstance(mode, ast.Constant) or not isinstance(mode.value, str) or any(c in mode.value for c in "wax+"):
                    fail(node, "open must have a literal read-only mode")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for leaf in ast.walk(target):
                    target_name = canonical(leaf)
                    if isinstance(leaf, (ast.Attribute, ast.Subscript)):
                        if target_name.startswith("os.environ"):
                            fail(node, "environment mutation is forbidden")
                        if target_name.split(".")[0] in PROTECTED:
                            # Required by the unchanged starter; only literal False is accepted.
                            if canonical(target) in TF32 and isinstance(getattr(node, "value", None), ast.Constant) and node.value.value is False:
                                continue
                            fail(node, "protected module mutation")
    if filename == "engine.py":
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Engine"]
        if len(classes) != 1:
            raise GuardError("engine.py:1: exactly one top-level Engine class is required")
        methods = {n.name: n for n in classes[0].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for method, expected in {"__init__": ["self", "model_path"], "generate": ["self", "input_ids", "max_new_tokens"]}.items():
            n = methods.get(method)
            if n is None or isinstance(n, ast.AsyncFunctionDef) or [a.arg for a in n.args.args] != expected or n.args.posonlyargs or n.args.kwonlyargs or n.args.vararg or n.args.kwarg or n.args.defaults:
                raise GuardError(f"engine.py:1: invalid {method} signature")


def check(engine_dir: Path) -> list[Path]:
    engine_dir = Path(engine_dir)
    files = []
    for p in sorted(engine_dir.rglob("*")):
        if "__pycache__" in p.parts:
            continue
        if p.is_symlink():
            raise GuardError(f"{p}: symbolic links are forbidden")
        if p.is_file():
            if any(part.startswith(".") for part in p.relative_to(engine_dir).parts):
                raise GuardError(f"{p}: hidden source is skipped by the starter packager")
            if p.suffix not in {".py", ".md"}:
                raise GuardError(f"{p}: only .py and .md source is allowed")
            files.append(p)
    if len(files) > 200 or sum(p.stat().st_size for p in files) >= MAX_BYTES:
        raise GuardError("source exceeds 200 files or 2 MiB")
    if engine_dir / "engine.py" not in files:
        raise GuardError("engine.py is missing")
    local = {p.relative_to(engine_dir).parts[0].removesuffix(".py") for p in files}
    if local & ALLOWED:
        raise GuardError("local modules may not shadow allowed dependencies")
    for p in files:
        if p.suffix == ".py":
            lint_source(p.read_text(encoding="utf-8"), p.relative_to(engine_dir).as_posix(), local)
    return files


def extract(payload: bytes, destination: Path) -> Path:
    if len(payload) >= MAX_BYTES:
        raise GuardError("archive exceeds 2 MiB")
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            total += member.size
            if (not member.isfile() or name.is_absolute() or ".." in name.parts or
                    "\\" in member.name or ":" in member.name or name.suffix not in {".py", ".md"} or
                    member.name in seen or len(seen) >= 200 or total >= MAX_BYTES):
                raise GuardError(f"unsafe archive entry: {member.name}")
            seen.add(member.name)
            path = destination.joinpath(*name.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src:
                path.write_bytes(src.read())
    check(destination)
    return destination


def physics_floors(weight_bytes: int, params: int, batch: int, S: int, N: int,
                   speculative: bool = False) -> dict:
    step = weight_bytes / 3.35e12
    decode = N * step / 8 if speculative else (N - 1) * step
    prefill = 2 * params * batch * S / 6e14
    return {"step_floor_s": step, "decode_floor_s": decode, "prefill_floor_s": prefill,
            "floor_tps": batch * N / (prefill + decode) if prefill + decode else 0,
            "speculative_relaxed": speculative}
