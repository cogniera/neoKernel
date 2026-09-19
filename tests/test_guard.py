import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from neokernel.guard import GuardError, check, extract, lint_source, physics_floors

STARTER = 'class Engine:\n def __init__(self, model_path): pass\n def generate(self, input_ids, max_new_tokens): yield [1]\n'


class GuardTests(unittest.TestCase):
    def test_starter(self):
        check(Path(__file__).resolve().parents[1] / "engine")

    def test_forbidden_patterns(self):
        patterns = ["import time", "import socket", "import subprocess", "import importlib",
                    "import builtins", "import sys", "time.time()", "torch.cuda.Event()",
                    "sys.modules['x'] = 1", "exec('x')", "eval('x')", "__import__('x')",
                    "open('x', 'w')", "open('x', mode='a')", "open('x', mode=mode)",
                    "setattr(torch, 'x', 1)", "monkeypatch = 1", "os.environ['X'] = '1'",
                    "os.putenv('X', '1')", "os.environ.update({'X': '1'})", "os.system('x')",
                    "torch.foo = 1", "triton.foo = 1", "transformers.foo = 1",
                    "torch.backends.cuda.matmul.allow_tf32 = True",
                    "import torch as t\nt.foo = 1", "from torch.cuda import Event\nEvent()",
                    "import torch\nt = torch\nt.foo = 1", "from torch import *"]
        for code in patterns:
            with self.subTest(code=code), self.assertRaises(GuardError):
                lint_source(code, "kernel.py", set())

    def test_model_eval_and_read_allowed(self):
        lint_source("model.eval()\nopen('weights', 'rb')\n", "kernel.py", set())

    def test_signatures(self):
        for source in ["", STARTER.replace("model_path", "path"), STARTER.replace("max_new_tokens", "n=1")]:
            with self.subTest(source=source), self.assertRaises(GuardError):
                lint_source(source, "engine.py", set())

    def test_files_limits_and_shadowing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "engine.py").write_text(STARTER)
            for name in ["weights.bin", "x.so", "x.cubin", "x.ptx", "x.cu", "torch.py"]:
                p = root / name
                p.write_text("")
                with self.subTest(name=name), self.assertRaises(GuardError):
                    check(root)
                p.unlink()
            (root / "big.md").write_bytes(b"x" * 2 * 1024 * 1024)
            with self.assertRaises(GuardError):
                check(root)
            (root / "big.md").unlink()
            for i in range(200):
                (root / f"x{i}.py").write_text("")
            with self.assertRaises(GuardError):
                check(root)

    def test_archive_traversal_and_links(self):
        for name, kind in [("../engine.py", tarfile.REGTYPE), ("/engine.py", tarfile.REGTYPE),
                           ("engine.py", tarfile.SYMTYPE), ("C:/engine.py", tarfile.REGTYPE)]:
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w:gz") as tar:
                item = tarfile.TarInfo(name)
                item.type = kind
                tar.addfile(item, io.BytesIO(b""))
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(GuardError):
                extract(data.getvalue(), Path(tmp))

    def test_floor(self):
        f = physics_floors(3350000000, 2000000000, 1, 512, 32)
        self.assertAlmostEqual(f["step_floor_s"], .001)
        self.assertAlmostEqual(f["decode_floor_s"], .031)
        self.assertAlmostEqual(physics_floors(3350000000, 2, 1, 1, 32, True)["decode_floor_s"], .004)


if __name__ == "__main__":
    unittest.main()
