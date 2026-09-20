"""No Modal client or network required to verify the upload boundary."""

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from neokernel.mounts import judge_source_files, mount_judge_source


class MountTests(unittest.TestCase):
    def test_smoke_check_uses_capped_l4_and_no_idle_charge(self):
        from types import SimpleNamespace
        from neokernel.cli import Remote
        from neokernel.schema import PUBLIC
        from neokernel.accounting import estimate_usd
        remote = Remote(smoke_check=True)
        remote.api = SimpleNamespace(check_smoke_remote=Mock(), check_remote=Mock(), judge_remote=Mock())
        remote.api.check_smoke_remote.remote.return_value = {
            'native': {}, 'gpu_seconds': 20, 'eligible': True, 'workloads': [], 'geomean_tps': 1}
        remote.spend = Mock()
        with patch('neokernel.cli.save_run'):
            remote.bench(b'source', [PUBLIC[0]], 1, correctness_only=True)
        remote.api.check_smoke_remote.remote.assert_called_once()
        remote.api.check_remote.remote.assert_not_called()
        remote.api.judge_remote.remote.assert_not_called()
        self.assertLess(estimate_usd('L4', 120), .05)
        remote.spend.reserve_amount.assert_called_once_with('L4', estimate_usd('L4', 120), 'bounded public-0 check')
        self.assertEqual(remote.spend.record.call_args.kwargs, {'idle_s': 0, 'timeout_s': 60})

    def test_only_direct_python_modules_and_package_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = {'neokernel/__init__.py', 'neokernel/modal_app.py', 'neokernel/judge.py', 'agent/package.py'}
            forbidden = {'neokernel/results/night.lock', 'neokernel/results/transaction.json',
                         'neokernel/results/nested.py', 'neokernel/results_backup/1/judge.py',
                         'neokernel/__pycache__/judge.py', 'neokernel/.virtual_documents/a.py',
                         'neokernel/.git/config.py', 'neokernel/docs/example.py', 'neokernel/program.md',
                         'docs/index.html', '.git/config', '.virtual_documents/example.py',
                         'agent/client.py', 'engine/engine.py'}
            for name in allowed | forbidden:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            expected = {(root / name, '/root/' + name) for name in allowed}
            self.assertEqual(set(judge_source_files(root)), expected)
            image = Mock()
            image.add_local_file.return_value = image
            self.assertIs(mount_judge_source(image, root), image)
            self.assertEqual({call.args for call in image.add_local_file.call_args_list}, expected)
            image.add_local_dir.assert_not_called()
            image.add_local_python_source.assert_not_called()

    def test_automatic_source_mount_disabled_for_every_function(self):
        path = Path(__file__).resolve().parents[1] / 'neokernel/modal_app.py'
        tree = ast.parse(path.read_text())
        options = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'dict':
                base = options.get(node.value.args[0].id, {}) if node.value.args else {}
                options[node.targets[0].id] = {**base, **{kw.arg: kw.value for kw in node.value.keywords}}
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'function':
                        kwargs = {}
                        for kw in decorator.keywords:
                            if kw.arg is None:
                                kwargs.update(options[kw.value.id])
                            else:
                                kwargs[kw.arg] = kw.value
                        self.assertIs(ast.literal_eval(kwargs['include_source']), False, node.name)


if __name__ == '__main__':
    unittest.main()
