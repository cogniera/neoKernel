"""Explicit remote source allowlist; never traverse local result directories."""

from pathlib import Path


def judge_source_files(root: Path) -> list[tuple[Path, str]]:
    root = root.resolve()
    paths = sorted((root / 'neokernel').glob('*.py')) + [root / 'agent' / 'package.py']
    selected = []
    for path in paths:
        if path.name.startswith('.') or not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Remote source must be a regular repository file: {path}')
        selected.append((path, '/root/' + path.relative_to(root).as_posix()))
    if not any(remote == '/root/agent/package.py' for _, remote in selected):
        raise ValueError('Remote judge requires agent/package.py')
    return selected


def mount_judge_source(image, root: Path):
    for local, remote in judge_source_files(root):
        image = image.add_local_file(local, remote)
    return image
