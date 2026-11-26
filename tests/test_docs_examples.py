"""Every Python example in the course docs must carry type annotations."""

from __future__ import annotations

import ast
import re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"
_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
_IGNORED_ARGUMENTS = {"self", "cls"}


def _blocks(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    blocks: list[tuple[int, str]] = []
    for match in _PYTHON_BLOCK.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        blocks.append((line, match.group(1)))
    return blocks


def _missing_annotations(source: str) -> list[str]:
    tree = ast.parse(source)
    missing: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for argument in arguments:
            if argument.arg in _IGNORED_ARGUMENTS:
                continue
            if argument.annotation is None:
                missing.append(f"{node.name}({argument.arg})")
        for extra in (node.args.vararg, node.args.kwarg):
            if extra is not None and extra.annotation is None:
                missing.append(f"{node.name}(*{extra.arg})")
        if node.returns is None:
            missing.append(f"{node.name}() -> ?")
    return missing


def test_python_examples_are_annotated() -> None:
    problems: list[str] = []
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        for line, source in _blocks(path):
            try:
                missing = _missing_annotations(source)
            except SyntaxError as error:
                problems.append(f"{path}:{line}: invalid Python example: {error}")
                continue
            for name in missing:
                problems.append(f"{path}:{line}: missing annotation in {name}")
    assert not problems, "\n".join(problems)
