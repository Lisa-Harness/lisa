"""Kernel-local import boundary tests.

These tests let the kernel package validate its own isolation without relying
on the root app test suite.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_KERNEL_ROOT = Path(__file__).resolve().parents[1]


def _collect_violations(
    root: Path,
    forbidden_prefixes: set[str],
) -> list[tuple[str, int, str]]:
    """Return (file_path, line_number, import_text) for every forbidden import."""
    violations: list[tuple[str, int, str]] = []
    for py_file in root.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if any(name.startswith(prefix) for prefix in forbidden_prefixes):
                        violations.append((str(py_file), node.lineno, f"import {name}"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if module is None:
                    continue
                if any(module.startswith(prefix) for prefix in forbidden_prefixes):
                    names = ", ".join(alias.name for alias in node.names)
                    violations.append((str(py_file), node.lineno, f"from {module} import {names}"))
    return violations


@pytest.mark.eval
def test_kernel_package_imports_no_app_or_sdk_modules() -> None:
    """kernel/ must remain isolated from app and SDK-layer packages."""
    violations = _collect_violations(
        _KERNEL_ROOT,
        forbidden_prefixes={"app.", "sdk.", "sdk.packs."},
    )
    if violations:
        lines = [f"  {path}:{lineno}: {text}" for path, lineno, text in violations]
        pytest.fail(
            "Forbidden higher-layer imports found in kernel:\n" + "\n".join(lines),
            pytrace=False,
        )
