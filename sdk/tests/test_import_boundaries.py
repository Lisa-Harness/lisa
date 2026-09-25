"""SDK-local import boundary tests.

These tests let the SDK package validate its own isolation without relying on
the root app test suite.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SDK_ROOT = Path(__file__).resolve().parents[1]


def _is_forbidden_import(
    name: str,
    forbidden_prefixes: set[str],
    allowed_prefixes: set[str] | None,
) -> bool:
    """Return True when ``name`` is forbidden and not explicitly allowed."""
    forbidden = any(name.startswith(prefix) for prefix in forbidden_prefixes)
    allowed = bool(allowed_prefixes) and any(
        name.startswith(prefix) for prefix in (allowed_prefixes or set())
    )
    return forbidden and not allowed


def _collect_file_violations(
    py_file: Path,
    forbidden_prefixes: set[str],
    allowed_prefixes: set[str] | None,
) -> list[tuple[str, int, str]]:
    """Return (file_path, line_number, import_text) for one file's forbidden imports."""
    violations: list[tuple[str, int, str]] = []
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if _is_forbidden_import(name, forbidden_prefixes, allowed_prefixes):
                    violations.append((str(py_file), node.lineno, f"import {name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module is None:
                continue
            if _is_forbidden_import(module, forbidden_prefixes, allowed_prefixes):
                names = ", ".join(alias.name for alias in node.names)
                violations.append((str(py_file), node.lineno, f"from {module} import {names}"))
    return violations


def _collect_violations(
    root: Path,
    forbidden_prefixes: set[str],
    allowed_prefixes: set[str] | None = None,
) -> list[tuple[str, int, str]]:
    """Return (file_path, line_number, import_text) for every forbidden import."""
    violations: list[tuple[str, int, str]] = []
    for py_file in root.rglob("*.py"):
        violations.extend(_collect_file_violations(py_file, forbidden_prefixes, allowed_prefixes))
    return violations


@pytest.mark.eval
def test_sdk_package_imports_no_app_modules() -> None:
    """sdk/ must remain isolated from the app layer."""
    violations = _collect_violations(
        _SDK_ROOT,
        forbidden_prefixes={"app."},
        allowed_prefixes={"kernel.", "sdk.packs."},
    )
    if violations:
        lines = [f"  {path}:{lineno}: {text}" for path, lineno, text in violations]
        pytest.fail(
            "Forbidden app-layer imports found in sdk:\n" + "\n".join(lines),
            pytrace=False,
        )
