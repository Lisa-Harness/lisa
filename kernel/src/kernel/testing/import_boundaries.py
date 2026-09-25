"""Shared import-boundary helpers for static architecture guard tests."""

from __future__ import annotations

import ast
from pathlib import Path

ImportViolation = tuple[str, int, str]


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


def _collect_file_import_violations(
    py_file: Path,
    forbidden_prefixes: set[str],
    allowed_prefixes: set[str] | None,
) -> list[ImportViolation]:
    """Return the import violations found in one parsed Python file."""
    violations: list[ImportViolation] = []
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


def collect_import_violations(
    root: Path,
    forbidden_prefixes: set[str],
    allowed_prefixes: set[str] | None = None,
) -> list[ImportViolation]:
    """Return (file_path, line_number, import_text) tuples for forbidden imports."""
    violations: list[ImportViolation] = []
    for py_file in root.rglob("*.py"):
        violations.extend(
            _collect_file_import_violations(py_file, forbidden_prefixes, allowed_prefixes)
        )
    return violations
