"""Shared import-boundary helpers for static architecture guard tests."""

from __future__ import annotations

import ast
from pathlib import Path

ImportViolation = tuple[str, int, str]


def collect_import_violations(
    root: Path,
    forbidden_prefixes: set[str],
    allowed_prefixes: set[str] | None = None,
) -> list[ImportViolation]:
    """Return (file_path, line_number, import_text) tuples for forbidden imports."""
    violations: list[ImportViolation] = []
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
                        if allowed_prefixes and any(
                            name.startswith(prefix) for prefix in allowed_prefixes
                        ):
                            continue
                        violations.append((str(py_file), node.lineno, f"import {name}"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if module is None:
                    continue
                if any(module.startswith(prefix) for prefix in forbidden_prefixes):
                    if allowed_prefixes and any(
                        module.startswith(prefix) for prefix in allowed_prefixes
                    ):
                        continue
                    names = ", ".join(alias.name for alias in node.names)
                    violations.append((str(py_file), node.lineno, f"from {module} import {names}"))
    return violations
