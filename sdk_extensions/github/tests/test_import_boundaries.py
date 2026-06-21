"""GitHub extension-local import boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel.testing.import_boundaries import collect_import_violations

_EXT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.eval
def test_github_extension_imports_no_app_modules() -> None:
    """sdk_extensions/github must remain isolated from the app layer."""
    violations = collect_import_violations(
        _EXT_ROOT,
        forbidden_prefixes={"app."},
        allowed_prefixes={"kernel.", "lisa_sdk.", "sdk_extensions."},
    )
    if violations:
        lines = [f"  {path}:{lineno}: {text}" for path, lineno, text in violations]
        pytest.fail(
            "Forbidden app-layer imports found in sdk_extensions.github:\n" + "\n".join(lines),
            pytrace=False,
        )
