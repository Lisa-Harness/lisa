"""Model-picker pack-local import boundary tests (sdk.packs.models)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel.testing.import_boundaries import collect_import_violations

_EXT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.eval
def test_model_picker_imports_no_app_modules() -> None:
    """sdk.packs.models must remain isolated from the app layer."""
    violations = collect_import_violations(
        _EXT_ROOT,
        forbidden_prefixes={"app."},
        allowed_prefixes={"kernel.", "sdk.", "sdk.packs."},
    )
    if violations:
        lines = [f"  {path}:{lineno}: {text}" for path, lineno, text in violations]
        pytest.fail(
            "Forbidden app-layer imports found in sdk.packs.models:\n" + "\n".join(lines),
            pytrace=False,
        )
