"""Import-boundary test for the docforge pack."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel.testing.import_boundaries import collect_import_violations

_PACK_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.eval
def test_docforge_imports_no_app_modules() -> None:
    """sdk.packs.docforge must remain isolated from the app layer."""
    violations = collect_import_violations(
        _PACK_ROOT,
        forbidden_prefixes={"app.", "hosts."},
        allowed_prefixes={"kernel.", "sdk.", "sdk.packs.", "markdown_it."},
    )
    if violations:
        lines = [f"  {path}:{lineno}: {text}" for path, lineno, text in violations]
        pytest.fail(
            "Forbidden app/host imports found in sdk.packs.docforge:\n" + "\n".join(lines),
            pytrace=False,
        )


def test_pack_imports_cleanly() -> None:
    """The pack must import without side effects."""
    import importlib

    importlib.import_module("sdk.packs.docforge")