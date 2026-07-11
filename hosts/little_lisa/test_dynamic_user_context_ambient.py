from __future__ import annotations

import pytest

from . import main as little_lisa


@pytest.fixture(scope="module", autouse=True)
def _registered_framework() -> None:
    little_lisa.setup_framework()


def test_assemble_dynamic_user_context_includes_ambient_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "user_slug", "")
    monkeypatch.setattr(
        little_lisa,
        "build_ambient_facts",
        lambda **kwargs: "- local_date: 2026-07-11\n- timezone: UTC",
    )

    context = little_lisa._assemble_dynamic_user_context("default", "hello", "")

    assert "## Ambient Facts" in context
    assert "local_date: 2026-07-11" in context
    assert "timezone: UTC" in context
    assert "## Dynamic User Context" in context
    assert context.index("## Ambient Facts") < context.index("## Dynamic User Context")
