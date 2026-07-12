"""Tests for dynamic user context assembly.

Migrated from ``hosts/little_lisa/test_dynamic_user_context_ambient.py``.

The original test monkeypatched ``little_lisa.build_ambient_facts`` and
``little_lisa._RUNTIME_STATE["user_slug"]``. In the new structure,
``assemble_dynamic_user_context`` is a pure function in
``hosts.core.context_assembly`` that takes all dependencies as keyword
arguments — there is no ``_RUNTIME_STATE``.
"""

from __future__ import annotations

import pytest

from hosts.core import context_assembly


def test_assemble_dynamic_user_context_includes_ambient_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        context_assembly,
        "build_ambient_facts",
        lambda **kwargs: "- local_date: 2026-07-11\n- timezone: UTC",
    )

    context = context_assembly.assemble_dynamic_user_context(
        "default",
        "hello",
        "",
        content_loader=lambda _path: "",
        user_slug="",
        timezone="UTC",
        state=None,
        session_context_loader=lambda: "",
        thread_digest_loader=lambda: "",
        recent_thread_excerpt_builder=lambda _n: "",
    )

    assert "## Ambient Facts" in context
    assert "local_date: 2026-07-11" in context
    assert "timezone: UTC" in context
    assert "## Dynamic User Context" in context
    assert context.index("## Ambient Facts") < context.index("## Dynamic User Context")


def test_assemble_dynamic_user_context_skips_user_context_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_read_authored_text(relative_path: str) -> str:
        if relative_path == "user_contexts/default.md":
            raise AssertionError("default user context should not be loaded")
        return ""

    monkeypatch.setattr(
        context_assembly,
        "build_ambient_facts",
        lambda **kwargs: "- timezone: UTC",
    )

    context = context_assembly.assemble_dynamic_user_context(
        "default",
        "hello",
        "",
        content_loader=_fake_read_authored_text,
        user_slug="",
        timezone="UTC",
        state=None,
        session_context_loader=lambda: "",
        thread_digest_loader=lambda: "",
        recent_thread_excerpt_builder=lambda _n: "",
    )

    assert "Selected User Context" not in context
    assert "user_slug: none" in context