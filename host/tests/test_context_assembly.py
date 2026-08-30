"""Tests for dynamic user context assembly.

Migrated from ``hosts/little_lisa/test_dynamic_user_context_ambient.py``.

The original test monkeypatched ``little_lisa.build_ambient_facts`` and
``little_lisa._RUNTIME_STATE["user_slug"]``. In the new structure,
``assemble_dynamic_user_context`` is a pure function in
``host.context_assembly`` that takes all dependencies as keyword
arguments — there is no ``_RUNTIME_STATE``.
"""

from __future__ import annotations

import pytest

from host import context_assembly
from host.session_context import ReplayControls


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
        recent_thread_excerpt_builder=lambda _n: "",
    )

    assert "Selected User Context" not in context
    assert "user_slug: none" in context


def test_assemble_dynamic_user_context_replay_manual_disables_history_sections() -> None:
    context = context_assembly.assemble_dynamic_user_context(
        "default",
        "hello there",
        "",
        content_loader=lambda _path: "",
        user_slug="adam",
        timezone="UTC",
        state=None,
        session_context_loader=lambda: "",
        recent_thread_excerpt_builder=lambda _n: "Recent thread context:\n- hi: there",
        replay_controls_loader=lambda: ReplayControls(trigger_mode="manual"),
    )

    assert 'source="thread_manifest"' not in context
    assert 'source="recent_conversation"' not in context
    assert "replay_trigger_mode: manual" in context


def test_assemble_dynamic_user_context_replay_conditional_min_query_chars() -> None:
    context = context_assembly.assemble_dynamic_user_context(
        "default",
        "short",
        "",
        content_loader=lambda _path: "",
        user_slug="adam",
        timezone="UTC",
        state=None,
        session_context_loader=lambda: "",
        recent_thread_excerpt_builder=lambda _n: "Recent thread context:\n- hi: there",
        replay_controls_loader=lambda: ReplayControls(
            trigger_mode="conditional",
            min_query_chars=20,
        ),
    )

    assert 'source="thread_manifest"' not in context
    assert 'source="recent_conversation"' not in context
    assert "replay_min_query_chars: 20" in context


def test_assemble_dynamic_user_context_replay_includes_diagnostic_reflection_toggle() -> None:
    context = context_assembly.assemble_dynamic_user_context(
        "default",
        "hello there",
        "",
        content_loader=lambda _path: "",
        user_slug="adam",
        timezone="UTC",
        state=None,
        session_context_loader=lambda: "",
        recent_thread_excerpt_builder=lambda _n: "Recent thread context:\n- hi: there",
        replay_controls_loader=lambda: ReplayControls(
            include_diagnostic_reflection=False,
        ),
    )

    assert "Diagnostic Reflection" not in context


def test_recent_conversation_wins_history_budget_over_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recent conversation must be allocated before the manifest in the history lane.

    Both sections share the ``history`` lane. When the lane budget cannot
    fit both, the recent-conversation excerpt (authoritative, carries the
    assistant's last answer the user is replying to) must survive; the
    thread manifest (inferred, addressable table of older turns) is the one
    that gets dropped. Allocating the manifest first amputates the turn the
    user is responding to (e.g. an Option A/B offer), so the next bare reply
    ("B") sees no record of the options and the model honestly reports it
    lost the context.
    """

    monkeypatch.setattr(
        context_assembly,
        "_DEFAULT_CONTEXT_BUDGET_VALUES",
        {
            "enabled": True,
            "token_char_ratio": 4,
            "core_max_tokens": 2000,
            "retrieval_max_tokens": 1500,
            "memory_max_tokens": 1200,
            "reflection_max_tokens": 1000,
            "history_max_tokens": 1500,  # production default: ~6000 chars
            "operational_max_tokens": 800,
        },
    )

    recent_excerpt = "Recent thread context:\n- proceed with option B: pick B"
    # ~6552 chars so it exceeds the 1500-token lane. With the manifest
    # allocated first it fills the lane and the recent excerpt is dropped
    # with ``budget_exhausted`` — the exact production bug.
    large_manifest = "x" * 6552

    context = context_assembly.assemble_dynamic_user_context(
        "default",
        "B",
        "",
        content_loader=lambda _path: "manifest hint",
        user_slug="adam",
        timezone="UTC",
        state=None,
        session_context_loader=lambda: "",
        recent_thread_excerpt_builder=lambda _n: recent_excerpt,
        thread_manifest_builder=lambda: large_manifest,
    )

    # The recent-conversation excerpt (the Option A/B answer) must survive.
    assert 'source="recent_conversation"' in context
    assert "proceed with option B" in context
    # The manifest must NOT arrive whole: either dropped (budget exhausted)
    # or truncated to the lane remainder — never full-strength alongside the
    # recent excerpt. Assert it was not allocated its full 6552 chars.
    assert context.count("xxxx") < 6552
