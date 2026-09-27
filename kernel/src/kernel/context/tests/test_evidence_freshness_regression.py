"""Regression tests for the kernel config.

Guards against one regression:

``kernel.config.Settings`` must import cleanly. Removing obsolete settings
fields has twice left dangling names in a module-level structure: the 2026-08-14
Pydantic-v2 breakage (dangling names in the numeric field validator made the
whole ``kernel`` package unimportable) and the memory-store cutover that
preceded it. This test constructs ``Settings`` after every field removal.
"""

from __future__ import annotations

from kernel.config import Settings


def test_settings_constructs_from_defaults() -> None:
    """The module-level Settings singleton can be constructed and reloaded."""

    settings = Settings()
    assert settings.max_tool_iterations == 300
    assert settings.sdk_content_roots == ""


def test_context_budget_lane_fields_are_host_owned() -> None:
    """The per-lane context budgets migrated to ``runtime.config``; the kernel
    keeps only the shared chars-per-token estimate and the mapper scoring
    weights. This pins the migration so accidental re-imports surface here.
    """

    settings = Settings()
    assert settings.context_budget_token_char_ratio == 4
    assert "authoritative" in settings.context_mapper_provenance_authority
    assert not hasattr(settings, "context_budget_enabled")
    assert not hasattr(settings, "memory_recall_enabled")
    assert not hasattr(settings, "passive_intelligence_enabled")
    assert not hasattr(settings, "workspace_id_fallback")
