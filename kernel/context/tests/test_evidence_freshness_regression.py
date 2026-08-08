"""Regression tests for the kernel config.

Guards against one regression:

``kernel.config.Settings`` must import cleanly. Removing the memory-store
settings fields left dangling names in the numeric field validator, which
made the whole ``kernel`` package unimportable under Pydantic v2.
"""

from __future__ import annotations

from kernel.config import Settings


def test_settings_imports_and_has_active_memory_knobs() -> None:
    """The module-level Settings singleton can be constructed and retains the
    active memory recall/auto-capture knobs after the memory-store removal.
    """
    settings = Settings()
    assert settings.memory_recall_enabled is True
    assert settings.memory_recall_pre_turn_limit == 3
    assert settings.memory_auto_capture_enabled is True
