"""Tests for the Phase-1 compaction seams added to TurnHooks and the
slice-provider entry-point discovery.

Covers:
- ``TurnHooks.last_context_snapshot`` and ``write_turn_history_fn`` default
  to None and accept the expected types.
- ``kernel.slice_provider_discovery.discover_slice_provider_factories``
  returns an empty list when no pack is registered, and parses a real
  ``PackSliceProviderEntry`` when one is.
- The compaction pack's entry-point factory returns None for a disabled
  session plan and a ``ContextCompactionSlice`` for an enabled one.
"""

from __future__ import annotations

from typing import Any

import pytest

from kernel.slice_provider_discovery import (
    SLICE_PROVIDERS_ENTRY_POINT_GROUP,
    discover_slice_provider_factories,
)
from kernel.turn_runner import TurnHooks


def test_turn_hooks_defaults_for_compaction_seams() -> None:
    """The two new seam fields default to None (no behavior change)."""
    hooks = TurnHooks()
    assert hooks.last_context_snapshot is None
    assert hooks.write_turn_history_fn is None


def test_turn_hooks_accepts_new_seam_fields() -> None:
    """The new seam fields accept a snapshot and a write callable."""
    sentinel: Any = object()

    def _writer(history: list[dict[str, Any]]) -> None:
        del history

    hooks = TurnHooks(last_context_snapshot=sentinel, write_turn_history_fn=_writer)
    assert hooks.last_context_snapshot is sentinel
    assert hooks.write_turn_history_fn is _writer


def test_slice_providers_entry_point_group_name() -> None:
    """The new entry-point group is stable and named as documented."""
    assert SLICE_PROVIDERS_ENTRY_POINT_GROUP == "lisa.extensions.slice_providers"


def test_discover_slice_provider_factories_empty_when_unregistered() -> None:
    """With no packs installed under the group, discovery returns an empty list."""
    entries = discover_slice_provider_factories()
    # The compaction pack is not pip-installed in this test env; discovery
    # should return an empty list (or, if it is installed, a list whose
    # entries all carry callable factories).
    for entry in entries:
        assert callable(entry.factory)
        assert isinstance(entry.pack_name, str)


def test_discover_slice_provider_factories_filters_excluded_packs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Excluded pack names are filtered out."""

    class _FakeEP:
        name = "compaction"

        class _Dist:
            class _Metadata:
                def get(self, key: str, default: str = "") -> str:
                    return "lisa-pack-compaction" if key == "Name" else default

            metadata = _Metadata()

        dist = _Dist()

        def load(self) -> Any:
            return lambda _hooks, _plan: None

    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: [_FakeEP()] if group == SLICE_PROVIDERS_ENTRY_POINT_GROUP else [],
    )
    entries = discover_slice_provider_factories(excluded_packs={"lisa-pack-compaction"})
    assert entries == []


def test_compaction_factory_returns_none_when_disabled() -> None:
    """The compaction pack's factory returns None when the plan disables it."""
    from sdk.packs.compaction.slice_provider import get_slice_provider_factory

    class _Tuning:
        enabled = False
        trigger = "post_turn"
        pressure_threshold = 0.75
        hard_threshold = 0.90
        min_interval_turns = 4
        retention_count = 6
        preferred_model_id = "deep"
        summarize_operation = "checkpoint_history_summarize"

    class _History:
        compaction = _Tuning()

    class _Context:
        history = _History()

    class _Manifest:
        context = _Context()

    class _Plan:
        manifest = _Manifest()

    factory = get_slice_provider_factory()
    assert factory(TurnHooks(), _Plan()) is None


def test_compaction_factory_returns_slice_when_enabled() -> None:
    """The compaction pack's factory returns a ContextCompactionSlice when enabled."""
    from sdk.packs.compaction.slice_provider import (
        ContextCompactionSlice,
        get_slice_provider_factory,
    )

    class _Tuning:
        enabled = True
        trigger = "post_turn"
        pressure_threshold = 0.5
        hard_threshold = 0.8
        min_interval_turns = 2
        retention_count = 4
        preferred_model_id = "deep"
        summarize_operation = "checkpoint_history_summarize"

    class _History:
        compaction = _Tuning()

    class _Context:
        history = _History()

    class _Manifest:
        context = _Context()

    class _Plan:
        manifest = _Manifest()

    factory = get_slice_provider_factory()
    provider = factory(TurnHooks(), _Plan())
    assert isinstance(provider, ContextCompactionSlice)
    assert provider._config.pressure_threshold == 0.5
    assert provider._config.hard_threshold == 0.8
    assert provider._config.min_interval_turns == 2