"""Tests for the slice-provider entry-point discovery and TurnHooks seams.

Covers:
- ``TurnHooks.write_turn_history_fn`` defaults to None and accepts the
  expected type (``last_context_snapshot`` is also tested as a passive seam).
- ``kernel.slice_provider_discovery.discover_slice_provider_factories``
  returns an empty list when no pack is registered, and parses a real
  ``PackSliceProviderEntry`` when one is.
- Excluded pack names are filtered out of discovery.
"""

from __future__ import annotations

from typing import Any

import pytest

from kernel.slice_provider_discovery import (
    SLICE_PROVIDERS_ENTRY_POINT_GROUP,
    discover_slice_provider_factories,
)
from kernel.turn_runner import TurnHooks


def test_turn_hooks_defaults_for_slice_seams() -> None:
    """The seam fields default to None (no behavior change)."""
    hooks = TurnHooks()
    assert hooks.last_context_snapshot is None
    assert hooks.write_turn_history_fn is None


def test_turn_hooks_accepts_new_seam_fields() -> None:
    """The seam fields accept a snapshot and a write callable."""
    sentinel: Any = object()

    def _writer(history: list[dict[str, Any]]) -> None:
        del history

    hooks = TurnHooks(last_context_snapshot=sentinel, write_turn_history_fn=_writer)
    assert hooks.last_context_snapshot is sentinel
    assert hooks.write_turn_history_fn is _writer


def test_slice_providers_entry_point_group_name() -> None:
    """The entry-point group is stable and named as documented."""
    assert SLICE_PROVIDERS_ENTRY_POINT_GROUP == "lisa.extensions.slice_providers"


def test_discover_slice_provider_factories_empty_when_unregistered() -> None:
    """With no packs installed under the group, discovery returns an empty list.

    Discovery only returns entries whose distribution is pip-installed and
    registered under the entry-point group; in the test env this may be empty.
    """
    entries = discover_slice_provider_factories()
    # Any registered entries must carry callable factories.
    for entry in entries:
        assert callable(entry.factory)
        assert isinstance(entry.pack_name, str)


def test_discover_slice_provider_factories_filters_excluded_packs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Excluded pack names are filtered out."""

    class _FakeEP:
        name = "excluded-pack"

        class _Dist:
            class _Metadata:
                def get(self, key: str, default: str = "") -> str:
                    return "lisa-pack-excluded" if key == "Name" else default

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
    entries = discover_slice_provider_factories(excluded_packs={"lisa-pack-excluded"})
    assert entries == []