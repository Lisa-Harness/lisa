"""Tests for host-supplied ``extra_item_builders`` plumbing in turn hooks.

The VS Code sidecar contributes host-side context (editor selection,
diagnostics) into the kernel's ``ExtensionItemBuilder`` seam.
``_build_item_builder_context`` composes host builders after the
entry-point builder — pack items win the ``item_id`` dedup — and leaves
the builder bare when none are supplied. ``handle_message`` threads the
builders through routing and turn execution so every resume path keeps
host context alive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from kernel.models import ContextItem
from runtime.testing import build_test_runtime
from runtime.turn_hooks_builder import _build_item_builder_context, build_turn_hooks
from sdk.context.composite_item_builder import CompositeExtensionItemBuilder
from sdk.context.extension_item_builder import EntryPointExtensionItemBuilder


def _make_runtime(tmp_path: Path) -> Any:
    models_yaml = "\n".join(
        [
            "version: 1",
            "mode_models: {default: gemma3:latest}",
            "router:",
            "  backend: lisa-router",
            "  rules: []",
            "models:",
            "  - model_id: gemma3:latest",
            "    provider: ollama",
        ]
    )
    return build_test_runtime(
        tmp_path,
        models_yaml=models_yaml,
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
    )


def _resolve_model(_model_id: str) -> Any:  # placeholder; never invoked here
    raise NotImplementedError


class _RecordingBuilder:
    """ExtensionItemBuilder double capturing the pair it is called with."""

    def __init__(self, items: list[ContextItem] | None = None) -> None:
        self.called_with: list[tuple[Any, Any]] = []
        self._items = items or []

    def build_extension_context_items(self, runtime_context: Any, state: Any) -> list[Any]:
        self.called_with.append((runtime_context, state))
        return list(self._items)


def _item(item_id: str) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        source="test",
        evidence_type="connector_status",
        title=item_id,
        body="body",
        collected_at="2026-01-01T00:00:00Z",
    )


def test_no_extra_builders_returns_bare_entry_builder(tmp_path: Path) -> None:
    """Without extras the entry-point builder is used directly (no composite)."""
    runtime = _make_runtime(tmp_path)
    item_builder, runtime_context = _build_item_builder_context(runtime, _resolve_model)
    assert isinstance(item_builder, EntryPointExtensionItemBuilder)
    assert not isinstance(item_builder, CompositeExtensionItemBuilder)
    assert runtime_context["runtime"] is runtime
    assert runtime_context["resolve_model"] is _resolve_model


def test_extra_builders_wrapped_in_composite_pack_first(tmp_path: Path) -> None:
    """The composite wraps entry-point first (pack wins dedup), then extras."""
    runtime = _make_runtime(tmp_path)
    host_builder = _RecordingBuilder()

    item_builder, _ = _build_item_builder_context(
        runtime, _resolve_model, extra_item_builders=[host_builder]
    )
    assert isinstance(item_builder, CompositeExtensionItemBuilder)
    builders: list[Any] = item_builder._builders  # ordered composition
    assert isinstance(builders[0], EntryPointExtensionItemBuilder)
    assert builders[1] is host_builder


def test_extra_builder_receives_shared_runtime_context(tmp_path: Path) -> None:
    """Host extras are called through the composite with the runtime context."""
    runtime = _make_runtime(tmp_path)
    host_builder = _RecordingBuilder()

    item_builder, runtime_context = _build_item_builder_context(
        runtime, _resolve_model, extra_item_builders=[host_builder]
    )
    # The entry-point group may or may not be installed in this environment;
    # calling the composite must still reach the host builder.
    item_builder.build_extension_context_items(runtime_context=runtime_context, state=None)
    assert host_builder.called_with, "host builder was not invoked through composite"
    ctx, _state = host_builder.called_with[0]
    assert ctx is runtime_context


def test_build_turn_hooks_composes_extras_into_context_seams(tmp_path: Path) -> None:
    """``build_turn_hooks`` exposes the composite on ``hooks.context_seams``."""
    runtime = _make_runtime(tmp_path)
    host_builder = _RecordingBuilder()

    hooks = build_turn_hooks(runtime, context=None, extra_item_builders=[host_builder])
    seams = hooks.context_seams
    assert isinstance(seams.extension_item_builder, CompositeExtensionItemBuilder)
    builders: list[Any] = seams.extension_item_builder._builders  # ordered composition
    assert isinstance(builders[0], EntryPointExtensionItemBuilder)
    assert builders[1] is host_builder

    # Invoking the composite reaches the host builder (no entry-point pack
    # contributes in this environment, but the call must not fail).
    seams.extension_item_builder.build_extension_context_items(
        runtime_context={"probe": True}, state=None
    )
    assert len(host_builder.called_with) == 1
    assert host_builder.called_with[0][0] == {"probe": True}


@pytest.mark.asyncio
async def test_build_unified_state_dedups_host_against_pack_items(tmp_path: Path) -> None:
    """Pack items win ``item_id``; host items with fresh ids are appended."""
    from runtime.state_provider import build_unified_state

    shared_id = "shared:record"
    pack_builder = _RecordingBuilder([_item(shared_id), _item("pack:only")])
    host_builder = _RecordingBuilder([_item(shared_id), _item("host:only")])
    composite = CompositeExtensionItemBuilder([pack_builder, host_builder])

    state = build_unified_state(
        state_snapshot=None,
        item_builder=composite,
        runtime_context={"probe": True},
    )
    ids = [item.item_id for item in state.context_items]
    # Pack-first ordering keeps the pack's copy of the shared id and the
    # host's uniquely-id'd records; no duplicates.
    assert ids == ["shared:record", "pack:only", "host:only"]


@pytest.mark.asyncio
async def test_composite_tolerates_raising_host_builder() -> None:
    """A broken host builder contributes nothing instead of failing the turn."""
    pack_builder = _RecordingBuilder([_item("pack:only")])

    class _BrokenBuilder:
        def build_extension_context_items(self, runtime_context: Any, state: Any) -> list[Any]:
            raise RuntimeError("editor bridge exploded")

    composite = CompositeExtensionItemBuilder([pack_builder, _BrokenBuilder()])
    items = composite.build_extension_context_items(runtime_context=None, state=None)
    assert [item.item_id for item in items] == ["pack:only"]
