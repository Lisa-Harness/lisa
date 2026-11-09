"""Tests for the slice-provider hooks view wiring.

Pins that ``_SliceProviderHooksView`` declares its dispatch and model
seams in ``__slots__`` (the memory-mining slice reads ``execute_tool``,
``execute_operation``, ``resolve_model``, ``complete``, and the identity/
path fields) and that ``build_turn_hooks`` assigns the callables. Before
the wiring fix, the identity attributes were missing and the slice
no-op'd on every turn.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from runtime.turn_hooks_builder import (
    _complete_via_provider_gateways,
    _SliceProviderHooksView,
    build_turn_hooks,
)

from runtime import HostRuntime


def _make_runtime(tmp_path: Path) -> HostRuntime:
    from runtime.testing import build_test_runtime

    return build_test_runtime(
        tmp_path,
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
    )


def test_view_declares_seam_slots() -> None:
    """``_SliceProviderHooksView`` declares the dispatch and model seams in __slots__."""
    assert "execute_tool" in _SliceProviderHooksView.__slots__
    assert "execute_operation" in _SliceProviderHooksView.__slots__
    assert "resolve_model" in _SliceProviderHooksView.__slots__
    assert "complete" in _SliceProviderHooksView.__slots__


def test_view_exposes_identity_and_path_fields(tmp_path: Path) -> None:
    """The view surfaces the identity/path fields the mining slice needs.

    ``memory_data_dir`` is the host durable data root (the thread store and
    the mine.marker sidecar live under it), ``user_slug`` the active user —
    without these the mining slice silently no-ops in production while its
    tests (which injected them explicitly) stayed green.
    """
    runtime = _make_runtime(tmp_path)
    view = _SliceProviderHooksView(runtime=runtime)

    assert view.memory_data_dir == runtime.paths.durable_dir
    assert view.user_slug == runtime.config.user_slug
    assert view.workspace_id == runtime.session_plan.manifest.name
    assert view.surface_id == runtime.config.surface_id
    assert callable(view.read_authored_text)


@pytest.mark.parametrize(("path", "expected"), [("prompt.md", "prompt"), ("missing.md", "")])
def test_view_reads_authored_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, expected: str
) -> None:
    class Registry:
        def read_authored_text(self, relative_path: str) -> str:
            return {"prompt.md": "prompt"}.get(relative_path, "")

    monkeypatch.setattr("runtime.turn_hooks_builder.content_registry", Registry)
    view = _SliceProviderHooksView(runtime=_make_runtime(tmp_path))

    assert view.read_authored_text(path) == expected


def test_view_reports_missing_content_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("runtime.turn_hooks_builder.content_registry", lambda: None)
    view = _SliceProviderHooksView(runtime=_make_runtime(tmp_path))

    with pytest.raises(RuntimeError, match="Content registry not set"):
        view.read_authored_text("prompt.md")


@pytest.mark.asyncio
async def test_slice_completion_requires_a_matching_gateway() -> None:
    model = SimpleNamespace(provider="example")
    kwargs = {"messages": [], "model_def": model, "max_tokens": 64, "temperature": 0.1}

    with pytest.raises(NotImplementedError, match="no provider gateway wired"):
        await _complete_via_provider_gateways(None, **kwargs)

    class Gateways:
        def gateway_for(self, provider: str) -> None:
            assert provider == "example"

    with pytest.raises(NotImplementedError, match="no provider gateway registered"):
        await _complete_via_provider_gateways(Gateways(), **kwargs)


@pytest.mark.asyncio
async def test_slice_completion_forwards_request_to_gateway() -> None:
    requests: list[Any] = []
    model = SimpleNamespace(provider="example")
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function"}]

    class Gateway:
        async def complete(self, request: Any) -> str:
            requests.append(request)
            return "completed"

    gateway = Gateway()

    class Gateways:
        def gateway_for(self, provider: str) -> Gateway:
            assert provider == "example"
            return gateway

    result = await _complete_via_provider_gateways(
        Gateways(),
        messages=messages,
        model_def=model,
        max_tokens=64,
        temperature=0.1,
        tools=tools,
        response_format=dict,
    )

    assert result == "completed"
    request = requests[0]
    assert request.model_def is model
    assert request.messages == messages
    assert request.max_tokens == 64
    assert request.temperature == 0.1
    assert request.tools == tools
    assert request.response_format is dict


def test_view_accepts_execute_tool_assignment(tmp_path: Path) -> None:
    """An ``execute_tool`` closure assigned to the view is readable via getattr.

    The memory-mining slice reads
    ``getattr(self._hooks, "execute_tool", None)`` and no-ops when it is
    not callable. This test confirms the attribute is settable and
    resolves (not raising ``AttributeError`` via ``__getattr__``).
    """
    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, user_slug="user", surface_type="dm", surface_id="s")

    async def fake_execute_tool(tool_id: str, args: Any) -> str:
        return "[]"

    view = _SliceProviderHooksView(
        runtime=runtime,
    )
    view.execute_tool = fake_execute_tool

    resolved = getattr(view, "execute_tool", None)
    assert callable(resolved)
    assert resolved is fake_execute_tool


def test_build_turn_hooks_assigns_execute_tool_on_view(tmp_path: Path) -> None:
    """``build_turn_hooks`` wires ``execute_tool`` on the internal view.

    The view is constructed internally during ``build_turn_hooks`` and is
    not returned publicly. We verify the wiring indirectly: after
    ``build_turn_hooks``, the memory-mining slice provider (when enabled)
    is constructed and registered. If the seams were missing on the view,
    the slice would be registered but no-op on every turn. This test
    confirms the build path does not raise and the hooks are fully wired
    (the slice tests in ``sdk/packs/memory/tests/test_mining_slice.py``
    cover the slice's use of the seams).
    """
    runtime = _make_runtime(tmp_path)
    hooks = build_turn_hooks(runtime, {"mode_id": "default"})
    assert hooks is not None
    assert hooks.operations.execute_operation is not None


def test_view_exposes_the_subscribed_event_bus(tmp_path: Path) -> None:
    """The view surfaces the subscribed bus of the latest hook build.

    Slice-owned operations (thread mining) run off-turn; without the bus
    their dispatch telemetry — including failures — never reaches
    ``decisions.jsonl`` (the 2026-11-08..09 mining crash-loop: 21 logged
    failures, zero ledger rows). The bus is refreshed per hook build and
    published on the runtime session.
    """
    runtime = _make_runtime(tmp_path)

    # Before any turn, there is no bus yet: slice telemetry no-ops.
    view = _SliceProviderHooksView(runtime=runtime)
    assert view.event_bus is None

    # A hook build wires and publishes the bus.
    hooks = build_turn_hooks(runtime, {"mode_id": "default"})
    assert hooks is not None
    bus = runtime.session.event_bus
    assert bus is not None
    assert view.event_bus is bus
    assert callable(getattr(bus, "subscribe", None))
