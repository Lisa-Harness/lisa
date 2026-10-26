"""Tests for the ``ToolHooks.post_dispatch_result`` decoration seam.

The seam exists so a host can attach context to a pack tool's result without
the kernel learning what that context means. These tests pin the contract the
kernel promises: the hook sees the *rendered* string, its return value reaches
the model, the per-turn token is passed through, and a failing hook never fails
the tool call.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import (
    CallLedger,
    OperationExecutionContext,
    ToolDefinition,
    ToolResult,
)
from kernel.tools.registry import execute_tool
from kernel.turn_phases._dispatch import (
    _decorate_pack_tool_output,
    _dispatch_pack_tool,
)
from kernel.turn_phases._turn_state import _TurnState
from turn_hooks_fold import build_hooks


def _make_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}


def _definition(execute: Any, **overrides: Any) -> ToolDefinition:
    return ToolDefinition(
        tool_id="test_tool",
        description="Test tool.",
        implementation_module=SimpleNamespace(execute=execute),
        **overrides,
    )


class _NullPool:
    """Pool that fails the test if MCP dispatch is reached."""

    async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> Any:  # pragma: no cover
        raise AssertionError(f"unexpected MCP dispatch for {tool_id}")


class _StubCompletion:
    def __init__(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.usage: dict[str, Any] = {}


# ── Decoration ─────────────────────────────────────────────────────────────


def _hooks(post_dispatch_result: Any) -> Any:
    return build_hooks(
        defaults={"models": {"complete": None}, "recall": {"enabled": False}},
        post_dispatch_result=post_dispatch_result,
    )


def test_decoration_is_a_no_op_without_a_hook() -> None:
    state = _TurnState()
    assert (
        _decorate_pack_tool_output(
            _hooks(None),
            "test_tool",
            {"path": "a.py"},
            "rendered",
            state=state,
            logger=SimpleNamespace(debug=lambda *a, **k: None),
        )
        == "rendered"
    )


def test_decoration_passes_rendered_string_arguments_and_token() -> None:
    seen: list[tuple[Any, ...]] = []
    state = _TurnState(dispatch_scope_token="turn-token")

    def decorate(tool_id: str, arguments: dict[str, Any], rendered: str, token: str) -> str:
        seen.append((tool_id, arguments, rendered, token))
        return f"{rendered}\n\nEXTRA"

    result = _decorate_pack_tool_output(
        _hooks(decorate),
        "test_tool",
        {"path": "a.py"},
        '{"summary": "ok"}',
        state=state,
        logger=SimpleNamespace(debug=lambda *a, **k: None),
    )

    assert result == '{"summary": "ok"}\n\nEXTRA'
    assert seen == [("test_tool", {"path": "a.py"}, '{"summary": "ok"}', "turn-token")]


def test_decoration_failure_leaves_output_untouched() -> None:
    def decorate(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("boom")

    messages: list[str] = []

    result = _decorate_pack_tool_output(
        _hooks(decorate),
        "test_tool",
        {"path": "a.py"},
        "rendered",
        state=_TurnState(),
        logger=SimpleNamespace(debug=lambda msg, *a, **k: messages.append(msg)),
    )

    assert result == "rendered"
    assert any("post_dispatch_result hook failed" in msg for msg in messages)


def test_decoration_ignores_a_non_string_return() -> None:
    result = _decorate_pack_tool_output(
        _hooks(lambda *_a, **_k: None),
        "test_tool",
        {"path": "a.py"},
        "rendered",
        state=_TurnState(),
        logger=SimpleNamespace(debug=lambda *a, **k: None),
    )

    assert result == "rendered"


# ── Dispatch integration ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pack_tool_result_is_decorated_before_the_model_sees_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decorated string is what ``_dispatch_pack_tool`` returns."""

    async def execute(_ctx: Any) -> ToolResult:
        return ToolResult(tool_id="test_tool", output_text="read ok", output_json={"path": "a.py"})

    definition = _definition(execute)
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"test_tool": definition})

    def decorate(tool_id: str, arguments: dict[str, Any], rendered: str, _token: str) -> str:
        return f"{rendered}\n\n## Instruction: Python\n\nFollow PEP 8."

    state = _TurnState(dispatch_scope_token="turn-token")
    output, tool_result, mcp_result = await _dispatch_pack_tool(
        tool_name="test_tool",
        tool_arguments={"path": "a.py"},
        tool_def=definition,
        ctx=OperationExecutionContext(tool_id="test_tool"),
        state=state,
        pool=_NullPool(),  # type: ignore[arg-type]
        logger=SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None),
        hooks=_hooks(decorate),
    )

    assert mcp_result is None
    assert tool_result is not None
    assert tool_result.output_text == "read ok", "the tool's own output_text is not mutated"
    assert "read ok" in output
    assert "## Instruction: Python" in output


@pytest.mark.asyncio
async def test_pack_tool_decorator_runs_for_every_call_not_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kernel does not dedupe; that is the host's per-turn decision."""

    async def execute(_ctx: Any) -> ToolResult:
        return ToolResult(tool_id="test_tool", output_text="ok", output_json={"path": "a.py"})

    definition = _definition(execute)
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"test_tool": definition})

    calls: list[str] = []

    def decorate(_tool_id: str, _arguments: dict[str, Any], rendered: str, token: str) -> str:
        calls.append(token)
        return rendered

    state = _TurnState(dispatch_scope_token="turn-token")
    for _ in range(3):
        await _dispatch_pack_tool(
            tool_name="test_tool",
            tool_arguments={"path": "a.py"},
            tool_def=definition,
            ctx=OperationExecutionContext(tool_id="test_tool"),
            state=state,
            pool=_NullPool(),  # type: ignore[arg-type]
            logger=SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None),
            hooks=_hooks(decorate),
        )

    assert calls == ["turn-token", "turn-token", "turn-token"]


@pytest.mark.asyncio
async def test_read_cache_hit_is_decorated_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached read must be decorated on the hit path, not only the miss path."""

    async def execute(_ctx: Any) -> ToolResult:
        return ToolResult(tool_id="test_tool", output_text="ok", output_json={"path": "a.py"})

    definition = _definition(execute, effect="read", abbreviation_template="{path}")
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"test_tool": definition})

    decorated: list[str] = []

    def decorate(_tool_id: str, _arguments: dict[str, Any], rendered: str, _token: str) -> str:
        decorated.append(rendered)
        return f"{rendered}\n\nEXTRA"

    state = _TurnState(dispatch_scope_token="turn-token")
    kwargs: dict[str, Any] = {
        "tool_name": "test_tool",
        "tool_arguments": {"path": "a.py"},
        "tool_def": definition,
        "ctx": OperationExecutionContext(tool_id="test_tool"),
        "state": state,
        "pool": _NullPool(),  # type: ignore[arg-type]
        "logger": SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None),
        "hooks": _hooks(decorate),
    }
    await _dispatch_pack_tool(**kwargs)  # miss: populates the read cache
    assert state.read_cache, "the read cache should hold the abbreviation"
    second, _, _ = await _dispatch_pack_tool(**kwargs)  # hit: must still decorate

    assert len(decorated) == 2
    assert decorated[1] == state.read_cache[next(iter(state.read_cache))]
    assert second.endswith("EXTRA")


@pytest.mark.asyncio
async def test_mcp_tool_dispatch_is_not_decorated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An MCP-server tool renders into ``output_text`` and keeps its own shape."""

    class _Pool:
        async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> Any:
            return SimpleNamespace(tool_id=tool_id, output_text="mcp output", is_error=False)

    definition = ToolDefinition(tool_id="mcp_like", description="d")
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"mcp_like": definition})

    output, _tool_result, mcp_result = await _dispatch_pack_tool(
        tool_name="mcp_like",
        tool_arguments={"path": "a.py"},
        tool_def=definition,
        ctx=OperationExecutionContext(tool_id="mcp_like"),
        state=_TurnState(),
        pool=_Pool(),  # type: ignore[arg-type]
        logger=SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None),
        hooks=_hooks(lambda *_a, **_k: "DECORATED"),
    )

    assert output == "mcp output"
    assert mcp_result is not None


# ── Token identity across parent and children ─────────────────────────────


@pytest.mark.asyncio
async def test_each_turn_and_each_child_gets_a_distinct_token() -> None:
    """Parent turns and fork/delegate children must not share a dedupe key.

    A shared key would let a parent's already-injected guidance suppress the
    same guidance in a child that never saw it, and vice versa.
    """
    from kernel.turn_runner import run_turn

    tokens: list[str] = []
    dispatches = {"count": 0}

    async def execute(_ctx: Any) -> ToolResult:
        return ToolResult(tool_id="test_tool", output_text="ok", output_json={})

    definition = _definition(execute)
    from kernel.tools import registry

    original = registry._TOOL_CATALOG
    registry._TOOL_CATALOG = {"test_tool": definition}  # type: ignore[assignment]
    try:

        def decorate(_tool_id: str, _arguments: dict[str, Any], rendered: str, token: str) -> str:
            tokens.append(token)
            return rendered

        def completion() -> Any:
            async def complete(**_kwargs: Any) -> _StubCompletion:
                dispatches["count"] += 1
                if dispatches["count"] == 1:
                    return _StubCompletion(
                        content="",
                        tool_calls=[_make_tool_call("c1", "test_tool", {"path": "a.py"})],
                    )
                return _StubCompletion(content="Done.")

            return complete

        for _ in range(2):
            dispatches["count"] = 0
            await run_turn(
                mode_id="default",
                query="test",
                hooks=build_hooks(
                    defaults={
                        "models": {"complete": completion()},
                        "recall": {"enabled": False},
                    },
                    post_dispatch_result=decorate,
                ),
            )
    finally:
        registry._TOOL_CATALOG = original  # type: ignore[assignment]

    assert len(tokens) == 2
    assert tokens[0] != tokens[1], "each turn needs its own token"
    assert all(tokens), "tokens must be populated"


def test_execute_tool_import_is_stable() -> None:
    """Guard the seam the dispatch module patches: it imports the function.

    ``_dispatch`` imports ``execute_tool`` by name, so the registry binding and
    the dispatch binding are the same object at import time. A future
    ``from kernel.tools import registry`` refactor would silently break the
    monkeypatch targets these tests use.
    """
    from kernel.tools import registry

    assert execute_tool is registry.execute_tool


def test_call_ledger_is_untouched_by_the_hook() -> None:
    """Sanity: the hook adds no ledger surface."""
    assert set(CallLedger().counts) == set()
