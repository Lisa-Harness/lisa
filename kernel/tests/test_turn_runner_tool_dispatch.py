"""Tests for tool dispatch branches in the turn runner.

Covers the three dispatch paths that fire after transition_stage interception:
  1. Operation dispatch — via hooks.execute_operation
  2. In-process pack tool dispatch — via execute_tool from registry
  3. MCP tool dispatch — via pool.call_tool

Also covers the CallLedger per-target max_calls_per_turn enforcement,
output stringification helpers, and error handling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from kernel.models import (
    CallLedger,
    OperationExecutionContext,
    OperationResult,
    ToolResult,
)
from kernel.turn_runner import (
    TurnHooks,
    _build_minimal_context,
    _extract_pre_turn_briefing,
    _stringify_pack_tool_result,
    _stringify_result,
    _to_tool_context,
    run_turn,
)

# ── Stubs ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None


def _make_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _build_hooks(complete_fn: Any, **overrides: Any) -> TurnHooks:
    return TurnHooks(complete=complete_fn, memory_recall_enabled=False, **overrides)


# ── CallLedger ────────────────────────────────────────────────────────


def test_call_ledger_record_increments_count() -> None:
    """record() increments the per-target counter."""
    ledger = CallLedger()
    ledger.record("tool_a")
    ledger.record("tool_a")
    ledger.record("tool_b")
    assert ledger.counts["tool_a"] == 2
    assert ledger.counts["tool_b"] == 1


def test_call_ledger_is_exhausted_at_cap() -> None:
    """is_exhausted returns True when count reaches max_calls."""
    ledger = CallLedger()
    ledger.record("tool_a")
    ledger.record("tool_a")
    assert ledger.is_exhausted("tool_a", 2) is True


def test_call_ledger_not_exhausted_below_cap() -> None:
    """is_exhausted returns False when count is below max_calls."""
    ledger = CallLedger()
    ledger.record("tool_a")
    assert ledger.is_exhausted("tool_a", 2) is False


def test_call_ledger_unlimited_when_max_calls_zero() -> None:
    """max_calls=0 means unlimited — is_exhausted always returns False."""
    ledger = CallLedger()
    ledger.record("tool_a")
    ledger.record("tool_a")
    ledger.record("tool_a")
    assert ledger.is_exhausted("tool_a", 0) is False


def test_call_ledger_remaining_calculates_correctly() -> None:
    """remaining() returns max_calls - current count."""
    ledger = CallLedger()
    ledger.record("tool_a")
    assert ledger.remaining("tool_a", 3) == 2


def test_call_ledger_remaining_unlimited_returns_max_calls() -> None:
    """When max_calls <= 0 (unlimited), remaining returns max_calls as-is."""
    ledger = CallLedger()
    ledger.record("tool_a")
    assert ledger.remaining("tool_a", 0) == 0


def test_call_ledger_separate_targets_independent() -> None:
    """Each target_id has an independent count."""
    ledger = CallLedger()
    ledger.record("tool_a")
    assert ledger.is_exhausted("tool_a", 1) is True
    assert ledger.is_exhausted("tool_b", 1) is False


# ── _stringify_result (operation result) ──────────────────────────────


def test_stringify_result_prefers_output_text() -> None:
    """When output_text is non-empty, it's used (JSON is ignored)."""
    result = OperationResult(output_text="Plain text output", output_json={"key": "val"})
    assert _stringify_result(result) == "Plain text output"


def test_stringify_result_falls_back_to_json_when_text_empty() -> None:
    """When output_text is empty, output_json is serialized."""
    result = OperationResult(output_text="", output_json={"key": "value"})
    serialized = _stringify_result(result)
    assert "key" in serialized
    assert "value" in serialized


def test_stringify_result_returns_done_when_both_empty() -> None:
    """When both output_text and output_json are empty, returns 'Done'."""
    result = OperationResult(output_text="", output_json={})
    assert _stringify_result(result) == "Done"


def test_stringify_result_strips_whitespace() -> None:
    """Leading/trailing whitespace is stripped from output_text."""
    result = OperationResult(output_text="  padded  ")
    assert _stringify_result(result) == "padded"


# ── _stringify_pack_tool_result ───────────────────────────────────────


def test_stringify_pack_tool_result_concatenates_summary_and_json() -> None:
    """When both summary and JSON payload are present, both are emitted."""
    result = ToolResult(
        tool_id="test_tool",
        output_text="Listed 5 entries.",
        output_json={"entries": [1, 2, 3, 4, 5]},
    )
    rendered = _stringify_pack_tool_result(result)
    assert "Listed 5 entries." in rendered
    assert "entries" in rendered
    assert "1" in rendered and "5" in rendered  # values present in indented JSON


def test_stringify_pack_tool_result_summary_only() -> None:
    """When only output_text is present, just the summary is returned."""
    result = ToolResult(tool_id="test_tool", output_text="Done.", output_json={})
    assert _stringify_pack_tool_result(result) == "Done."


def test_stringify_pack_tool_result_json_only() -> None:
    """When only output_json is present, just the JSON is returned."""
    result = ToolResult(tool_id="test_tool", output_text="", output_json={"data": "x"})
    rendered = _stringify_pack_tool_result(result)
    assert '"data"' in rendered
    assert "x" in rendered


def test_stringify_pack_tool_result_both_empty_returns_done() -> None:
    """When both are empty, returns 'Done'."""
    result = ToolResult(tool_id="test_tool")
    assert _stringify_pack_tool_result(result) == "Done"


def test_stringify_pack_tool_result_summary_stripped() -> None:
    """Whitespace in the summary is stripped before concatenation."""
    result = ToolResult(tool_id="t", output_text="  spaced  ", output_json={"a": 1})
    rendered = _stringify_pack_tool_result(result)
    assert rendered.startswith("spaced")


def test_stringify_pack_tool_result_surfaces_error_when_no_output() -> None:
    """When a tool returns an error with empty output_text/json, the error is surfaced.

    Regression: previously, _stringify_pack_tool_result ignored the ``error``
    field on ToolResult. When a pack tool (e.g. coding_list_directory) failed
    with an error (e.g. missing workspace_root config), output_text and
    output_json were both empty, so the function returned "Done" — hiding
    the actual error from the model and causing it to report that the tool
    "returned a generic Done message" instead of diagnosing the failure.
    """
    result = ToolResult(
        tool_id="coding_list_directory",
        error="workspace_root is not a directory: /nonexistent",
    )
    rendered = _stringify_pack_tool_result(result)
    assert "Error:" in rendered
    assert "workspace_root is not a directory" in rendered
    assert rendered != "Done"


def test_stringify_pack_tool_result_error_alongside_summary() -> None:
    """When both summary and error are present, both are emitted."""
    result = ToolResult(
        tool_id="coding_list_directory",
        output_text="Listed 0 entries.",
        error="Partial failure: some paths unreadable",
    )
    rendered = _stringify_pack_tool_result(result)
    assert "Listed 0 entries." in rendered
    assert "Error:" in rendered
    assert "Partial failure" in rendered


def test_stringify_pack_tool_result_error_alongside_json() -> None:
    """When both JSON payload and error are present, both are emitted."""
    result = ToolResult(
        tool_id="coding_find_files",
        output_json={"files": ["a.py"]},
        error="Search was truncated",
    )
    rendered = _stringify_pack_tool_result(result)
    assert "Error:" in rendered
    assert "Search was truncated" in rendered
    assert '"files"' in rendered
    assert "a.py" in rendered


# ── _extract_pre_turn_briefing ─────────────────────────────────────────


def test_extract_pre_turn_briefing_from_output_text() -> None:
    """When output_text is non-empty, it's used directly."""
    result = OperationResult(output_text="Briefing text here")
    assert _extract_pre_turn_briefing(result) == "Briefing text here"


def test_extract_pre_turn_briefing_from_json_output_text_field() -> None:
    """When output_text is empty but output_json has output_text, that's used."""
    result = OperationResult(
        output_text="",
        output_json={"output_text": "From JSON", "other": "ignored"},
    )
    assert _extract_pre_turn_briefing(result) == "From JSON"


def test_extract_pre_turn_briefing_from_json_body_markdown_field() -> None:
    """Falls back to body_markdown field in output_json."""
    result = OperationResult(
        output_text="",
        output_json={"body_markdown": "# Markdown briefing"},
    )
    assert _extract_pre_turn_briefing(result) == "# Markdown briefing"


def test_extract_pre_turn_briefing_from_json_summary_field() -> None:
    """Falls back to summary field in output_json."""
    result = OperationResult(
        output_text="",
        output_json={"summary": "A short summary"},
    )
    assert _extract_pre_turn_briefing(result) == "A short summary"


def test_extract_pre_turn_briefing_empty_when_all_fields_empty() -> None:
    """When all fields are empty, returns empty string."""
    result = OperationResult(output_text="", output_json={"output_text": ""})
    assert _extract_pre_turn_briefing(result) == ""


# ── _to_tool_context ──────────────────────────────────────────────────


def test_to_tool_context_carries_identity_fields() -> None:
    """_to_tool_context copies identity/scope fields from the source context."""
    ctx = OperationExecutionContext(
        workspace_id="ws-1",
        user_id="user-1",
        user_slug="slug-1",
        surface_type="personal_dm",
        surface_id="surf-1",
        memory_scope="personal",
        mode_id="default",
        query="hello",
    )
    tool_ctx = _to_tool_context("my_tool", ctx)
    assert tool_ctx.tool_id == "my_tool"
    assert tool_ctx.workspace_id == "ws-1"
    assert tool_ctx.user_id == "user-1"
    assert tool_ctx.user_slug == "slug-1"
    assert tool_ctx.mode_id == "default"


def test_to_tool_context_defaults_when_source_missing_fields() -> None:
    """_to_tool_context uses getattr defaults when source ctx lacks fields."""
    ctx = SimpleNamespace()
    tool_ctx = _to_tool_context("my_tool", ctx)
    assert tool_ctx.tool_id == "my_tool"
    assert tool_ctx.workspace_id == "default"
    assert tool_ctx.user_slug == "default"
    assert tool_ctx.parameters == {}


# ── _build_minimal_context ─────────────────────────────────────────────


def test_build_minimal_context_populates_all_fields() -> None:
    """_build_minimal_context builds an OperationExecutionContext with all fields."""
    ctx = _build_minimal_context(
        mode_id="default",
        query="test",
        resolve_model_fn=lambda x: x,
        complete_fn=lambda **kw: None,
        read_authored_text_fn=lambda *a: "",
        execute_tool_fn=lambda *a: None,
        execute_operation_fn=lambda *a: None,
        workspace_id="ws",
        user_id="u",
        user_slug="s",
        user_name="Name",
        principal_key="pk",
        conversation_id="conv",
        surface_type="personal_dm",
        surface_id="sid",
        memory_scope="personal",
        session_plan_digest="digest",
    )
    assert ctx.mode_id == "default"
    assert ctx.query == "test"
    assert ctx.workspace_id == "ws"
    assert ctx.user_id == "u"
    assert ctx.user_slug == "s"
    assert ctx.user_name == "Name"
    assert ctx.principal_key == "pk"
    assert ctx.conversation_id == "conv"
    assert ctx.surface_id == "sid"
    assert ctx.memory_scope == "personal"
    assert ctx.session_plan_digest == "digest"


def test_build_minimal_context_uses_defaults() -> None:
    """_build_minimal_context uses sensible defaults when optional fields omitted."""
    ctx = _build_minimal_context(
        mode_id="default",
        query="test",
        resolve_model_fn=None,
        complete_fn=None,
        read_authored_text_fn=None,
        execute_tool_fn=None,
        execute_operation_fn=None,
    )
    assert ctx.workspace_id == "lisa-host"
    assert ctx.user_id == "lisa-user"
    assert ctx.user_slug == "lisa-user"
    assert ctx.user_name == "Lisa User"
    assert ctx.conversation_id == "lisa-host-console"
    assert ctx.surface_type == "personal_dm"
    assert ctx.surface_id == "lisa-host-console"
    assert ctx.memory_scope == "personal"


# ── Tool dispatch: operation branch ──────────────────────────────────


@pytest.mark.asyncio
async def test_operation_dispatch_via_hooks_execute_operation() -> None:
    """When the model calls an operation by name, it dispatches through
    hooks.execute_operation (not pool.call_tool)."""
    dispatched: list[str] = []

    async def stub_execute_op(
        operation_id: str, _ctx: OperationExecutionContext
    ) -> OperationResult:
        dispatched.append(operation_id)
        return OperationResult(output_text=f"Ran {operation_id}")

    async def two_step(**kwargs: Any) -> _StubCompletion:
        if len(dispatched) == 0:
            return _StubCompletion(
                content="",
                tool_calls=[_make_tool_call("c1", "web_search", {"q": "test"})],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step, execute_operation=stub_execute_op)
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Done."
    assert result.tool_call_count == 1
    assert dispatched == ["web_search"]


@pytest.mark.asyncio
async def test_operation_dispatch_error_returns_error_message() -> None:
    """When execute_operation raises, the tool output contains the error."""
    from kernel.operations.registry import OperationExecutionError

    async def failing_op(_id: str, _ctx: OperationExecutionContext) -> OperationResult:
        raise OperationExecutionError("Operation 'bad' failed: boom")

    async def two_step(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(
            content="",
            tool_calls=[_make_tool_call("c1", "web_search", {"q": "x"})],
        )

    # The second call should see the error in the tool result and recover.
    call_count = 0

    async def two_step_with_recovery(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_make_tool_call("c1", "web_search", {"q": "x"})],
            )
        return _StubCompletion(content="Recovered.")

    hooks = _build_hooks(two_step_with_recovery, execute_operation=failing_op)
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Recovered."


# ── Tool dispatch: MCP branch ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_tool_dispatch_via_pool_call_tool() -> None:
    """Unknown tool names (not in operation or tool catalog) dispatch via pool."""
    from kernel.tools.mcp_client import MCPToolResult

    pool_calls: list[str] = []

    class _StubPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> MCPToolResult:
            pool_calls.append(tool_id)
            return MCPToolResult(tool_id=tool_id, output_text="MCP result")

        async def __aenter__(self) -> _StubPool:
            return self

        async def __aexit__(self, *_a: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_make_tool_call("c1", "custom.mcp_tool", {})],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step, mcp_tool_pool=_StubPool())
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Done."
    assert pool_calls == ["custom.mcp_tool"]


@pytest.mark.asyncio
async def test_mcp_tool_error_continues_turn() -> None:
    """When pool.call_tool raises, the error is caught and the turn continues."""

    class _FailingPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> Any:
            raise RuntimeError("MCP server crashed")

        async def __aenter__(self) -> _FailingPool:
            return self

        async def __aexit__(self, *_a: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_make_tool_call("c1", "custom.tool", {})],
            )
        return _StubCompletion(content="Recovered from MCP error.")

    hooks = _build_hooks(two_step, mcp_tool_pool=_FailingPool())
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Recovered from MCP error."


# ── Tool dispatch: empty output coercion ──────────────────────────────


@pytest.mark.asyncio
async def test_empty_tool_output_coerced_to_done() -> None:
    """When a tool returns empty output, it's coerced to 'Done'."""
    from kernel.tools.mcp_client import MCPToolResult

    class _EmptyPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(tool_id=tool_id, output_text="")

        async def __aenter__(self) -> _EmptyPool:
            return self

        async def __aexit__(self, *_a: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    captured_tool_outputs: list[str] = []
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_make_tool_call("c1", "custom.tool", {})],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step, mcp_tool_pool=_EmptyPool())
    await run_turn(mode_id="default", query="test", hooks=hooks)

    assert "Done" in captured_tool_outputs


# ── Thought signature preservation (Gemini) ───────────────────────────


@pytest.mark.asyncio
async def test_thought_signature_preserved_in_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The thought_signature from a tool_call is preserved in the tool result message."""
    from kernel.tools.mcp_client import MCPToolResult

    class _StubPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(tool_id=tool_id, output_text="ok")

        async def __aenter__(self) -> _StubPool:
            return self

        async def __aexit__(self, *_a: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    captured_messages: list[dict[str, Any]] = []
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {"name": "custom.tool", "arguments": "{}"},
                        "thought_signature": "sig_abc123",
                    }
                ],
            )
        captured_messages.extend(kwargs.get("messages", []))
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step, mcp_tool_pool=_StubPool())
    await run_turn(mode_id="default", query="test", hooks=hooks)

    tool_msgs = [m for m in captured_messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1
    assert tool_msgs[0].get("thought_signature") == "sig_abc123"


@pytest.mark.asyncio
async def test_thought_signature_preserved_nested_in_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thought_signature nested under 'function' is also preserved."""
    from kernel.tools.mcp_client import MCPToolResult

    class _StubPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(tool_id=tool_id, output_text="ok")

        async def __aenter__(self) -> _StubPool:
            return self

        async def __aexit__(self, *_a: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    captured_messages: list[dict[str, Any]] = []
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "custom.tool",
                            "arguments": "{}",
                            "thought_signature": "nested_sig",
                        },
                    }
                ],
            )
        captured_messages.extend(kwargs.get("messages", []))
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step, mcp_tool_pool=_StubPool())
    await run_turn(mode_id="default", query="test", hooks=hooks)

    tool_msgs = [m for m in captured_messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1
    assert tool_msgs[0].get("thought_signature") == "nested_sig"


# ── _extract_completion_usage ────────────────────────────────────────


def test_extract_completion_usage_from_flat_attributes() -> None:
    """_extract_completion_usage reads flat prompt_tokens/completion_tokens."""
    from kernel.turn_runner import _extract_completion_usage

    completion = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cached_tokens=20,
        reasoning_tokens=5,
        duration_ms=123.4,
        latency_ms=200,
        usage=None,
    )
    result = _extract_completion_usage(completion)
    assert result[0] == 100  # prompt_tokens
    assert result[1] == 50  # completion_tokens
    assert result[2] == 150  # total_tokens
    assert result[3] == 20  # cached_tokens
    assert result[4] == 5  # reasoning_tokens
    assert result[5] == 123.4  # duration_ms
    assert result[6] == 200  # latency_ms


def test_extract_completion_usage_from_usage_dict() -> None:
    """_extract_completion_usage reads from a usage dict when flat attrs are 0."""
    from kernel.turn_runner import _extract_completion_usage

    completion = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cached_tokens=0,
        reasoning_tokens=0,
        duration_ms=0.0,
        latency_ms=0,
        usage={
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "total_tokens": 280,
            "cached_tokens": 10,
            "reasoning_tokens": 3,
        },
    )
    result = _extract_completion_usage(completion)
    assert result[0] == 200
    assert result[1] == 80
    assert result[2] == 280
    assert result[3] == 10
    assert result[4] == 3


def test_extract_completion_usage_computes_total_when_missing() -> None:
    """When total_tokens is 0, it's computed as prompt + completion."""
    from kernel.turn_runner import _extract_completion_usage

    completion = SimpleNamespace(
        prompt_tokens=50,
        completion_tokens=30,
        total_tokens=0,
        cached_tokens=0,
        reasoning_tokens=0,
        duration_ms=0.0,
        latency_ms=0,
        usage=None,
    )
    result = _extract_completion_usage(completion)
    assert result[2] == 80  # 50 + 30


def test_extract_completion_usage_from_usage_object() -> None:
    """_extract_completion_usage reads from a usage object (not dict)."""
    from kernel.turn_runner import _extract_completion_usage

    completion = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cached_tokens=0,
        reasoning_tokens=0,
        duration_ms=0.0,
        latency_ms=0,
        usage=SimpleNamespace(
            prompt_tokens=300,
            completion_tokens=100,
            total_tokens=400,
            cached_tokens=0,
            reasoning_tokens=0,
        ),
    )
    result = _extract_completion_usage(completion)
    assert result[0] == 300
    assert result[1] == 100
    assert result[2] == 400


def test_extract_completion_usage_handles_input_tokens_alias() -> None:
    """_extract_completion_usage reads input_tokens as an alias for prompt_tokens."""
    from kernel.turn_runner import _extract_completion_usage

    completion = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cached_tokens=0,
        reasoning_tokens=0,
        duration_ms=0.0,
        latency_ms=0,
        usage={"input_tokens": 150, "output_tokens": 60, "total_tokens": 210},
    )
    result = _extract_completion_usage(completion)
    assert result[0] == 150
    assert result[1] == 60
    assert result[2] == 210


def test_extract_completion_usage_defaults_to_zero() -> None:
    """All values default to 0 when nothing is present."""
    from kernel.turn_runner import _extract_completion_usage

    completion = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cached_tokens=0,
        reasoning_tokens=0,
        duration_ms=0.0,
        latency_ms=0,
        usage=None,
    )
    result = _extract_completion_usage(completion)
    assert all(v == 0 or v == 0.0 for v in result)


# ── StreamedCompletionAdapter ─────────────────────────────────────────


def test_streamed_completion_adapter_bridges_content_and_tool_calls() -> None:
    """StreamedCompletionAdapter exposes content, tool_calls, and usage."""
    from kernel.provider_contracts import StreamedCompletion
    from kernel.turn_runner import StreamedCompletionAdapter

    aggregated = StreamedCompletion(
        content="Hello",
        tool_calls=[],
        usage={},
        finish_reason="stop",
    )
    adapter = StreamedCompletionAdapter(aggregated)
    assert adapter.content == "Hello"
    assert adapter.tool_calls == []
    assert adapter.usage == {}


def test_streamed_completion_adapter_repr() -> None:
    """The repr includes content and finish_reason for debugging."""
    from kernel.provider_contracts import StreamedCompletion
    from kernel.turn_runner import StreamedCompletionAdapter

    aggregated = StreamedCompletion(
        content="Hi",
        tool_calls=[],
        usage={},
        finish_reason="stop",
    )
    adapter = StreamedCompletionAdapter(aggregated)
    repr_str = repr(adapter)
    assert "Hi" in repr_str
    assert "stop" in repr_str


# ── CheckpointTuning ──────────────────────────────────────────────────


def test_checkpoint_tuning_defaults_disabled() -> None:
    """CheckpointTuning defaults to enabled=False."""
    from kernel.turn_runner import CheckpointTuning

    tuning = CheckpointTuning()
    assert tuning.enabled is False
    assert tuning.on_stage_transition is True
    assert tuning.every_n_turns == 0
    assert tuning.explicit is True


def test_checkpoint_tuning_from_mapping() -> None:
    """CheckpointTuning.from_mapping builds from a config dict."""
    from kernel.turn_runner import CheckpointTuning

    tuning = CheckpointTuning.from_mapping(
        {
            "enabled": True,
            "on_stage_transition": False,
            "every_n_turns": 5,
            "explicit": False,
        }
    )
    assert tuning.enabled is True
    assert tuning.on_stage_transition is False
    assert tuning.every_n_turns == 5
    assert tuning.explicit is False


def test_checkpoint_tuning_from_mapping_none_returns_defaults() -> None:
    """CheckpointTuning.from_mapping(None) returns defaults."""
    from kernel.turn_runner import CheckpointTuning

    tuning = CheckpointTuning.from_mapping(None)
    assert tuning.enabled is False


def test_checkpoint_tuning_from_mapping_ignores_unknown_keys() -> None:
    """Unknown keys in the mapping are ignored."""
    from kernel.turn_runner import CheckpointTuning

    tuning = CheckpointTuning.from_mapping({"enabled": True, "unknown_key": 42})
    assert tuning.enabled is True


# ── _should_checkpoint ─────────────────────────────────────────────────


def test_should_checkpoint_false_when_disabled() -> None:
    """_should_checkpoint returns False when tuning is None or disabled."""
    from kernel.turn_runner import CheckpointTuning, TurnHooks, _should_checkpoint

    hooks = TurnHooks()
    assert (
        _should_checkpoint(
            tuning=None,
            transitioned_to="reason",
            hooks=hooks,
        )
        is False
    )

    tuning = CheckpointTuning(enabled=False)
    assert (
        _should_checkpoint(
            tuning=tuning,
            transitioned_to="reason",
            hooks=hooks,
        )
        is False
    )


def test_should_checkpoint_true_on_stage_transition() -> None:
    """_should_checkpoint returns True when a transition occurred and
    on_stage_transition is enabled."""
    from kernel.turn_runner import CheckpointTuning, TurnHooks, _should_checkpoint

    hooks = TurnHooks()
    tuning = CheckpointTuning(enabled=True, on_stage_transition=True)
    assert (
        _should_checkpoint(
            tuning=tuning,
            transitioned_to="reason",
            hooks=hooks,
        )
        is True
    )


def test_should_checkpoint_false_on_transition_when_disabled() -> None:
    """_should_checkpoint returns False when on_stage_transition is False."""
    from kernel.turn_runner import CheckpointTuning, TurnHooks, _should_checkpoint

    hooks = TurnHooks()
    tuning = CheckpointTuning(enabled=True, on_stage_transition=False)
    assert (
        _should_checkpoint(
            tuning=tuning,
            transitioned_to="reason",
            hooks=hooks,
        )
        is False
    )


def test_should_checkpoint_true_on_every_n_turns() -> None:
    """_should_checkpoint returns True when turn_count is a multiple of every_n_turns."""
    from kernel.workflow_position import WorkflowPosition
    from kernel.turn_runner import CheckpointTuning, TurnHooks, _should_checkpoint

    session = WorkflowPosition(
        mode_id="ponder",
        stage_id="reason",
        entered_at="2026-01-01T00:00:00Z",
        turn_count=10,
    )
    hooks = TurnHooks(workflow_position=lambda: session)
    tuning = CheckpointTuning(enabled=True, every_n_turns=5, on_stage_transition=False)
    assert (
        _should_checkpoint(
            tuning=tuning,
            transitioned_to=None,
            hooks=hooks,
        )
        is True
    )


def test_should_checkpoint_false_when_turn_count_not_multiple() -> None:
    """_should_checkpoint returns False when turn_count is not a multiple."""
    from kernel.workflow_position import WorkflowPosition
    from kernel.turn_runner import CheckpointTuning, TurnHooks, _should_checkpoint

    session = WorkflowPosition(
        mode_id="ponder",
        stage_id="reason",
        entered_at="2026-01-01T00:00:00Z",
        turn_count=7,
    )
    hooks = TurnHooks(workflow_position=lambda: session)
    tuning = CheckpointTuning(enabled=True, every_n_turns=5, on_stage_transition=False)
    assert (
        _should_checkpoint(
            tuning=tuning,
            transitioned_to=None,
            hooks=hooks,
        )
        is False
    )


def test_should_checkpoint_false_when_not_workflow_for_n_turns() -> None:
    """every_n_turns requires a wired workflow position (Phase 6c: no mode-kind gate)."""
    from kernel.turn_runner import CheckpointTuning, TurnHooks, _should_checkpoint

    hooks = TurnHooks()
    tuning = CheckpointTuning(enabled=True, every_n_turns=5, on_stage_transition=False)
    assert (
        _should_checkpoint(
            tuning=tuning,
            transitioned_to=None,
            hooks=hooks,
        )
        is False
    )
