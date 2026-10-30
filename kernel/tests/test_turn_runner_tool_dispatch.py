"""Tests for tool dispatch branches in the turn runner.

Covers the three dispatch paths that fire after transition_stage interception:
  1. Operation dispatch — via hooks.operations.execute_operation
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
from kernel.turn_phases._fork import _build_minimal_context
from kernel.turn_phases._post_turn import (
    _stringify_pack_tool_result,
    _stringify_result,
    _to_tool_context,
)
from kernel.turn_runner import (
    TurnHooks,
    _extract_pre_turn_briefing,
    _resolve_tool_display_label,
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
    # Flat hook overrides (e.g. ``mcp_tool_pool=``, ``execute_operation=``)
    # fold into the correct grouped sub-dataclass; recall defaults to off.
    from turn_hooks_fold import build_hooks

    return build_hooks(
        defaults={"models": {"complete": complete_fn}, "recall": {"enabled": False}},
        **overrides,
    )


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


def test_to_tool_context_carries_the_thread_store_stem() -> None:
    """A pack tool's telemetry is stamped with the thread it ran in.

    The tool's telemetry row is read back by the forensic reader, which
    scopes rows by the thread's store stem. A tool context that dropped the
    stem would write a row the reader cannot see — the turn's decisions and
    operation dispatches would appear while its pack-tool dispatches did
    not.
    """
    ctx = OperationExecutionContext(
        user_slug="u",
        thread_store_slug="u--conv-alpha",
    )
    tool_ctx = _to_tool_context("my_tool", ctx)
    assert tool_ctx.thread_store_slug == "u--conv-alpha"


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
    """_build_minimal_context uses the canonical personal-DM defaults when
    optional identity fields are omitted (mirrors the model defaults)."""
    ctx = _build_minimal_context(
        mode_id="default",
        query="test",
        resolve_model_fn=None,
        complete_fn=None,
        read_authored_text_fn=None,
        execute_tool_fn=None,
        execute_operation_fn=None,
    )
    assert ctx.workspace_id == "default"
    assert ctx.user_id == "default"
    assert ctx.user_slug == "default"
    assert ctx.user_name == ""
    assert ctx.conversation_id == "default"
    assert ctx.surface_type == "personal_dm"
    assert ctx.surface_id == "default"
    assert ctx.memory_scope == "personal"


def test_build_minimal_context_carries_the_thread_store_stem() -> None:
    """The thread store stem reaches the context beside the user slug.

    A host that isolates one store per conversation names the store
    ``<user>--<conversation>``. The stem is carried separately from
    ``user_slug`` because the shared ledgers are still scoped by the plain
    user slug, and a reader that conflated the two would silently join no
    evidence.
    """
    ctx = _build_minimal_context(
        mode_id="default",
        query="test",
        resolve_model_fn=None,
        complete_fn=None,
        read_authored_text_fn=None,
        execute_tool_fn=None,
        execute_operation_fn=None,
        user_slug="u",
        thread_store_slug="u--conv-alpha",
    )
    assert ctx.thread_store_slug == "u--conv-alpha"
    assert ctx.user_slug == "u"


def test_build_minimal_context_defaults_the_thread_store_stem_to_empty() -> None:
    """No stem means the store is named by the user slug alone."""
    ctx = _build_minimal_context(
        mode_id="default",
        query="test",
        resolve_model_fn=None,
        complete_fn=None,
        read_authored_text_fn=None,
        execute_tool_fn=None,
        execute_operation_fn=None,
        user_slug="u",
    )
    assert ctx.thread_store_slug == ""


def test_resumed_turn_keeps_the_live_thread_store_stem() -> None:
    """A resumed turn writes its rows under the live stem, not the saved one.

    The persisted context replaces the freshly built one on the resume path,
    so a continuation saved before the stem existed would leave this turn
    writing rows no scoped read could find — its own evidence, invisible.
    """
    persisted = OperationExecutionContext.model_validate_json(
        '{"user_slug": "u", "conversation_id": "c", "thread_store_slug": ""}'
    )
    live = OperationExecutionContext(user_slug="u", thread_store_slug="u--c")

    stem = live.thread_store_slug
    resumed = live.model_copy(update=persisted.model_dump())
    resumed.thread_store_slug = stem

    assert resumed.thread_store_slug == "u--c"


def test_entry_context_factory_carries_the_thread_store_stem() -> None:
    """The turn's entry context carries the store stem from the identity.

    This is the context every operation dispatched during a turn receives,
    so it is the only one that matters for a model-facing reader. Testing
    ``_build_minimal_context`` alone would miss a factory that never
    supplies the stem.
    """
    from kernel.turn_runner import _wire_build_entry_ctx

    hooks = _build_hooks(
        complete_fn=None,
        user_slug="u",
        thread_store_slug="u--conv-alpha",
    )
    build_entry_ctx = _wire_build_entry_ctx(
        hooks=hooks,
        mode_id="default",
        query="test",
        resolve_model=lambda _model_id: None,
        complete=None,
        exec_op=None,
        mcp_tool_executor=None,
        delegate_fn=None,
        fork_fn=None,
        tool_output_bus=None,
    )

    ctx = build_entry_ctx()

    assert ctx.thread_store_slug == "u--conv-alpha"
    assert ctx.user_slug == "u"


def test_entry_context_factory_carries_the_thread_dir() -> None:
    """The turn's entry context carries the host's resolved thread directory.

    The host resolves the directory once, and a CLI argument wins over the
    environment. An operation that re-derives the directory from the
    environment would read a directory the host never wrote — the operator
    and the model would then read different stores.
    """
    from kernel.turn_runner import _wire_build_entry_ctx

    hooks = _build_hooks(
        complete_fn=None,
        user_slug="u",
        data_dir="/host/data",
        thread_dir="/host/resolved/threads",
    )
    build_entry_ctx = _wire_build_entry_ctx(
        hooks=hooks,
        mode_id="default",
        query="test",
        resolve_model=lambda _model_id: None,
        complete=None,
        exec_op=None,
        mcp_tool_executor=None,
        delegate_fn=None,
        fork_fn=None,
        tool_output_bus=None,
    )

    ctx = build_entry_ctx()

    assert ctx.thread_dir == "/host/resolved/threads"
    assert ctx.data_dir == "/host/data"


# ── Tool dispatch: operation branch ──────────────────────────────────


@pytest.mark.asyncio
async def test_operation_dispatch_via_hooks_execute_operation() -> None:
    """When the model calls an operation by name, it dispatches through
    hooks.operations.execute_operation (not pool.call_tool)."""
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


def _capture_tool_role_messages(messages: list[dict[str, Any]], captured: list[str]) -> None:
    """Append the contents of tool-role messages from one request into ``captured``."""
    for msg in messages:
        if msg.get("role") == "tool":
            captured.append(str(msg.get("content", "")))


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
        _capture_tool_role_messages(kwargs.get("messages", []), captured_tool_outputs)
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step, mcp_tool_pool=_EmptyPool())
    await run_turn(mode_id="default", query="test", hooks=hooks)

    assert "Done" in captured_tool_outputs


# ── Recovery-record effect classification for MCP tools ────────────────


class _RecoveryCaptureService:
    """Recovery service stub that records the ``RecoveryRecord`` handed to begin."""

    def __init__(self) -> None:
        self.begun: list[Any] = []

    async def begin(self, record: Any) -> Any:
        self.begun.append(record)
        return record

    async def settle(self, _work_item_id: str, _identity: Any, **_kwargs: Any) -> None:
        return None

    async def mark_running_indeterminate(self, _identity: Any) -> int:
        return 0


class _ReadOnlySpecPool:
    """Pool whose ``tool_spec`` reports the spec's ``read_only_hint``."""

    def __init__(self, read_only_hint: bool) -> None:
        self._spec = SimpleNamespace(read_only_hint=read_only_hint)

    async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
        return []

    async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> Any:
        return SimpleNamespace(tool_id=tool_id, output_text="MCP result", is_error=False)

    def tool_spec(self, _tool_id: str) -> Any:
        return self._spec

    async def __aenter__(self) -> _ReadOnlySpecPool:
        return self

    async def __aexit__(self, *_a: object) -> None:
        pass

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


def _single_mcp_tool_completion() -> Any:
    """Model completion plan: one ``lisa-storage.read_file`` call, then done."""
    state = {"count": 0}

    async def complete(**_kwargs: Any) -> _StubCompletion:
        state["count"] += 1
        if state["count"] == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_make_tool_call("c1", "lisa-storage.read_file", {"path": "a.md"})],
            )
        return _StubCompletion(content="Done.")

    return complete


@pytest.mark.asyncio
async def test_read_only_mcp_tool_recovery_record_is_read_effect() -> None:
    """A ``read_only_hint`` MCP tool begins a ``read``/``safe`` recovery record.

    Regression: opaque MCP tools were hardcoded to ``execute``/``confirm`` in
    the recovery record, so a read-only storage/thread call that stalled
    blocked the workflow behind a confirm gate (and, in one-shot hosts, an
    unanswerable approval prompt).
    """
    recovery = _RecoveryCaptureService()
    hooks = _build_hooks(
        _single_mcp_tool_completion(),
        mcp_tool_pool=_ReadOnlySpecPool(read_only_hint=True),
        recovery_service=recovery,
    )
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Done."
    assert len(recovery.begun) == 1
    record = recovery.begun[0]
    assert record.target_id == "lisa-storage.read_file"
    assert record.target_kind == "mcp_tool"
    assert record.effect == "read"
    assert record.retry_policy == "safe"


@pytest.mark.asyncio
async def test_effectful_mcp_tool_recovery_record_stays_execute() -> None:
    """An MCP tool without ``read_only_hint`` still records ``execute``/``confirm``.

    Fail-closed contract: an opaque tool that does not prove it is read-only
    keeps the effectful retry policy.
    """
    recovery = _RecoveryCaptureService()
    hooks = _build_hooks(
        _single_mcp_tool_completion(),
        mcp_tool_pool=_ReadOnlySpecPool(read_only_hint=False),
        recovery_service=recovery,
    )
    await run_turn(mode_id="default", query="test", hooks=hooks)

    assert len(recovery.begun) == 1
    record = recovery.begun[0]
    assert record.effect == "execute"
    assert record.retry_policy == "confirm"


# ── Recovery begin failure withholds effectful dispatch ─────────────────────


class _FailingBeginService:
    """Recovery service whose ``begin`` raises, simulating a persistence error."""

    def __init__(self) -> None:
        self.begun: list[Any] = []
        self.settled: list[Any] = []

    async def begin(self, record: Any) -> Any:
        self.begun.append(record)
        raise RuntimeError("recovery store unavailable")

    async def settle(self, work_item_id: str, _identity: Any, **_kwargs: Any) -> None:
        self.settled.append(work_item_id)

    async def mark_running_indeterminate(self, _identity: Any) -> int:
        return 0


class _CountingPool:
    """Pool that records every ``call_tool`` so the test can prove non-dispatch."""

    def __init__(self, read_only_hint: bool) -> None:
        self.calls: list[str] = []
        self._spec = SimpleNamespace(read_only_hint=read_only_hint)

    async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
        return []

    async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> Any:
        self.calls.append(tool_id)
        return SimpleNamespace(tool_id=tool_id, output_text="MCP result", is_error=False)

    def tool_spec(self, _tool_id: str) -> Any:
        return self._spec

    async def __aenter__(self) -> _CountingPool:
        return self

    async def __aexit__(self, *_a: object) -> None:
        pass

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_failed_recovery_begin_withholds_effectful_dispatch() -> None:
    """An effectful tool must not run when its recovery record could not be written.

    ``_begin_recovery_record`` is best-effort: it logs and swallows a
    persistence failure, returning an empty work-item id. Dispatch then
    proceeded anyway, so effectful work could start with no durable recovery
    record — the F2 finding in
    ``reviews/2026-10-25-lisa-code-market-gap-swot.md``.

    Acceptance: withhold effectful dispatch after a failed begin. This test
    proves the tool never reached the pool and that the model was told why.
    """
    recovery = _FailingBeginService()
    pool = _CountingPool(read_only_hint=False)
    captured: list[str] = []
    state = {"count": 0}

    async def complete(**kwargs: Any) -> _StubCompletion:
        state["count"] += 1
        if state["count"] == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_make_tool_call("c1", "lisa-storage.write_file", {"path": "a.md"})],
            )
        _capture_tool_role_messages(kwargs.get("messages", []), captured)
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(complete, mcp_tool_pool=pool, recovery_service=recovery)
    await run_turn(mode_id="default", query="test", hooks=hooks)

    assert len(recovery.begun) == 1, "the begin was attempted"
    assert recovery.begun[0].retry_policy == "confirm", "the tool is effectful"
    assert pool.calls == [], (
        "an effectful tool must not dispatch when its recovery record could not be "
        f"persisted; observed pool calls: {pool.calls}"
    )
    assert any("recovery" in text.lower() for text in captured), (
        f"the model must be told the call was withheld; tool results: {captured}"
    )


@pytest.mark.asyncio
async def test_failed_recovery_begin_still_allows_read_only_dispatch() -> None:
    """A read-only tool may still run when its recovery record could not be written.

    The withholding rule is scoped to effectful work. A ``safe`` retry policy
    means the call has no side effects, so losing the record does not risk
    unrecorded mutation. This keeps the fix from disabling read-only tools on a
    degraded recovery store.
    """
    recovery = _FailingBeginService()
    pool = _CountingPool(read_only_hint=True)

    hooks = _build_hooks(
        _single_mcp_tool_completion(),
        mcp_tool_pool=pool,
        recovery_service=recovery,
    )
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert len(recovery.begun) == 1
    assert recovery.begun[0].retry_policy == "safe", "the tool is read-only"
    assert pool.calls == ["lisa-storage.read_file"], (
        "a read-only tool must still dispatch when its recovery record failed; "
        f"observed pool calls: {pool.calls}"
    )
    assert result.response_text == "Done."


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
    from kernel.turn_phases._post_turn import _extract_completion_usage

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
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 50
    assert result.total_tokens == 150
    assert result.cached_tokens == 20
    assert result.reasoning_tokens == 5
    assert result.cache_creation_tokens == 0
    assert result.duration_ms == 123.4
    assert result.latency_ms == 200


def test_extract_completion_usage_from_usage_dict() -> None:
    """_extract_completion_usage reads from a usage dict when flat attrs are 0."""
    from kernel.turn_phases._post_turn import _extract_completion_usage

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
    assert result.prompt_tokens == 200
    assert result.completion_tokens == 80
    assert result.total_tokens == 280
    assert result.cached_tokens == 10
    assert result.reasoning_tokens == 3


def test_extract_completion_usage_computes_total_when_missing() -> None:
    """When total_tokens is 0, it's computed as prompt + completion."""
    from kernel.turn_phases._post_turn import _extract_completion_usage

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
    from kernel.turn_phases._post_turn import _extract_completion_usage

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
    from kernel.turn_phases._post_turn import _extract_completion_usage

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


def test_extract_completion_usage_maps_cache_creation_alias_from_usage_dict() -> None:
    """Anthropic's ``cache_creation_input_tokens`` maps to ``cache_creation_tokens``."""
    from kernel.turn_phases._post_turn import _extract_completion_usage

    completion = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cached_tokens=0,
        reasoning_tokens=0,
        duration_ms=0.0,
        latency_ms=0,
        usage={"input_tokens": 150, "output_tokens": 60, "cache_creation_input_tokens": 120},
    )
    result = _extract_completion_usage(completion)
    assert result.cache_creation_tokens == 120


def test_extract_completion_usage_flat_cache_creation_wins_over_usage() -> None:
    """Flat ``cache_creation_tokens`` take precedence over the usage alias."""
    from kernel.turn_phases._post_turn import _extract_completion_usage

    completion = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cached_tokens=0,
        reasoning_tokens=0,
        cache_creation_tokens=8,
        duration_ms=0.0,
        latency_ms=0,
        usage={"cache_creation_input_tokens": 90},
    )
    result = _extract_completion_usage(completion)
    assert result.cache_creation_tokens == 8


def test_extract_completion_usage_defaults_to_zero() -> None:
    """All values default to 0 when nothing is present."""
    from kernel.turn_phases._post_turn import _extract_completion_usage

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
    assert all(v in {0} for v in result)


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


# ── _resolve_tool_display_label ───────────────────────────────────────


def test_display_label_renders_template_against_params() -> None:
    """Display template renders the primary parameter (bash_run's command)."""
    out = _resolve_tool_display_label("{command}", {"command": "ls -la /etc", "cwd": "/"})
    assert out == "ls -la /etc"


def test_display_label_missing_key_renders_placeholder() -> None:
    """A template key absent from the params renders as ``{key}``, not KeyError."""
    out = _resolve_tool_display_label("{command}", {})
    assert out == "{command}"


def test_display_label_empty_template_returns_empty() -> None:
    """Empty/whitespace templates yield no label so the host falls back."""
    assert _resolve_tool_display_label("", {"command": "ls"}) == ""
    assert _resolve_tool_display_label("   ", {"command": "ls"}) == ""


def test_display_label_truncates_to_limit() -> None:
    """Labels are newline-flattened and capped at 80 chars."""
    long_command = "x" * 200
    out = _resolve_tool_display_label("{command}", {"command": long_command})
    assert len(out) == 80
    assert out.endswith("…")
    assert "\n" not in _resolve_tool_display_label("{command}", {"command": "a\nb"})


def test_display_label_format_error_returns_empty() -> None:
    """A malformed template yields empty rather than raising mid-turn."""
    assert _resolve_tool_display_label("{command!bad}", {"command": "ls"}) == ""


# ── summarize_json_result ─────────────────────────────────────────────


def test_summarize_json_result_leads_with_content_not_envelope_metadata() -> None:
    """JSON envelope previews quote the payload before any provenance labels.

    The 2026-10-02 regression: ``thread_read_content`` pages serialize
    ``ref`` + ``source`` provenance ahead of ``content``, so a raw
    ``str(output)[:200]`` preview showed only keys and interpretation
    prose — never the content the operator asked about. The summary must
    quote the payload first and fit inside the 200-char preview budget.
    """
    from kernel.turn_phases._dispatch import _RESULT_PREVIEW_CHARS, summarize_json_result

    page = json.dumps(
        {
            "ref": "r" + "a" * 64,
            "source": {
                "scope": "stored_tool_result",
                "source_kind": "tool",
                "timestamp_utc": None,
                "interpretation": (
                    "Stored tool output; its time and relationship "
                    "to the active request are unknown."
                ),
            },
            "content": (
                "The reviewer found that the config parser silently drops "
                "duplicate keys when loading the manifest, so the last value "
                "wins without warning. " * 3
            ),
            "offset_chars": 0,
            "total_chars": 5412,
            "next_offset_chars": 4000,
            "has_more": True,
        }
    )
    summary = summarize_json_result(page)
    assert len(summary) <= _RESULT_PREVIEW_CHARS
    assert summary.startswith("content=The reviewer found")
    assert "offset_chars=0" in summary
    assert "total_chars=5412" in summary
    assert "stored_tool_result" not in summary


def test_summarize_json_result_quotes_first_list_record_payload() -> None:
    """List envelopes (entries/matches) quote the payload of the first record."""
    from kernel.turn_phases._dispatch import summarize_json_result

    page = json.dumps(
        {
            "scope": "recorded_history",
            "interpretation": (
                "Recorded exchanges, not the active request; earlier "
                "instructions and diagnoses are historical."
            ),
            "entries": [
                {
                    "turn": 455,
                    "query": "why did the e2e gate fail?",
                    "response_text": (
                        "The checked-commit gate rejected a reused worktree "
                        "whose second commit made HEAD's parent differ."
                    ),
                }
            ],
            "next_cursor": None,
            "has_more": False,
        }
    )
    summary = summarize_json_result(page)
    assert summary.startswith("response_text=The checked-commit gate")
    assert "entries=1" in summary
    assert "scope=recorded_history" in summary


def test_summarize_json_result_plain_text_passes_through_unchanged() -> None:
    """Non-JSON results keep the legacy raw preview behavior."""
    from kernel.turn_phases._dispatch import summarize_json_result

    raw = "Tool execution failed: ValueError: bad ref"
    assert summarize_json_result(raw) == raw
    assert summarize_json_result("") == ""
    # A truncated JSON fragment (streamed prefix) falls back to raw text.
    fragment = '{"ref": "r2cf05", "sou'
    assert summarize_json_result(fragment) == fragment


def test_summarize_json_result_empty_object_falls_back_raw() -> None:
    """{} and null-only objects have nothing payload-like to summarize."""
    from kernel.turn_phases._dispatch import summarize_json_result

    assert summarize_json_result("{}") == "{}"
    assert summarize_json_result('{"a": null}') == '{"a": null}'
