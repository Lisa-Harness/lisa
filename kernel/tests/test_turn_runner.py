"""Tests for kernel/turn_runner.py — the shared model+tool loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kernel.models import (
    CapabilityManifest,
    OperationExecutionContext,
    OperationResult,
)
from kernel.operations.registry import (
    execute_operation,
    get_operation_catalog,
)
from kernel.turn_runner import TurnHooks, TurnResult, run_turn

# ── Module-level setup ────────────────────────────────────────────
_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "core").resolve()

# The shared conftest fixture (_mcp_catalog_setup) handles all catalog wiring.


# ── Test helpers ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    """Fake model completion for tests."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_stub_complete(response: str, tool_calls: list[dict[str, Any]] | None = None):
    """Return an async complete stub that returns the given response."""

    async def stub(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content=response, tool_calls=tool_calls)

    return stub


async def _stub_execute_operation(
    operation_id: str, _ctx: OperationExecutionContext
) -> OperationResult:
    return OperationResult(output_text=f"Executed {operation_id}")


def _build_minimal_hooks(complete_fn) -> TurnHooks:
    return TurnHooks(complete=complete_fn)


# ── Core flow tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_requires_complete_hook() -> None:
    """run_turn raises NotImplementedError when hooks.complete is None."""
    hooks = TurnHooks()  # no complete

    with pytest.raises(NotImplementedError, match="complete"):
        await run_turn(mode_id="default", query="hello", hooks=hooks)


@pytest.mark.asyncio
async def test_run_turn_simple_chat_no_tools() -> None:
    """Basic chat: model returns text immediately, no tool calls."""
    hooks = _build_minimal_hooks(_make_stub_complete("Hello, world!"))

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert result.response_text == "Hello, world!"
    assert result.tool_call_count == 0
    assert result.iteration_count == 1


@pytest.mark.asyncio
async def test_run_turn_simple_chat_default_mode() -> None:
    """Chat through the default mode (no pre-turn ops) produces a TurnResult."""
    hooks = _build_minimal_hooks(_make_stub_complete("Response from default mode."))

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert result.response_text == "Response from default mode."
    assert result.tool_call_count == 0
    assert result.iteration_count == 1


@pytest.mark.asyncio
async def test_run_turn_missing_complete_raises_not_implemented() -> None:
    """A search (action) mode with entry ops needs complete_fn, and the
    operation fails with an OperationExecutionError wrapping the
    NotImplementedError when hooks.complete is missing."""
    from kernel.operations.registry import OperationExecutionError

    hooks = TurnHooks()  # no complete

    with pytest.raises(OperationExecutionError, match="web_search"):
        await run_turn(mode_id="search", query="python asyncio", hooks=hooks)


@pytest.mark.asyncio
async def test_run_turn_tool_loop_single_tool_call() -> None:
    """Model returns one tool call, then text — one tool loop iteration.

    Uses the 'default' mode which has no pre-turn operations, so the
    stub complete function's two-step logic works predictably.
    """
    call_counts = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        call_counts.append(1)
        if len(call_counts) == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {"name": "web_search", "arguments": '{"q": "test"}'},
                    }
                ],
            )
        return _StubCompletion(content="Here is your search result.")

    hooks = _build_minimal_hooks(two_step)
    hooks.execute_operation = _stub_execute_operation

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Here is your search result."
    assert result.tool_call_count == 1
    assert result.iteration_count == 2


@pytest.mark.asyncio
async def test_run_turn_max_tool_iterations_capped() -> None:
    """Tool loop capped at max_tool_iterations."""
    call_counts = []

    async def infinite_tools(**kwargs: Any) -> _StubCompletion:
        call_counts.append(1)
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"call_{len(call_counts)}",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"q": "test"}',
                    },
                }
            ],
        )

    hooks = _build_minimal_hooks(infinite_tools)
    hooks.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        max_tool_iterations=3,
    )

    assert result.iteration_count == 3
    assert result.tool_call_count == 3
    assert "did not produce a final response" in result.response_text


@pytest.mark.asyncio
async def test_run_turn_calls_on_turn_complete() -> None:
    """on_turn_complete is called after a successful turn."""
    side_effects: list[dict[str, Any]] = []

    async def on_complete(
        mode_id: str, query: str, response_text: str, messages: list[dict[str, Any]]
    ) -> None:
        side_effects.append(
            {
                "mode_id": mode_id,
                "query": query,
                "response_text": response_text,
                "message_count": len(messages),
            }
        )

    hooks = _build_minimal_hooks(_make_stub_complete("Done."))
    hooks.on_turn_complete = on_complete

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert len(side_effects) == 1
    assert side_effects[0]["mode_id"] == "default"
    assert side_effects[0]["query"] == "hello"
    assert side_effects[0]["response_text"] == "Done."
    assert side_effects[0]["message_count"] >= 2  # system + user + assistant


@pytest.mark.asyncio
async def test_run_turn_omits_empty_structured_pre_turn_briefing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured pre-turn ops with empty output_text should not inject JSON fallback."""

    captured_messages: list[dict[str, Any]] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        captured_messages.extend(kwargs["messages"])
        return _StubCompletion(content="Done.")

    async def pre_turn_result(
        mode_id: str,
        ctx: OperationExecutionContext,
        ledger: Any = None,
    ) -> OperationResult:
        _ = (mode_id, ctx, ledger)
        return OperationResult(
            output_text='{"briefing_points": [], "output_text": "", "source_kind": "", "source_urls": []}',
            output_json={
                "briefing_points": [],
                "output_text": "",
                "source_kind": "",
                "source_urls": [],
            },
        )

    monkeypatch.setattr("kernel.turn_runner.run_mode_pre_turn_operations", pre_turn_result)

    hooks = _build_minimal_hooks(complete)
    result = await run_turn(mode_id="discuss", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert len(captured_messages) >= 2
    user_message = str(captured_messages[1]["content"])
    assert "## Pre-turn Briefing" not in user_message
    assert "briefing_points" not in user_message


@pytest.mark.asyncio
async def test_run_turn_includes_non_empty_structured_pre_turn_briefing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured pre-turn ops still inject authored output_text when present."""

    captured_messages: list[dict[str, Any]] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        captured_messages.extend(kwargs["messages"])
        return _StubCompletion(content="Done.")

    async def pre_turn_result(
        mode_id: str,
        ctx: OperationExecutionContext,
        ledger: Any = None,
    ) -> OperationResult:
        _ = (mode_id, ctx, ledger)
        return OperationResult(
            output_text='{"output_text": "fallback json dump"}',
            output_json={
                "output_text": "Fresh web context for this discussion:\n- grounded point",
                "briefing_points": ["grounded point"],
            },
        )

    monkeypatch.setattr("kernel.turn_runner.run_mode_pre_turn_operations", pre_turn_result)

    hooks = _build_minimal_hooks(complete)
    result = await run_turn(mode_id="discuss", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert len(captured_messages) >= 2
    user_message = str(captured_messages[1]["content"])
    assert "## Pre-turn Briefing" in user_message
    assert "Fresh web context for this discussion:" in user_message
    assert "fallback json dump" not in user_message


@pytest.mark.asyncio
async def test_run_turn_prefers_explicit_model_override() -> None:
    """A host-supplied preferred model override beats the mode default."""

    resolved_model_ids: list[str] = []

    def resolve_model(model_id: str) -> Any:
        resolved_model_ids.append(model_id)
        return SimpleNamespace(model_id=model_id, reasoning_model=False)

    hooks = _build_minimal_hooks(_make_stub_complete("Done."))
    hooks.resolve_model = resolve_model

    result = await run_turn(
        mode_id="default",
        query="hello",
        hooks=hooks,
        preferred_model_id="haiku",
    )

    assert result.response_text == "Done."
    assert resolved_model_ids == ["haiku"]


@pytest.mark.asyncio
async def test_run_turn_injects_workflow_stage_instruction() -> None:
    """Workflow modes inject the current stage's instruction_body into the
    system prompt via the assemble_system_prompt hook's stage_instruction arg.

    Without this, loaded stage instructions (inquire.md, reason.md, etc.)
    are never injected and the FSM is dead at runtime.
    """
    from kernel.ipsar_fsm import get_workflow_catalog

    captured_stage_instructions: list[str | None] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        captured_messages = kwargs["messages"]
        return _StubCompletion(content="Done.")

    def assemble_system(mode_instruction: str, stage_instruction: str | None) -> str:
        captured_stage_instructions.append(stage_instruction)
        parts = [mode_instruction]
        if stage_instruction:
            parts.append(stage_instruction)
        return "\n\n".join(parts)

    def workflow_session() -> Any:
        from kernel.models import WorkflowSessionState

        mode = get_workflow_catalog().get("thinkabout")
        if mode is None:
            return None
        return WorkflowSessionState(
            user_id="test-user",
            user_slug="test",
            mode_id="thinkabout",
            stage_id="inquire",
            entered_at="2026-01-01T00:00:00Z",
        )

    hooks = _build_minimal_hooks(complete)
    hooks.assemble_system_prompt = assemble_system
    hooks.workflow_session = workflow_session

    result = await run_turn(mode_id="thinkabout", query="analyze this", hooks=hooks)

    assert result.response_text == "Done."
    assert len(captured_stage_instructions) >= 1
    stage_instruction = captured_stage_instructions[0]
    assert stage_instruction is not None
    assert "Inquire" in stage_instruction or "inquire" in stage_instruction.lower()


@pytest.mark.asyncio
async def test_run_turn_workflow_transition_dispatches_apply_transition() -> None:
    """When the model calls transition_stage, the loop intercepts it,
    calls apply_transition, updates the session via workflow_session_updater,
    and reports transitioned_to on the TurnResult.
    """
    from kernel.models import WorkflowSessionState

    updated_sessions: list[WorkflowSessionState] = []

    # Mutable session so workflow_session() always returns the latest state.
    session = WorkflowSessionState(
        user_id="test-user",
        user_slug="test",
        mode_id="thinkabout",
        stage_id="inquire",
        entered_at="2026-01-01T00:00:00Z",
    )

    def workflow_session() -> Any:
        return session

    def workflow_session_updater(updated: WorkflowSessionState) -> None:
        updated_sessions.append(updated)
        # Mutate in place so workflow_session() sees the new stage.
        session.stage_id = updated.stage_id
        session.turn_count = updated.turn_count
        session.stage_data = updated.stage_data

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
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "reason", "rationale": "enough evidence"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done reasoning.")

    hooks = _build_minimal_hooks(two_step)
    hooks.workflow_session = workflow_session
    hooks.workflow_session_updater = workflow_session_updater

    result = await run_turn(mode_id="thinkabout", query="analyze this", hooks=hooks)

    assert result.response_text == "Done reasoning."
    assert result.transitioned_to == "reason"
    assert len(updated_sessions) == 1
    assert updated_sessions[0].stage_id == "reason"
    assert updated_sessions[0].turn_count == 1


@pytest.mark.asyncio
async def test_run_turn_workflow_stuck_session_nudge_injects_message() -> None:
    """After _STAGE_NUDGE_THRESHOLD turns in the same stage, the loop
    injects a nudge user-message prompting the model to transition.
    """
    from kernel.models import WorkflowSessionState

    session = WorkflowSessionState(
        user_id="test-user",
        user_slug="test",
        mode_id="thinkabout",
        stage_id="inquire",
        entered_at="2026-01-01T00:00:00Z",
        turn_count=4,  # at the nudge threshold
    )

    def workflow_session() -> Any:
        return session

    captured_messages: list[dict[str, Any]] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        captured_messages.clear()
        captured_messages.extend(kwargs["messages"])
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow_session = workflow_session

    result = await run_turn(mode_id="thinkabout", query="analyze this", hooks=hooks)

    assert result.response_text == "Done."
    # The nudge message should be present in the messages sent to the model.
    nudge_found = any(
        "been in the" in str(msg.get("content", ""))
        for msg in captured_messages
        if msg.get("role") == "user"
    )
    assert nudge_found


@pytest.mark.asyncio
async def test_run_turn_mcp_tool_not_registered() -> None:
    """When a tool name is not on any MCP server, the pool returns an error
    and the turn continues."""
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
                            "name": "custom_tool_not_registered",
                            "arguments": "{}",
                        },
                    }
                ],
            )
        return _StubCompletion(content="Used custom tool.")

    hooks = _build_minimal_hooks(two_step)
    # No mcp_tool_servers configured — pool has no tools registered.

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Used custom tool."
    assert result.tool_call_count == 1


@pytest.mark.asyncio
async def test_default_hooks_are_noops() -> None:
    """TurnHooks defaults do not crash when invoked directly."""
    hooks = TurnHooks()

    # Default resolve_model returns a stub
    model = hooks.resolve_model("test") if hooks.resolve_model else None
    if hooks.resolve_model:
        assert model is not None

    # Default assemble_system_prompt
    if hooks.assemble_system_prompt:
        result = hooks.assemble_system_prompt("instruction", None)
        assert result == "instruction"

    # Default assemble_user_context returns empty (accepts optional manifest)
    if hooks.assemble_user_context:
        result = hooks.assemble_user_context("m", "q", "b", None)
        assert result == ""

    # Default mcp_tool_servers is an empty list
    assert hooks.mcp_tool_servers == []


@pytest.mark.asyncio
async def test_turn_result_struct() -> None:
    """TurnResult is a plain dataclass with expected fields."""
    result = TurnResult(response_text="ok", tool_call_count=3, iteration_count=2)

    assert result.response_text == "ok"
    assert result.tool_call_count == 3
    assert result.iteration_count == 2


@pytest.mark.asyncio
async def test_run_turn_passes_capability_manifest_to_assemble_user_context() -> None:
    """The turn runner compiles a CapabilityManifest from the live tool schemas
    and passes it to assemble_user_context. This closes the Tooling Paradox:
    the capability context now reflects exactly what is injected into the
    current turn's function schema.
    """
    from kernel.models import CapabilityManifest

    captured_manifests: list[CapabilityManifest | None] = []

    def assemble_user(
        _mode_id: str,
        _query: str,
        _pre_turn_briefing: str,
        capability_manifest: CapabilityManifest | None = None,
    ) -> str:
        captured_manifests.append(capability_manifest)
        return ""

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.assemble_user_context = assemble_user

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert len(captured_manifests) == 1
    manifest = captured_manifests[0]
    assert manifest is not None
    assert isinstance(manifest, CapabilityManifest)
    # The manifest should have at least one entry (operations from the catalog)
    assert len(manifest.entries) >= 1
    # Every entry should be marked available with confidence 1.0
    for entry in manifest.entries:
        assert entry.status == "available"
        assert entry.confidence == 1.0


# ── Unified FSM routing tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_mode_handoff_via_transition_stage() -> None:
    """When the model calls transition_stage with a mode_id target (not a
    workflow stage), the loop performs a mid-turn mode switch: swaps the
    system prompt, rebuilds the tool surface, and sets handoff_to on the
    TurnResult.
    """
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
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "markdown", "rationale": "user wants to write a doc"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done in markdown mode.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="write a markdown doc", hooks=hooks)

    assert result.response_text == "Done in markdown mode."
    assert result.handoff_to == "markdown"
    assert result.transitioned_to is None  # no workflow stage transition


@pytest.mark.asyncio
async def test_run_turn_fork_transition_basic() -> None:
    """When the model calls transition_stage with a fork: target, the loop
    fans out child turns in parallel and collects results into fork_results.
    """
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
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "fork:default,default"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Synthesized from fork results.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="research and write", hooks=hooks)

    assert result.response_text == "Synthesized from fork results."
    assert result.fork_results is not None
    assert len(result.fork_results) == 2
    for child in result.fork_results:
        assert child.status in ("ok", "error", "timeout", "cancelled")


@pytest.mark.asyncio
async def test_run_turn_emits_model_call_stage_events() -> None:
    """The on_stage hook fires model_call and model_call_complete around
    each model completion, carrying iteration and token metadata.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        events.append((label, dict(data)))

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.on_stage = on_stage

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    labels = [label for label, _ in events]
    assert "model_call" in labels
    assert "model_call_complete" in labels
    # model_call_complete should be the last event and carry token counts.
    last_label, last_data = events[-1]
    assert last_label == "model_call_complete"
    assert "iteration" in last_data
    assert "tool_calls" in last_data


@pytest.mark.asyncio
async def test_run_turn_emits_tool_dispatch_stage_events() -> None:
    """The on_stage hook fires tool_dispatch and tool_dispatch_complete
    around each operation/tool dispatch, with a tool_kind label.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        events.append((label, dict(data)))

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
                            "name": "web_search",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(two_step)
    hooks.execute_operation = _stub_execute_operation
    hooks.on_stage = on_stage

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Done."
    assert result.tool_call_count == 1
    dispatch_events = [e for e in events if e[0] == "tool_dispatch"]
    complete_events = [e for e in events if e[0] == "tool_dispatch_complete"]
    assert len(dispatch_events) == 1
    assert len(complete_events) == 1
    assert dispatch_events[0][1]["tool"] == "web_search"
    assert dispatch_events[0][1]["tool_kind"] == "operation"
    assert "duration_ms" in complete_events[0][1]


@pytest.mark.asyncio
async def test_run_turn_emits_fork_child_stage_events_per_child() -> None:
    """The on_stage hook fires fork_start, one fork_child_start and one
    fork_child_complete per child, then fork_complete — so a transport
    can render a line per active call.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        events.append((label, dict(data)))

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
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "fork:default,default"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Synthesized.")

    hooks = _build_minimal_hooks(two_step)
    hooks.on_stage = on_stage

    result = await run_turn(mode_id="default", query="fan out", hooks=hooks)

    assert result.fork_results is not None
    assert len(result.fork_results) == 2
    labels = [label for label, _ in events]
    assert labels.count("fork_start") == 1
    assert labels.count("fork_child_start") == 2
    assert labels.count("fork_child_complete") == 2
    assert "fork_complete" in labels
    # fork_start should carry the target list.
    start_event = next(e for e in events if e[0] == "fork_start")
    assert start_event[1]["children"] == 2
    assert len(start_event[1]["targets"]) == 2
    # fork_child_start events carry a child_index.
    child_starts = [e for e in events if e[0] == "fork_child_start"]
    child_indices = sorted(e[1]["child_index"] for e in child_starts)
    assert child_indices == [0, 1]
    # fork_complete carries ok/failed counts.
    complete_event = next(e for e in events if e[0] == "fork_complete")
    assert complete_event[1]["children"] == 2
    assert complete_event[1]["ok"] + complete_event[1]["failed"] == 2


@pytest.mark.asyncio
async def test_run_turn_on_stage_hook_failure_is_swallowed() -> None:
    """A failing on_stage hook is logged and swallowed; the turn continues."""
    call_count = 0

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        raise RuntimeError("transport exploded")

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.on_stage = on_stage

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    # The turn still completes despite the on_stage hook raising every call.
    assert result.response_text == "Done."


@pytest.mark.asyncio
async def test_routing_fsm_synthesis() -> None:
    """The routing FSM synthesis produces default ↔ all routable modes."""
    from kernel.ipsar_fsm import (
        get_routable_mode_ids,
        get_routing_targets,
        synthesize_routing_transitions,
    )

    transitions = synthesize_routing_transitions()
    # default should be able to hand off to every routable mode (except itself)
    default_targets = transitions.get("default", [])
    routable_ids = get_routable_mode_ids()
    for rid in routable_ids:
        if rid != "default":
            assert rid in default_targets, f"default should be able to hand off to {rid}"

    # Every routable mode should be able to hand off back to default
    for rid in routable_ids:
        if rid != "default":
            targets = get_routing_targets(rid)
            assert "default" in targets, f"{rid} should be able to hand off back to default"


def test_parse_fork_target_string() -> None:
    """The fork target parser handles mode_id and mode_id/stage_id syntax."""
    from kernel.ipsar_fsm import parse_fork_target_string

    # Simple mode_ids
    assert parse_fork_target_string("fork:coding,markdown") == [
        ("coding", ""),
        ("markdown", ""),
    ]
    # mode_id/stage_id pairs
    assert parse_fork_target_string("fork:coding/implement,thinkabout/review") == [
        ("coding", "implement"),
        ("thinkabout", "review"),
    ]
    # Mixed
    assert parse_fork_target_string("fork:search,coding/implement") == [
        ("search", ""),
        ("coding", "implement"),
    ]
    # Empty
    assert parse_fork_target_string("") == []
    assert parse_fork_target_string("fork:") == []


def test_is_fork_target() -> None:
    """The fork target detector recognizes fork: prefix."""
    from kernel.ipsar_fsm import is_fork_target

    assert is_fork_target("fork:coding,markdown") is True
    assert is_fork_target("FORK:coding") is True
    assert is_fork_target("coding") is False
    assert is_fork_target("reason") is False


@pytest.mark.asyncio
async def test_run_turn_mode_handoff_preserves_chat_history() -> None:
    """When the model hands off to another mode, the chat history (messages
    list) is preserved — only messages[0] (system prompt) is swapped.
    """
    call_count = 0
    captured_messages: list[list[dict[str, Any]]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        # Capture the messages list at each iteration
        msgs = kwargs.get("messages", [])
        captured_messages.append([dict(m) for m in msgs])
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "markdown"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="write a doc", hooks=hooks)

    assert result.handoff_to == "markdown"
    # After handoff, the model should have been called at least twice
    assert call_count >= 2
    # The first message (system prompt) should have changed after handoff
    # but the user message should be preserved
    if len(captured_messages) >= 2:
        # messages[0] is system, messages[1] is user
        # The user content should be the same across iterations
        first_user = captured_messages[0][1].get("content", "")
        second_user = captured_messages[1][1].get("content", "")
        # The user message content should contain the original query
        assert "write a doc" in first_user


# ── Proactive capability routing hint tests ─────────────────────────


def _empty_manifest() -> CapabilityManifest:
    """A manifest with no entries — simulates a mode with no tools."""
    return CapabilityManifest(generated_at="2026-01-01T00:00:00Z", entries=[])


def _manifest_with_search() -> CapabilityManifest:
    """A manifest advertising a web_search capability."""
    from kernel.models import CapabilityEntry

    return CapabilityManifest(
        generated_at="2026-01-01T00:00:00Z",
        entries=[
            CapabilityEntry(
                capability_id="web_search",
                kind="tool",
                status="available",
                source="mcp_pool",
            )
        ],
    )


def test_routing_hint_returns_empty_when_no_routing_targets() -> None:
    """No routing targets means no hint is injected (zero tokens)."""
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="understand the latest frameworks and do a SWOT",
        capability_manifest=_empty_manifest(),
        routing_targets=[],
    )
    assert hint == ""


def test_routing_hint_returns_empty_when_mode_has_search_and_no_reasoning() -> None:
    """A mode that already has search and a pure-search query gets no hint."""
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="search",
        query="what is the latest news on rust",
        capability_manifest=_manifest_with_search(),
        routing_targets=["default", "thinkabout"],
    )
    assert hint == ""


def test_routing_hint_prefers_thinkabout_for_research_then_reason() -> None:
    """A query pairing a research cue with a reasoning cue (understand +
    SWOT + latest) should point to thinkabout, not the generic target list.
    This is the regression case for the incident where 'latest frameworks
    and do a SWOT' was routed to the one-shot search action mode.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="get a better understanding of the latest frameworks and agent building tools on the market today and do a SWOT based on it",
        capability_manifest=_empty_manifest(),
        routing_targets=["search", "thinkabout", "markdown", "coding"],
    )
    assert hint != ""
    assert "thinkabout" in hint
    # Must explicitly steer away from the one-shot search action mode
    assert "search" in hint.lower()
    assert "one-shot" in hint.lower() or "one shot" in hint.lower()
    assert "transition_stage" in hint


def test_routing_hint_detects_analyze_tradeoff_intent() -> None:
    """'analyze the latest tradeoffs' is research-then-reason → thinkabout."""
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="analyze the latest tradeoffs in serverless frameworks",
        capability_manifest=_empty_manifest(),
        routing_targets=["search", "thinkabout"],
    )
    assert "thinkabout" in hint


def test_routing_hint_falls_back_to_generic_list_for_tool_gap() -> None:
    """A pure tool-gap query (write a file) with no reasoning cue lists all
    targets rather than singling out thinkabout.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="write a file with the results",
        capability_manifest=_empty_manifest(),
        routing_targets=["markdown", "coding", "thinkabout"],
    )
    assert hint != ""
    assert "transition_stage" in hint
    # No thinkabout-specific steering copy
    assert "one-shot" not in hint.lower()


def test_routing_hint_search_gap_without_reasoning_lists_targets() -> None:
    """A pure 'search the web' query (no reasoning cue) gets the generic
    target list, not the thinkabout-preferred copy.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="search the web for cloud costs",
        capability_manifest=_empty_manifest(),
        routing_targets=["search", "thinkabout"],
    )
    assert hint != ""
    assert "transition_stage" in hint
    assert "one-shot" not in hint.lower()


# ── Two-goal fork routing hint tests ──────────────────────────────────


def test_routing_hint_forks_for_remember_and_write_markdown() -> None:
    """A request pairing remember/learn with write-a-markdown-doc has two
    concurrent goals. The hint must steer the model to a fork fan-out
    (`fork:learn,markdown`), not a single-mode handoff.

    Regression case for docs/lisa-markdown-stage-failure-2026-07-17.md: the
    model called markdown_forge_document / markdown_generate from default
    mode (mode-scoped tools, rejected by tool policy) instead of forking.
    The proactive hint must surface before the model wastes an iteration.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="remember that pretty much verbatim and also write a markdown of it",
        capability_manifest=_empty_manifest(),
        routing_targets=["learn", "markdown", "coding", "thinkabout", "search"],
    )
    assert hint != ""
    assert "transition_stage" in hint
    # Must name the fork target, not a single mode handoff
    assert "fork:learn,markdown" in hint
    # Must warn that mode-scoped tools would be rejected from default
    assert "mode-scoped" in hint.lower() or "rejected" in hint.lower()


def test_routing_hint_fork_requires_both_routable_modes() -> None:
    """When one of learn/markdown is not a routing target, do not suggest a
    fork — fall back to the single-goal branches (or no hint).
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    # markdown not routable from here → no fork hint; learn+write cues with
    # no tool/search/research-reason gap means no hint at all.
    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="remember the decision and write a markdown of it",
        capability_manifest=_empty_manifest(),
        routing_targets=["learn", "coding"],  # markdown missing
    )
    assert "fork:" not in hint


def test_routing_hint_fork_preserves_target_case() -> None:
    """The fork target string preserves the original (non-lowercased) mode
    IDs from routing_targets, matching what the FSM catalog expects.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="take a note of this and write a markdown document",
        capability_manifest=_empty_manifest(),
        routing_targets=["learn", "markdown"],
    )
    assert "fork:learn,markdown" in hint
