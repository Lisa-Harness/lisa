"""Tests for kernel/turn_runner.py — the shared model+tool loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kernel.models import (
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
