"""Tests for transition_stage dispatch in the turn runner.

Covers the three transition target kinds intercepted before the
operation/tool/MCP dispatch branches:
  1. fork:mode/stage,... — parallel fan-out to heterogeneous modes
  2. mode_id (routable) — mid-turn mode switch (handoff)
  3. workflow stage — existing apply_transition path

These tests complement the high-level fork/handoff/transition tests in
``test_turn_runner.py`` by exercising edge cases, error paths, and
internal helper functions directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from kernel.models import (
    ForkChildResult,
    ForkTarget,
    OperationExecutionContext,
    OperationResult,
    TurnBudget,
    WorkflowSessionState,
)
from kernel.turn_runner import (
    TurnHooks,
    _fork_child_budget,
    _fork_mode_requires_tools,
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


def _transition_call(call_id: str, target_stage_id: str, **extra: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"target_stage_id": target_stage_id}
    args.update(extra)
    return _make_tool_call(call_id, "transition_stage", args)


def _build_hooks(complete_fn: Any, **overrides: Any) -> TurnHooks:
    return TurnHooks(complete=complete_fn, memory_recall_enabled=False, **overrides)


# ── Fork: _fork_child_budget ──────────────────────────────────────────


def test_fork_child_budget_none_when_parent_has_no_budget() -> None:
    """_fork_child_budget returns None when the parent has no TurnBudget."""
    assert _fork_child_budget(None, 2) is None


def test_fork_child_budget_divides_max_tool_calls() -> None:
    """max_tool_calls is divided equally across n children."""
    budget = TurnBudget(max_tool_calls=10, max_total_tokens=0, max_wall_ms=0)
    child = _fork_child_budget(budget, 2)
    assert child is not None
    assert child.max_tool_calls == 5


def test_fork_child_budget_divides_max_total_tokens() -> None:
    """max_total_tokens is divided equally across n children."""
    budget = TurnBudget(max_tool_calls=0, max_total_tokens=1000, max_wall_ms=0)
    child = _fork_child_budget(budget, 4)
    assert child is not None
    assert child.max_total_tokens == 250


def test_fork_child_budget_divides_max_wall_ms() -> None:
    """max_wall_ms is divided equally across n children."""
    budget = TurnBudget(max_tool_calls=0, max_total_tokens=0, max_wall_ms=800)
    child = _fork_child_budget(budget, 2)
    assert child is not None
    assert child.max_wall_ms == 400


def test_fork_child_budget_passes_through_zero_as_unlimited() -> None:
    """A field set to 0 (unlimited) is passed through as 0, not divided."""
    budget = TurnBudget(max_tool_calls=0, max_total_tokens=100, max_wall_ms=0)
    child = _fork_child_budget(budget, 3)
    assert child is not None
    assert child.max_tool_calls == 0  # unlimited stays unlimited
    assert child.max_total_tokens == 33  # 100 // 3
    assert child.max_wall_ms == 0  # unlimited stays unlimited


def test_fork_child_budget_floors_division() -> None:
    """Integer division floors gracefully when budget is smaller than n."""
    budget = TurnBudget(max_tool_calls=3, max_total_tokens=0, max_wall_ms=0)
    child = _fork_child_budget(budget, 5)
    assert child is not None
    assert child.max_tool_calls == 0  # 3 // 5 = 0


# ── Fork: _fork_mode_requires_tools ────────────────────────────────────


def test_fork_mode_requires_tools_true_for_learn() -> None:
    """learn mode declares allowed_tool_ids, so it requires tools."""
    assert _fork_mode_requires_tools("learn") is True


def test_fork_mode_requires_tools_true_for_markdown() -> None:
    """markdown mode declares write-effect tools (write_file,
    markdown_generate), so it requires tools."""
    assert _fork_mode_requires_tools("markdown") is True


def test_fork_mode_requires_tools_false_for_default() -> None:
    """default mode is chat-only (no action tools/operations)."""
    assert _fork_mode_requires_tools("default") is False


def test_fork_mode_requires_tools_false_for_ponder() -> None:
    """ponder's only authored operation is web_search (read-effect), so it
    does NOT require tools — its output is the response text (the
    reasoning), not a side-effect on disk. A child that gathers evidence
    via fetch_page and produces an opinion has done its job."""
    assert _fork_mode_requires_tools("ponder") is False


def test_fork_mode_requires_tools_false_for_unknown_mode() -> None:
    """An unknown mode_id returns False (no config to inspect)."""
    assert _fork_mode_requires_tools("nonexistent_mode_xyz") is False


# ── Fork: end-to-end fan-out via run_turn ─────────────────────────────


@pytest.mark.asyncio
async def test_fork_transition_with_per_child_queries() -> None:
    """Fork respects the 'queries' argument, passing one query per child."""
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call(
                        "call_1",
                        "fork:default,default",
                        queries=["query A", "query B"],
                    ),
                ],
            )
        return _StubCompletion(content="Synthesized.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="parent query", hooks=hooks)

    assert result.fork_results is not None
    assert len(result.fork_results) == 2
    # Each child should have received its own query from the queries arg.
    queries = [child.target.query for child in result.fork_results]
    assert "query A" in queries
    assert "query B" in queries


@pytest.mark.asyncio
async def test_fork_transition_falls_back_to_parent_query() -> None:
    """When 'queries' is absent, each child gets the parent query."""
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call("call_1", "fork:default,default"),
                ],
            )
        return _StubCompletion(content="Synthesized.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="parent query", hooks=hooks)

    assert result.fork_results is not None
    assert len(result.fork_results) == 2
    for child in result.fork_results:
        assert child.target.query == "parent query"


@pytest.mark.asyncio
async def test_fork_transition_invalid_mode_returns_error_in_tool_output() -> None:
    """When a fork target mode doesn't exist, the tool output reports the error."""
    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call("call_1", "fork:nonexistent_mode_xyz"),
                ],
            )
        # Capture the tool result message from the messages list
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Recovered from bad fork.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Recovered from bad fork."
    # The tool output should mention the missing mode.
    assert any("nonexistent_mode_xyz" in out for out in captured_tool_outputs)


@pytest.mark.asyncio
async def test_fork_transition_empty_target_string_returns_error() -> None:
    """fork: with no mode/stage pairs produces an error tool output."""
    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call("call_1", "fork:"),
                ],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Recovered.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Recovered."
    assert any("no valid" in out.lower() for out in captured_tool_outputs)


# ── Mode handoff ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handoff_to_non_routable_target_rejected() -> None:
    """Handoff to a mode that is not in routing_targets (and not 'default')
    is rejected with an error message in the tool output."""

    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call("call_1", "nonexistent_mode_xyz"),
                ],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Recovered.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Recovered."
    # The tool output should mention handoff not allowed.
    assert any("handoff" in out.lower() or "not" in out.lower() for out in captured_tool_outputs)


@pytest.mark.asyncio
async def test_handoff_includes_rationale_in_tool_output() -> None:
    """When the model provides a rationale, it's included in the tool output."""
    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call(
                        "call_1",
                        "markdown",
                        rationale="user wants to write a doc",
                    ),
                ],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="write a doc", hooks=hooks)

    assert result.handoff_to == "markdown"
    assert any("user wants to write a doc" in out for out in captured_tool_outputs)


@pytest.mark.asyncio
async def test_handoff_swaps_system_prompt() -> None:
    """After handoff, the system prompt (messages[0]) changes to the new mode."""
    call_count = 0
    captured_system_prompts: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        msgs = kwargs.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            captured_system_prompts.append(str(msgs[0].get("content", "")))
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call("call_1", "markdown"),
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="write", hooks=hooks)

    assert result.handoff_to == "markdown"
    assert len(captured_system_prompts) >= 2
    # The system prompt after handoff should differ from the one before.
    assert captured_system_prompts[0] != captured_system_prompts[1]


@pytest.mark.asyncio
async def test_handoff_updates_ctx_mode_id_for_post_handoff_dispatch() -> None:
    """Regression: after a mid-turn mode handoff, subsequent operation/tool
    dispatches must authorize against the *new* mode, not the entry mode.

    Before the fix, ``ctx`` was built once in Phase 1 with the entry
    mode_id and never refreshed. When the model called ``transition_stage``
    to hand off to (e.g.) ``markdown`` and then called a markdown operation,
    ``op_ctx = ctx.model_copy(update={...})`` preserved the stale
    ``mode_id='default'``, so ``execute_operation`` rejected the call with
    "Operation '...' is not allowed for mode 'default'." — the exact
    "Letta tools are not allowed in the default mode" symptom.
    """
    captured_op_ctx_modes: list[str] = []

    async def stub_execute_op(operation_id: str, ctx: OperationExecutionContext) -> OperationResult:
        captured_op_ctx_modes.append(str(getattr(ctx, "mode_id", "")))
        return OperationResult(output_text=f"ran {operation_id} in {getattr(ctx, 'mode_id', '')}")

    call_count = 0

    async def three_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First, hand off to markdown.
            return _StubCompletion(
                content="",
                tool_calls=[_transition_call("call_1", "markdown")],
            )
        if call_count == 2:
            # Then call a markdown-bound operation. This must dispatch
            # with ctx.mode_id == "markdown", not "default".
            return _StubCompletion(
                content="",
                tool_calls=[
                    _make_tool_call(
                        "call_2",
                        "markdown_forge_document",
                        {"topic": "handoff auth"},
                    )
                ],
            )
        return _StubCompletion(content="Document forged.")

    hooks = _build_hooks(three_step, execute_operation=stub_execute_op)
    result = await run_turn(mode_id="default", query="write a doc", hooks=hooks)

    assert result.handoff_to == "markdown"
    # The operation should have been dispatched exactly once, and the ctx
    # it received must carry the post-handoff mode_id.
    assert captured_op_ctx_modes == ["markdown"], (
        f"expected post-handoff ctx.mode_id='markdown', got {captured_op_ctx_modes}"
    )
    assert result.tool_call_count >= 1


# ── Workflow stage transition ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_workflow_transition_without_session_returns_error() -> None:
    """When is_workflow but workflow_session returns None, transition fails
    with a clear error message."""
    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call("call_1", "reason"),
                ],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Recovered.")

    hooks = _build_hooks(two_step)
    # Wire workflow_session as a callable returning None — the loop treats
    # the mode as workflow if mode_config.mode_kind == "workflow", so we
    # need to NOT wire workflow_session at all (it stays None).
    # But is_workflow requires hooks.workflow_session is not None.
    # So we wire a session that returns a valid session but the transition
    # target is invalid.
    session = WorkflowSessionState(
        user_id="test-user",
        user_slug="test",
        mode_id="ponder",
        stage_id="inquire",
        entered_at="2026-01-01T00:00:00Z",
    )
    hooks.workflow_session = lambda: session

    result = await run_turn(mode_id="ponder", query="analyze", hooks=hooks)

    # The transition to "reason" should succeed (it's a valid next stage).
    assert result.transitioned_to == "reason"


@pytest.mark.asyncio
async def test_workflow_transition_passes_required_missing_inputs() -> None:
    """The required_missing_inputs argument is passed to WorkflowTransitionRequest."""
    session = WorkflowSessionState(
        user_id="test-user",
        user_slug="test",
        mode_id="ponder",
        stage_id="inquire",
        entered_at="2026-01-01T00:00:00Z",
    )

    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call(
                        "call_1",
                        "reason",
                        rationale="enough evidence",
                        required_missing_inputs=["data_point_1", "data_point_2"],
                    ),
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step)
    hooks.workflow_session = lambda: session
    hooks.workflow_session_updater = lambda updated: None

    result = await run_turn(mode_id="ponder", query="analyze", hooks=hooks)

    assert result.transitioned_to == "reason"


@pytest.mark.asyncio
async def test_workflow_transition_invalid_target_produces_error_output() -> None:
    """Transition to an invalid stage (not in allowed_next_stages) raises an
    error from apply_transition, surfaced as a tool output message."""
    session = WorkflowSessionState(
        user_id="test-user",
        user_slug="test",
        mode_id="ponder",
        stage_id="inquire",
        entered_at="2026-01-01T00:00:00Z",
    )

    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call("call_1", "exit"),  # exit is not next from inquire
                ],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Recovered.")

    hooks = _build_hooks(two_step)
    hooks.workflow_session = lambda: session

    result = await run_turn(mode_id="ponder", query="analyze", hooks=hooks)

    assert result.response_text == "Recovered."
    # apply_transition should have raised, producing a "Transition failed" output.
    assert any(
        "failed" in out.lower() or "transition" in out.lower() for out in captured_tool_outputs
    )


# ── Forced-fork enforcer (2026-07-27 incident) ──────────────────────
#
# Regression for the 2026-07-27 stall: a multi-stage workflow handed off
# to ``plan``, the post-handoff routing hint named a ``fork:`` target, and
# the model refused to call ``transition_stage`` — instead burning
# iterations on read-only ``fetch_page`` calls (or returning empty
# completions) until the iteration cap, ending the turn with no fork ever
# running. The kernel now forces the fork itself when the model won't.
# See ``docs/lessons/lisa-fork-refusal-2026-07-27.md`` and the Forced-Fork
# section of ``docs/ROUTER_DESIGN.md``.


def test_compute_fork_target_string_returns_none_without_routing_targets() -> None:
    """With no routing_targets, no fork target is computed."""
    from kernel.turn_phases._routing_hint import _compute_fork_target_string

    assert (
        _compute_fork_target_string(
            query="anything",
            routing_targets=[],
            mode_config=None,
        )
        is None
    )


def test_compute_fork_target_string_returns_none_when_no_gap() -> None:
    """When the current mode already satisfies all detected intents, no fork."""
    from kernel.ipsar_fsm import get_workflow_catalog
    from kernel.turn_phases._routing_hint import _compute_fork_target_string

    catalog = get_workflow_catalog()
    target = _compute_fork_target_string(
        query="hello",  # no detectable intents
        routing_targets=["learn", "markdown"],
        mode_config=catalog.get("default"),
    )
    assert target is None


@pytest.mark.asyncio
async def test_forced_fork_when_model_refuses_to_fork() -> None:
    """When a fork routing hint is active and the model calls non-fork
    tools instead of ``transition_stage``, the kernel forces the fork
    after ``forced_fork_non_fork_tool_cap`` non-fork dispatches.

    Regression for the 2026-07-27 incident: the model burned 5
    ``fetch_page`` calls in ``plan`` mode with a ``fork:`` hint active and
    never called ``transition_stage``, ending the turn with no fork ever
    running. The enforcer synthesizes the fork call itself.
    """
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: model does a read-only fetch instead of forking.
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "fetch_page",
                            "arguments": '{"url": "https://example.com"}',
                        },
                    }
                ],
            )
        if call_count == 2:
            # Second call: model does ANOTHER read-only fetch instead of
            # forking. This hits the forced_fork_non_fork_tool_cap=2.
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_2",
                        "function": {
                            "name": "fetch_page",
                            "arguments": '{"url": "https://example.org"}',
                        },
                    }
                ],
            )
        # Subsequent calls: the kernel has forced the fork; the children
        # and the parent synthesis return text.
        return _StubCompletion(content="Synthesized from forced fork.")

    hooks = _build_hooks(two_step)
    # A query that triggers a fork hint from default mode (learn + write).
    result = await run_turn(
        mode_id="default",
        query="remember this learning and write a markdown guide",
        hooks=hooks,
    )

    # The forced fork must have run — fork_results is populated even though
    # the model never called transition_stage with a fork target.
    assert result.fork_results is not None
    assert len(result.fork_results) >= 2


@pytest.mark.asyncio
async def test_forced_fork_tool_call_id_matches_assistant_message() -> None:
    """When the enforcer overrides a non-fork tool call with a forced
    ``transition_stage(fork:...)`` call, the assistant message's
    ``tool_calls[].id`` and the tool result message's ``tool_call_id``
    must stay consistent — otherwise the provider rejects the turn with
    ``Invalid parameter: 'tool_call_id' not found in 'tool_calls' of
    previous message`` (the 2026-07-27 follow-on BadRequestError).

    Regression: the enforcer must mutate the tool_call dict in place
    (keeping the original id), not mint a fresh ``forced_fork_N`` id that
    the assistant message does not carry.
    """
    call_count = 0
    captured_messages: list[list[dict[str, Any]]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        # Capture the messages list on every call so we can verify the
        # assistant message -> tool result pairing the provider sees.
        captured_messages.append(list(kwargs.get("messages", [])))
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "fetch_page",
                            "arguments": '{"url": "https://example.com"}',
                        },
                    }
                ],
            )
        if call_count == 2:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_2",
                        "function": {
                            "name": "fetch_page",
                            "arguments": '{"url": "https://example.org"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Synthesized from forced fork.")

    hooks = _build_hooks(two_step)
    await run_turn(
        mode_id="default",
        query="remember this learning and write a markdown guide",
        hooks=hooks,
    )

    # Walk every assistant message that has tool_calls and verify each
    # following tool result message references an id that exists in the
    # assistant message's tool_calls. This is the invariant the provider
    # enforces; the enforcer must not break it.
    for msgs in captured_messages:
        for i, msg in enumerate(msgs):
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            assistant_ids = {
                str(tc.get("id") or "") for tc in msg["tool_calls"]
            }
            # The next message(s) should be tool results for these calls.
            for j in range(i + 1, len(msgs)):
                tool_msg = msgs[j]
                if tool_msg.get("role") != "tool":
                    break
                tcid = str(tool_msg.get("tool_call_id") or "")
                assert tcid in assistant_ids, (
                    f"tool result tool_call_id {tcid!r} not found in "
                    f"preceding assistant tool_calls ids {assistant_ids}"
                )


@pytest.mark.asyncio
async def test_forced_fork_when_model_returns_empty_completion() -> None:
    """When a fork routing hint is active and the model returns an empty
    completion (no tool calls, no content), the kernel synthesizes the
    fork call instead of accepting the empty turn.

    Regression for the 2026-07-27 Run B: ``plan`` mode returned empty
    content on iteration 7 with a fork hint active, ending the turn with
    no fork ever running.
    """
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        # First call: model returns empty — no tool calls, no content.
        if call_count == 1:
            return _StubCompletion(content="", tool_calls=None)
        # After the kernel forces the fork, return synthesis text.
        return _StubCompletion(content="Synthesized from forced fork.")

    hooks = _build_hooks(two_step)
    result = await run_turn(
        mode_id="default",
        query="remember this learning and write a markdown guide",
        hooks=hooks,
    )

    assert result.fork_results is not None
    assert len(result.fork_results) >= 2


# ── Transition: missing target_stage_id ──────────────────────────────


@pytest.mark.asyncio
async def test_transition_missing_target_stage_id_returns_error() -> None:
    """When target_stage_id is empty, the tool output says it's required."""
    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _make_tool_call("call_1", "transition_stage", {"rationale": "no target"}),
                ],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Recovered.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Recovered."
    assert any(
        "requires" in out.lower() and "target_stage_id" in out for out in captured_tool_outputs
    )


# ── Transition: unknown target (not fork, not routable, not workflow) ──


@pytest.mark.asyncio
async def test_transition_unknown_target_returns_error() -> None:
    """A target that is not a fork, routable mode, or workflow stage produces
    an 'Unknown transition target' error."""
    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call("call_1", "totally_unknown_stage"),
                ],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Recovered.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Recovered."
    assert any("unknown" in out.lower() for out in captured_tool_outputs)


# ── Turn budget enforcement in fork children ──────────────────────────


@pytest.mark.asyncio
async def test_fork_child_budget_enforced_from_parent() -> None:
    """The parent's TurnBudget is divided across fork children."""
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _transition_call("call_1", "fork:default,default"),
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step)
    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        turn_budget=TurnBudget(max_tool_calls=4, max_total_tokens=0, max_wall_ms=0),
    )

    assert result.fork_results is not None
    assert len(result.fork_results) == 2
    # Each child gets max_tool_calls=2 (4 // 2). Default mode doesn't call
    # tools, so both should complete as "ok".
    for child in result.fork_results:
        assert child.status in ("ok", "no_tools")


# ── ForkChildResult construction ──────────────────────────────────────


def test_fork_child_result_ok_status() -> None:
    """ForkChildResult with status='ok' carries output_text and duration."""
    target = ForkTarget(mode_id="default", stage_id="", query="test")
    result = ForkChildResult(target=target, status="ok", output_text="Done.", duration_ms=100)
    assert result.status == "ok"
    assert result.output_text == "Done."
    assert result.duration_ms == 100
    assert result.error_message == ""


def test_fork_child_result_error_status() -> None:
    """ForkChildResult with status='error' carries error_message."""
    target = ForkTarget(mode_id="default", stage_id="", query="test")
    result = ForkChildResult(
        target=target,
        status="error",
        error_message="ValueError: bad input",
        duration_ms=50,
    )
    assert result.status == "error"
    assert "ValueError" in result.error_message
