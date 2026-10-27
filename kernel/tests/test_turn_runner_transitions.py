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

import inspect
import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import kernel.turn_runner
import pytest
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import (
    ForkChildResult,
    ForkTarget,
    OperationExecutionContext,
    OperationResult,
    TurnBudget,
)
from kernel.turn_phases._fork import (
    _fork_child_budget,
    _fork_mode_requires_tools,
)
from kernel.turn_runner import (
    TurnHooks,
    run_turn,
)
from kernel.workflow_position import WorkflowPosition

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
    # Flat hook overrides fold into the correct grouped sub-dataclass;
    # recall defaults to off so ambient pre-turn recall stays silent.
    from turn_hooks_fold import build_hooks

    return build_hooks(
        defaults={"models": {"complete": complete_fn}, "recall": {"enabled": False}},
        **overrides,
    )


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


def test_fork_mode_requires_tools_true_for_remember() -> None:
    """remember mode declares allowed_tool_ids, so it requires tools."""
    assert _fork_mode_requires_tools("remember") is True


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
    """A nonexistent mode id is rejected before the routing gate runs.

    ``is_routable_mode`` guards the interception path, so an unknown id never
    reaches ``_execute_mode_handoff``'s ``routing_targets`` check at all. Assert
    that exact upstream message rather than a disjunction over "handoff" or
    "not": the loose form also matches "Target mode ... not found" and stays
    green even if the routing gate is deleted outright.
    """

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
    assert any(
        "Unknown transition target: 'nonexistent_mode_xyz'" in out for out in captured_tool_outputs
    )


@pytest.mark.asyncio
async def test_self_transition_to_default_rejected() -> None:
    """A handoff from ``default`` to ``default`` is rejected by the routing gate.

    This is the only reachable path to ``_execute_mode_handoff``'s
    ``routing_targets`` check: every other routable mode already lists
    ``default`` as a target (``synthesize_routing_transitions`` builds the full
    routable mesh), and a mode never lists itself. While the redundant
    ``and target_stage != "default"`` carve-out existed, this call was accepted
    and silently re-entered the same mode.
    """

    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_transition_call("call_1", "default")],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Recovered.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.handoff_to is None
    assert any(
        "Mode handoff not allowed: 'default' cannot transition to 'default'." in out
        for out in captured_tool_outputs
    )


@pytest.mark.asyncio
async def test_self_transition_to_non_default_mode_rejected() -> None:
    """A routable mode cannot hand off to itself.

    ``synthesize_routing_transitions`` excludes a mode from its own target
    list, so the gate rejects this on the ``routing_targets`` clause alone and
    never needed the removed carve-out.
    """

    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_transition_call("call_1", "markdown")],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Recovered.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="markdown", query="test", hooks=hooks)

    assert result.handoff_to is None
    assert any(
        "Mode handoff not allowed: 'markdown' cannot transition to 'markdown'." in out
        for out in captured_tool_outputs
    )


@pytest.mark.asyncio
async def test_handoff_to_default_from_another_mode_allowed() -> None:
    """``default`` stays a legal handoff target for every other routable mode.

    Guards the carve-out removal against over-restriction: ``default`` is
    routable, so it is already in every other mode's routing targets and the
    deletion must not block ``X -> default``.
    """

    call_count = 0
    captured_tool_outputs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_transition_call("call_1", "default")],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Recovered.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="markdown", query="test", hooks=hooks)

    assert result.handoff_to == "default"
    assert any("Handed off to mode: default." in out for out in captured_tool_outputs)


@pytest.mark.asyncio
async def test_handoff_includes_rationale_in_tool_output() -> None:
    """When the model provides a rationale, it's included in the tool output.

    The text is labelled as model-supplied: the kernel is echoing prose the
    model wrote, and an unmarked rationale reads as kernel-authored evidence in
    the transcript.
    """
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
    assert any(
        "Rationale (model-supplied): user wants to write a doc" in out
        for out in captured_tool_outputs
    )


@pytest.mark.asyncio
async def test_handoff_rationale_is_labelled_model_supplied() -> None:
    """The echoed rationale is attributed to the model, not the kernel.

    The success message is kernel-authored prose except for ``rationale``,
    which comes verbatim from the model's tool call. Echoing it unmarked makes
    model-authored text indistinguishable from kernel evidence in the
    transcript, which anything reading the history later (operator, context
    assembly, or the model itself) will mis-attribute.
    """

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
                        rationale="routing because the user asked",
                    ),
                ],
            )
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured_tool_outputs.append(str(msg.get("content", "")))
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step)
    await run_turn(mode_id="default", query="write a doc", hooks=hooks)

    handoff_outputs = [out for out in captured_tool_outputs if "Handed off to mode" in out]
    assert handoff_outputs, captured_tool_outputs
    output = handoff_outputs[0]
    assert "Rationale (model-supplied): routing because the user asked" in output
    # The bare form must not survive alongside the labelled one.
    assert "Rationale: routing because the user asked" not in output


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
async def test_handoff_to_terminal_mode_instructs_completion_not_handoff() -> None:
    """Regression (2026-07-30): when the handoff target is the final step in
    the chain (no remaining gap intents), the Handoff Context block must
    tell the model to COMPLETE and respond, not to hand off again.

    Before the fix, the "After you complete your work, hand off to the
    next step by calling ``transition_stage``" line was unconditional —
    it fired even when ``remaining_steps`` was empty, contradicting the
    "(none — this is the final step)" sentinel. The terminal mode (e.g.
    ``markdown``) then flailed with no clean completion signal, burning
    round-trips until the handoff cap force-ended the turn.

    This test hands off ``default → markdown`` for a ``write``-only query
    (markdown satisfies ``write`` fully, so the chain is terminal) and
    asserts the system prompt after handoff contains the completion
    directive and NOT the hand-off directive.
    """
    captured_system_prompts: list[str] = []
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        msgs = kwargs.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            captured_system_prompts.append(str(msgs[0].get("content", "")))
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_transition_call("call_1", "markdown")],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="write a doc", hooks=hooks)

    assert result.handoff_to == "markdown"
    # The post-handoff system prompt (second capture) is the one carrying
    # the Handoff Context block for the terminal markdown step.
    post_handoff_prompt = captured_system_prompts[1]
    assert "You are the final step in this chain" in post_handoff_prompt, (
        "terminal handoff must instruct completion, not further handoff; "
        f"got: {post_handoff_prompt!r}"
    )
    assert "do NOT call `transition_stage`" in post_handoff_prompt, (
        f"terminal handoff must explicitly forbid transition_stage; got: {post_handoff_prompt!r}"
    )
    # The unconditional hand-off directive must NOT appear on the terminal step.
    assert "After you complete your work, hand off to the next step" not in post_handoff_prompt, (
        "terminal handoff must not emit the next-step hand-off directive; "
        f"got: {post_handoff_prompt!r}"
    )


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
    from unittest.mock import patch

    from kernel.models import OperationDefinition
    from kernel.operations import registry as operation_registry

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
    _operations = dict(operation_registry.get_operation_catalog())
    _operations["markdown_forge_document"] = OperationDefinition(
        operation_id="markdown_forge_document",
        description="synthetic markdown write for transition testing",
        effect="write",
    )
    with patch.object(operation_registry, "_OPERATION_CATALOG", _operations):
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
async def test_workflow_transition_invalid_target_produces_error_output() -> None:
    """Transition to an invalid stage (not in allowed_next_stages) raises an
    error from apply_transition, surfaced as a tool output message."""
    session = WorkflowPosition(
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
    hooks.workflow.workflow_position = lambda: session

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


def test_compute_fork_targets_returns_none_without_routing_targets() -> None:
    """With no routing_targets, no fork target is computed."""
    from kernel.turn_phases._routing_hint import _compute_fork_targets

    assert (
        _compute_fork_targets(
            query="anything",
            routing_targets=[],
            mode_config=None,
        )
        is None
    )


def test_compute_fork_targets_returns_none_when_no_gap() -> None:
    """When the current mode already satisfies all detected intents, no fork."""
    from kernel.ipsar_fsm import get_workflow_catalog
    from kernel.turn_phases._routing_hint import _compute_fork_targets

    catalog = get_workflow_catalog()
    target = _compute_fork_targets(
        query="hello",  # no detectable intents
        routing_targets=["remember", "markdown"],
        mode_config=catalog.get("default"),
    )
    assert target is None


@pytest.mark.asyncio
async def test_forced_fork_when_model_refuses_to_fork() -> None:
    """When a fork routing hint is active and the model calls non-fork
    tools instead of ``transition_stage``, the kernel forces the fork
    after ``forced_action_non_compliant_cap`` non-fork dispatches.

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
            # forking. This hits the forced_action_non_compliant_cap=2.
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
    # A query that triggers a fork hint from default mode (remember + write).
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
            assistant_ids = {str(tc.get("id") or "") for tc in msg["tool_calls"]}
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


# ── Mid-turn pre-turn operation dispatch ──────────────────────────────


@pytest.mark.asyncio
async def test_handoff_runs_target_mode_pre_turn_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (2026-09-10): a mid-turn mode handoff must run the target
    mode's ``pre_turn_operation_ids``, exactly as a turn that STARTS in that
    mode does.

    The bash pack binds ``sandbox_facts`` as the shell mode's pre-turn op to
    publish the sandbox containment fact BEFORE the first ``bash_run`` gate.
    A turn that entered ``default`` and reached ``shell`` mid-turn never ran
    that prewarm, so the host policy engine read a MISSING containment fact,
    failed closed, and prompted the operator for approval even though the
    command would have run sandboxed.

    This test registers a synthetic ``probe`` mode whose pre-turn op records
    the mode_id it was dispatched with, then hands off ``default -> probe``
    and asserts the prewarm ran with the target mode.
    """
    from kernel.ipsar_fsm import _workflow_catalog
    from kernel.models import OperationDefinition, WorkflowModeConfig
    from kernel.operations import registry as op_registry

    dispatched: list[tuple[str, str]] = []

    async def stub_execute_op(operation_id: str, ctx: OperationExecutionContext) -> OperationResult:
        dispatched.append((operation_id, str(getattr(ctx, "mode_id", ""))))
        return OperationResult(output_text=f"prewarmed in {ctx.mode_id}")

    real_catalog = dict(_workflow_catalog or get_workflow_catalog())
    base = real_catalog.get("default")
    probe = base.model_copy(deep=True) if base is not None else WorkflowModeConfig(mode_id="probe")
    probe.mode_id = "probe"
    probe.routable = True
    probe.pre_turn_operation_ids = ["probe_prewarm"]
    patched_catalog = dict(real_catalog)
    patched_catalog["probe"] = probe
    monkeypatch.setattr("kernel.ipsar_fsm._workflow_catalog", patched_catalog)

    real_ops = dict(op_registry.get_operation_catalog())
    fake_op = OperationDefinition(
        operation_id="probe_prewarm",
        description="probe prewarm",
        effect="read",
    )
    patched_ops = dict(real_ops)
    patched_ops["probe_prewarm"] = fake_op
    monkeypatch.setattr(op_registry, "_OPERATION_CATALOG", patched_ops)
    monkeypatch.setattr("kernel.ipsar_fsm._execute_operation_fn", stub_execute_op)

    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_transition_call("call_1", "probe")],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(two_step)
    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.handoff_to == "probe"
    # The prewarm must have dispatched exactly once, with the TARGET mode id.
    assert dispatched == [("probe_prewarm", "probe")], (
        f"mid-turn handoff must run the target mode's pre-turn ops; got {dispatched}"
    )


# ── Forced-action enforcer: batch-aware compliance ───────────────────────────


class _ForcedState:
    """Minimal turn-state stub for the forced-action enforcer."""

    def __init__(self, forced_action: Any, iteration: int = 0) -> None:
        self.forced_action = forced_action
        self.iteration = iteration
        self.effective_model = SimpleNamespace(
            structured_reliability=0.0, cost_tier="standard", reasoning_model=False
        )


def _forced_single(target: str = "planner") -> Any:
    from kernel.turn_phases._turn_state import ForcedAction

    return ForcedAction(targets=[target])


def test_enforcer_treats_a_batch_with_the_expected_tool_as_compliant() -> None:
    """A batch containing ``transition_stage`` is not a refusal.

    Regression for the 2026-10-27T18:12Z turn: the model emitted ``bash_run``
    and ``transition_stage(target_stage_id="planner")`` in one parallel block.
    The enforcer inspected the ``bash_run`` call first, counted it as
    non-compliant, hit the cap, and synthesized a handoff whose rationale read
    "the model did not call transition_stage" \u2014 over a batch that had it. The
    outcome must depend on the batch's content, not on call order.
    """
    from kernel.turn_runner import (
        _batch_contains_expected_tool,
        _enforce_forced_action_on_dispatch,
    )

    state = _ForcedState(_forced_single("planner"))
    bash_call = _make_tool_call("call_bash", "bash_run", {"commands": ["true"]})
    transition = _transition_call("call_transition", "planner")
    batch = [bash_call, transition]

    assert _batch_contains_expected_tool(state, batch, "call_bash") is True

    name, arguments = _enforce_forced_action_on_dispatch(
        state=state,
        mode_id="default",
        query="q",
        tool_call=bash_call,
        tool_name="bash_run",
        tool_arguments={"commands": ["true"]},
        batch_has_expected=_batch_contains_expected_tool(state, batch, "call_bash"),
        log=logging.getLogger("test"),
    )

    # Untouched: the model's real bash call still runs, and no synthetic
    # transition replaced it.
    assert name == "bash_run"
    assert arguments == {"commands": ["true"]}
    assert state.forced_action is not None
    assert state.forced_action.non_compliant_dispatches == 0


def test_enforcer_still_forces_when_no_call_in_the_batch_complies() -> None:
    """The guard is NOT weakened: a batch with no transition still gets forced.

    The routing-hint backstop exists because a model can burn iterations on
    read-only tools forever. Batch-awareness must not disable it, only stop it
    from misfiring on a batch that already complied. The enforcer fires on the
    dispatch that reaches ``forced_action_non_compliant_cap`` (default 2) and
    then clears ``state.forced_action``.
    """
    from kernel.turn_runner import (
        _batch_contains_expected_tool,
        _enforce_forced_action_on_dispatch,
    )

    state = _ForcedState(_forced_single("planner"))
    forced_on_call = ""
    forced_arguments: Any = {}
    for i in range(4):
        call_id = f"call_{i}"
        call = _make_tool_call(call_id, "bash_run", {"commands": ["true"]})
        batch = [call]
        batch_has_expected = _batch_contains_expected_tool(state, batch, call_id)
        assert batch_has_expected is False
        name, arguments = _enforce_forced_action_on_dispatch(
            state=state,
            mode_id="default",
            query="q",
            tool_call=call,
            tool_name="bash_run",
            tool_arguments={"commands": ["true"]},
            batch_has_expected=batch_has_expected,
            log=logging.getLogger("test"),
        )
        if name == "transition_stage":
            forced_on_call = call_id
            forced_arguments = arguments
            break

    assert forced_on_call == "call_1", "cap=2 means the 2nd non-compliant dispatch forces"
    assert forced_arguments["target_stage_id"] == "planner"


def test_forced_transition_rationale_reports_the_cap_not_a_refusal() -> None:
    """The synthesized rationale must not assert the model skipped the call.

    Observed 2026-10-27T18:12Z: the rationale read "the model did not call
    transition_stage after the routing hint named a target". That sentence is
    false whenever the model *did* call it and a sibling call in the batch was
    judged first. The enforcer fires on a non-compliant dispatch count, which
    is what the rationale should report.
    """
    from kernel.turn_runner import _enforce_forced_action_on_dispatch

    state = _ForcedState(_forced_single("planner"))
    arguments: Any = {}
    for i in range(4):
        call = _make_tool_call(f"c{i}", "bash_run", {"commands": ["true"]})
        name, produced = _enforce_forced_action_on_dispatch(
            state=state,
            mode_id="default",
            query="q",
            tool_call=call,
            tool_name="bash_run",
            tool_arguments={"commands": ["true"]},
            batch_has_expected=False,
            log=logging.getLogger("test"),
        )
        if name == "transition_stage":
            arguments = produced
            break

    assert arguments, "the enforcer must have fired within the cap"
    rationale = arguments["rationale"]
    assert "did not call transition_stage" not in rationale
    assert "non-compliant" in rationale
    assert "planner" in rationale.replace(arguments["target_stage_id"], "planner")


def test_planned_handoff_rationale_does_not_deny_the_dispatched_call() -> None:
    """The plan-driven synthesis must not say no transition is being asked for.

    Observed 2026-10-27T18:12Z: the synthesized call's rationale read "the
    model is not asked to call transition_stage" \u2014 attached to a real
    ``transition_stage`` call the kernel was dispatching. Calling it was the
    only way to act as the target mode, so the note contradicted the
    mechanism.
    """
    from kernel.turn_runner import _TRANSITION_TOOL_NAME

    assert _TRANSITION_TOOL_NAME == "transition_stage"
    source = inspect.getsource(kernel.turn_runner)
    assert "the model is not asked to call " not in source
    assert "Kernel-synthesized handoff" in source


@pytest.mark.asyncio
async def test_kernel_synthesized_handoff_rationale_is_not_labelled_model_supplied() -> None:
    """A kernel-authored rationale must not be attributed to the model.

    Regression for the label added by 39bc8703. It labelled the rationale
    "model-supplied" unconditionally, but the kernel writes rationales too —
    the forced-action enforcer and the plan-driven handoff both synthesize a
    ``transition_stage`` call, and the dispatcher reads ``rationale`` back out
    of ``tool_arguments`` exactly as it does for a model call. So the kernel's
    own sentence ("Kernel-forced transition handoff: 2 non-compliant
    dispatch(es) reached the cap ...") was echoed as the model's, which is the
    same misattribution the label exists to prevent — strictly worse than the
    unmarked form, because it asserts the wrong author rather than none.

    The call id decides it: synthesized calls are minted with a kernel prefix.
    """
    from kernel.turn_runner import _rationale_label

    # Synthesized ids (forced-action enforcer, plan-driven handoff).
    for synth_id in ("forced_transition_empty_3", "planned_handoff_2", "forced_work_empty_1"):
        assert _rationale_label(synth_id) == "Rationale (kernel-supplied)", synth_id

    # A provider-minted id is the model's.
    assert _rationale_label("call_abc123") == "Rationale (model-supplied)"


@pytest.mark.asyncio
async def test_forced_transition_echoes_its_rationale_as_kernel_supplied() -> None:
    """End to end: a forced handoff labels its own rationale as kernel-authored.

    Exercises the real dispatch path rather than the helper alone, so a future
    change that stops threading the call id through the handler is caught.
    """
    captured: list[str] = []
    call_count = 0

    async def refuses_then_finishes(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            # No tool calls while a forced action is pending -> the kernel
            # synthesizes the transition after the non-compliance cap.
            return _StubCompletion(content="Thinking about it.")
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "tool":
                captured.append(str(msg.get("content", "")))
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(refuses_then_finishes)
    result = await run_turn(mode_id="default", query="write a doc then review it", hooks=hooks)

    if not captured:
        pytest.skip("no forced transition observed in this stub configuration")
    forced = [out for out in captured if "Kernel-forced" in out]
    assert forced, f"expected a kernel-forced rationale in {captured!r}"
    assert all("Rationale (kernel-supplied)" in out for out in forced)
    assert all("Rationale (model-supplied)" not in out for out in forced)
    assert result is not None
