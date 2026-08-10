"""Tests for the centralized ``evaluate_turn_decision`` loop contract.

The decision helper reduces every model iteration to one of three
outcomes: ``CONTINUE``, ``HANDOFF``, or ``FINISH``. These tests cover
each branch in isolation, confirming the function is pure (no side
effects) and that the priority ordering is correct:

1. Transition request → ``HANDOFF`` (always wins).
2. Terminal position → ``FINISH`` (receipt-backed completion).
3. Tool calls + budget remaining → ``CONTINUE``.
4. Budget exhausted, no transition → ``FINISH`` (stall/cap).
5. No tools, no transition, budget remaining → ``FINISH`` (final answer).
"""

from __future__ import annotations

from kernel.turn_phases._routing import (
    TurnDecision,
    evaluate_turn_decision,
)


def test_transition_request_produces_handoff() -> None:
    """A ``transition_stage`` call always produces ``HANDOFF`` with the target."""
    result = evaluate_turn_decision(
        has_tool_calls=False,
        has_transition_request=True,
        transition_target="markdown",
        is_terminal=False,
        budget_exhausted=False,
        has_content=True,
    )
    assert result.decision is TurnDecision.HANDOFF
    assert result.target == "markdown"


def test_transition_request_wins_over_terminal() -> None:
    """Even on a terminal position, a transition request takes priority."""
    result = evaluate_turn_decision(
        has_tool_calls=False,
        has_transition_request=True,
        transition_target="search",
        is_terminal=True,
        budget_exhausted=False,
        has_content=False,
    )
    assert result.decision is TurnDecision.HANDOFF
    assert result.target == "search"


def test_terminal_position_produces_finish() -> None:
    """A terminal workflow position (all steps satisfied) produces ``FINISH``."""
    result = evaluate_turn_decision(
        has_tool_calls=False,
        has_transition_request=False,
        transition_target=None,
        is_terminal=True,
        budget_exhausted=False,
        has_content=True,
    )
    assert result.decision is TurnDecision.FINISH
    assert "terminal" in result.reason


def test_tool_calls_with_budget_produces_continue() -> None:
    """Tool calls with remaining budget produce ``CONTINUE``."""
    result = evaluate_turn_decision(
        has_tool_calls=True,
        has_transition_request=False,
        transition_target=None,
        is_terminal=False,
        budget_exhausted=False,
        has_content=False,
    )
    assert result.decision is TurnDecision.CONTINUE


def test_budget_exhausted_with_no_transition_produces_finish() -> None:
    """Budget exhaustion with no pending transition produces ``FINISH``."""
    result = evaluate_turn_decision(
        has_tool_calls=False,
        has_transition_request=False,
        transition_target=None,
        is_terminal=False,
        budget_exhausted=True,
        has_content=False,
    )
    assert result.decision is TurnDecision.FINISH
    assert "budget" in result.reason


def test_budget_exhausted_with_tool_calls_produces_finish() -> None:
    """Tool calls do not override an exhausted budget — the harness yields."""
    result = evaluate_turn_decision(
        has_tool_calls=True,
        has_transition_request=False,
        transition_target=None,
        is_terminal=False,
        budget_exhausted=True,
        has_content=False,
    )
    assert result.decision is TurnDecision.FINISH


def test_no_tools_no_transition_produces_finish() -> None:
    """A model answer with no tool calls and no transition produces ``FINISH``."""
    result = evaluate_turn_decision(
        has_tool_calls=False,
        has_transition_request=False,
        transition_target=None,
        is_terminal=False,
        budget_exhausted=False,
        has_content=True,
    )
    assert result.decision is TurnDecision.FINISH
    assert "final answer" in result.reason


def test_empty_response_produces_finish() -> None:
    """An empty model response (no tools, no content) still produces ``FINISH``."""
    result = evaluate_turn_decision(
        has_tool_calls=False,
        has_transition_request=False,
        transition_target=None,
        is_terminal=False,
        budget_exhausted=False,
        has_content=False,
    )
    assert result.decision is TurnDecision.FINISH
    assert "no content" in result.reason


def test_transition_with_empty_target_does_not_handoff() -> None:
    """A transition request with an empty target falls through to other checks."""
    result = evaluate_turn_decision(
        has_tool_calls=True,
        has_transition_request=True,
        transition_target=None,
        is_terminal=False,
        budget_exhausted=False,
        has_content=False,
    )
    # Empty target means the transition is invalid; the tool calls keep the loop going.
    assert result.decision is TurnDecision.CONTINUE