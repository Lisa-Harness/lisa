"""Tests for the letta-agent action mode.

The letta-agent mode is a zero-Lisa-model delegation path: it runs two
entry operations (letta_ensure_agent, letta_send_message) in order and
returns their joined output_text without calling the host's
``complete`` function. These tests verify the mode declaration, routing
precedence vs. the chat-mode ``letta`` mode, and the no-model-call
execution path.
"""

from __future__ import annotations

from typing import Any

import pytest

from kernel.models import OperationExecutionContext, OperationResult
from kernel.turn_runner import TurnHooks, run_turn


# ── Mode declaration ─────────────────────────────────────────────────


def test_letta_agent_mode_is_loaded_as_action() -> None:
    """The letta-agent mode is in the catalog with mode_kind=action."""
    from kernel.ipsar_fsm import get_workflow_catalog

    cfg = get_workflow_catalog().get("letta-agent")
    assert cfg is not None, "letta-agent mode not loaded — check SDK_CONTENT_ROOTS"
    assert cfg.mode_kind == "action"
    assert cfg.entry_operation_ids == ["letta_ensure_agent", "letta_send_message"]
    assert cfg.require_query is True
    assert cfg.routable is True


def test_letta_agent_mode_has_higher_priority_than_letta() -> None:
    """letta-agent (priority 9) sorts before letta (priority 8) so the
    more-specific command_pattern wins in parse_route."""
    from kernel.ipsar_fsm import get_workflow_catalog

    catalog = get_workflow_catalog()
    letta_agent = catalog.get("letta-agent")
    letta = catalog.get("letta")
    assert letta_agent is not None and letta is not None
    assert letta_agent.priority > letta.priority


# ── Routing precedence ───────────────────────────────────────────────


def _parse_route_fresh(message: str) -> tuple[str, str]:
    """Call parse_route after clearing its compiled-pattern cache."""
    import kernel.ipsar as ipsar_mod

    ipsar_mod._mode_route_cache = None  # force rebuild with current catalog
    return ipsar_mod.parse_route(message)


def test_parse_route_letta_agent_verbs_route_to_letta_agent() -> None:
    """/letta-agent and /ask-letta-agent route to the action mode."""
    assert _parse_route_fresh("/letta-agent hello") == ("letta-agent", "hello")
    assert _parse_route_fresh("/ask-letta-agent hi") == ("letta-agent", "hi")
    assert _parse_route_fresh("letta-agent bare") == ("letta-agent", "bare")


def test_parse_route_letta_verbs_still_route_to_letta_chat() -> None:
    """/letta and /ask-letta keep routing to the chat mode (unchanged)."""
    assert _parse_route_fresh("/letta hello") == ("letta", "hello")
    assert _parse_route_fresh("/ask-letta hi") == ("letta", "hi")


def test_parse_route_bare_agent_routes_to_letta_chat() -> None:
    """Bare /agent <query> stays on the chat letta mode (less surprising).

    Both patterns match /agent; the existing letta mode wins by
    alphabetical insertion order (letta-agent < letta in sorted(iterdir)).
    This is the intended default — users who want the zero-model path
    type the explicit /letta-agent.
    """
    assert _parse_route_fresh("/agent hi") == ("letta", "hi")


def test_parse_route_letta_agent_no_query_returns_empty_query() -> None:
    """Bare /letta-agent matches the action mode with an empty query.

    The mode declares require_query: true, so a host should reject this
    before reaching run_turn; parse_route itself returns an empty string.
    """
    mode_id, query = _parse_route_fresh("/letta-agent")
    assert mode_id == "letta-agent"
    assert query == ""


# ── No-model-call execution path ─────────────────────────────────────
#
# Action-mode entry operations dispatch through the kernel's injected
# operation callback (kernel.ipsar_fsm._execute_operation_fn), NOT through
# hooks.execute_operation (which is only used for model-loop tool dispatch).
# These tests monkeypatch _execute_operation_fn to a stub so we can observe
# the ordered entry-op chain without needing a live Letta server.


@pytest.fixture()
def _stub_operation_callback() -> tuple[list[str], Any]:
    """Monkeypatch kernel.ipsar_fsm._execute_operation_fn for one test.

    Returns (executed_list, register_fn) where register_fn installs a stub
    execute_operation callback that records op ids into executed_list and
    dispatches to the caller-supplied mapping. Restores the original callback
    after the test.
    """
    import kernel.ipsar_fsm as ipsar_fsm

    original = ipsar_fsm._execute_operation_fn
    executed: list[str] = []

    def install(responses: dict[str, OperationResult]) -> None:
        async def _stub(operation_id: str, _ctx: OperationExecutionContext) -> OperationResult:
            executed.append(operation_id)
            return responses.get(
                operation_id,
                OperationResult(output_text=f"unexpected op: {operation_id}"),
            )

        ipsar_fsm._execute_operation_fn = _stub

    yield executed, install

    ipsar_fsm._execute_operation_fn = original


@pytest.mark.asyncio
async def test_letta_agent_turn_runs_entry_ops_without_model_call(
    _stub_operation_callback: tuple[list[str], Any],
) -> None:
    """A letta-agent turn runs the two entry ops and returns their joined
    output_text WITHOUT calling hooks.complete (no Lisa model round-trip).
    """
    executed, install = _stub_operation_callback
    install({
        "letta_ensure_agent": OperationResult(
            output_text="Letta agent resolved: agent-123",
            output_json={"agent_id": "agent-123", "created": False},
        ),
        "letta_send_message": OperationResult(
            output_text="Hello from the Letta agent.",
            output_json={"agent_id": "agent-123", "stop_reason": "stop"},
        ),
    })

    complete_calls: list[dict[str, Any]] = []

    async def complete_should_not_fire(**kwargs: Any) -> Any:
        complete_calls.append(kwargs)
        raise AssertionError("complete() must not be called in letta-agent action mode")

    hooks = TurnHooks(complete=complete_should_not_fire)

    result = await run_turn(mode_id="letta-agent", query="hello agent", hooks=hooks)

    # No model call happened.
    assert complete_calls == []
    # Both entry ops ran in order.
    assert executed == ["letta_ensure_agent", "letta_send_message"]
    # Action mode returns joined output_text and iteration_count=1.
    assert "Letta agent resolved" in result.response_text
    assert "Hello from the Letta agent." in result.response_text
    assert result.tool_call_count == 0
    assert result.iteration_count == 1


@pytest.mark.asyncio
async def test_letta_agent_turn_without_mapped_agent_returns_error_text(
    _stub_operation_callback: tuple[list[str], Any],
) -> None:
    """If ensure_agent reports no mapped agent, send_message's error text
    still reaches the user via the joined output_text — no model call."""
    _executed, install = _stub_operation_callback
    install({
        "letta_ensure_agent": OperationResult(
            output_text="⛔ Letta is not reachable at http://localhost:8283",
            output_json={"error": "connection_refused"},
        ),
        "letta_send_message": OperationResult(
            output_text="⛔ No Letta agent mapped for this identity.",
            output_json={"error": "no_mapped_agent"},
        ),
    })

    complete_calls: list[dict[str, Any]] = []

    async def complete_should_not_fire(**kwargs: Any) -> Any:
        complete_calls.append(kwargs)
        raise AssertionError("complete() must not be called in letta-agent action mode")

    hooks = TurnHooks(complete=complete_should_not_fire)

    result = await run_turn(mode_id="letta-agent", query="hello", hooks=hooks)

    assert complete_calls == []
    assert "not reachable" in result.response_text or "No Letta agent" in result.response_text
    assert result.tool_call_count == 0
    assert result.iteration_count == 1