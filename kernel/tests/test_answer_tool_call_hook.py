"""Tests for the ``ToolHooks.answer_tool_call`` execution-override seam.

The seam exists so a host can answer a tool dispatch instead of executing it
(turn replay answers from a recording). These tests pin the kernel's promise:
the hook's string is the tool output, no executor runs, ledger/receipt
tracking still happens, a raising hook surfaces like a tool error (a failed
dispatch outcome, not a propagated exception), and a ``None`` hook is
ordinary execution.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import CallLedger, OperationExecutionContext, OperationResult
from kernel.turn_phases._dispatch import dispatch_capability
from kernel.turn_phases._turn_state import _TurnState
from kernel.turn_runner import TurnHooks


class _ExplodingPool:
    """Pool double that fails the test if MCP dispatch is reached."""

    async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> Any:  # pragma: no cover
        raise AssertionError(f"unexpected MCP dispatch for {tool_id}")


async def _exploding_exec_op(_operation_id: str, _ctx: Any) -> OperationResult:  # pragma: no cover
    raise AssertionError("unexpected operation execution")


def _hooks(answer: Any) -> TurnHooks:
    hooks = TurnHooks()
    hooks.tools.answer_tool_call = answer
    return hooks


async def _dispatch(
    hooks: TurnHooks,
    state: _TurnState,
    ledger: CallLedger,
    *,
    operation_catalog: dict[str, Any] | None = None,
    tool_catalog: dict[str, Any] | None = None,
    exec_op: Any = None,
) -> Any:
    """Drive one dispatch with the verified ``dispatch_capability`` signature."""
    return await dispatch_capability(
        tool_name="test_tool",
        tool_arguments={"path": "a.py"},
        tool_kind="function",
        ctx=OperationExecutionContext(),
        state=state,
        ledger=ledger,
        pool=_ExplodingPool(),
        operation_catalog=operation_catalog or {},
        tool_catalog=tool_catalog or {"test_tool": SimpleNamespace(effect="read")},
        exec_op=exec_op or _exploding_exec_op,
        hooks=hooks,
        tool_call_id="call_recorded_1",
    )


@pytest.mark.asyncio
async def test_answerer_replaces_execution_and_returns_its_string() -> None:
    """The hook's return value is the tool output; no executor runs."""
    seen: list[tuple[str, dict[str, Any], str]] = []

    def answer(tool_name: str, arguments: dict[str, Any], tool_call_id: str) -> str:
        seen.append((tool_name, arguments, tool_call_id))
        return "recorded output"

    state = _TurnState()
    ledger = CallLedger()
    outcome = await _dispatch(_hooks(answer), state, ledger)

    assert seen == [("test_tool", {"path": "a.py"}, "call_recorded_1")]
    assert outcome.tool_output == "recorded output"
    assert outcome.is_ok is True
    assert outcome.operation_result is None
    assert outcome.tool_result is None
    assert ledger.counts == {"test_tool": 1}
    assert state.total_tool_calls == 1
    assert [r.capability_id for r in state.effect_receipts] == ["test_tool"]


@pytest.mark.asyncio
async def test_raising_answerer_surfaces_like_a_tool_error() -> None:
    """A raising hook becomes a failed dispatch outcome, not a propagated exception."""

    def answer(_tool_name: str, _arguments: dict[str, Any], _tool_call_id: str) -> str:
        raise RuntimeError("replay tool gap")

    state = _TurnState()
    ledger = CallLedger()
    outcome = await _dispatch(_hooks(answer), state, ledger)

    assert outcome.is_ok is False
    assert outcome.tool_output == "Tool execution failed: RuntimeError: replay tool gap"
    assert ledger.counts == {"test_tool": 1}
    assert state.total_tool_calls == 1


@pytest.mark.asyncio
async def test_none_answerer_executes_the_tool_normally() -> None:
    """Without the hook, the operation executes and its output reaches the model."""

    async def execute_operation(_operation_id: str, _ctx: Any) -> OperationResult:
        return OperationResult(output_text="live output")

    state = _TurnState()
    ledger = CallLedger()
    outcome = await _dispatch(
        TurnHooks(),
        state,
        ledger,
        operation_catalog={"test_tool": SimpleNamespace(effect="read")},
        tool_catalog={},
        exec_op=execute_operation,
    )

    assert outcome.tool_output == "live output"
    assert outcome.is_ok is True
    assert outcome.operation_result is not None
    assert outcome.operation_result.output_text == "live output"
    assert [r.capability_id for r in state.effect_receipts] == ["test_tool"]
