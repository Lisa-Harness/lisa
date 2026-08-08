"""Phase 8 (one verification primitive): receipt-ledger contracts.

``EffectReceipt`` is the universal ground truth for "did work actually
happen". Three behaviors pinned here:

  1. **Dispatch-layer enforcement** — ``_enforce_write_receipt``: a
     capability with declared effect ``write`` / ``external_write`` whose
     dispatch succeeded must have produced a receipt in that dispatch;
     a gap is a loud kernel error, never a silent miscount.
  2. **Parent-turn validation** — the fork-child completion contract
     applied to the parent: a mode with a side-effecting completion
     contract (``mode_completion_effects`` non-empty) that exits the
     loop with no receipt for its action tools is blocked as
     ``no_tools`` and its unverified prose is replaced with a
     deterministic failure message. Chat modes (no contract), yields,
     and sub-turns are exempt.
  3. **Control** — a clean exit in a write-contract mode WITH a receipt
     is not blocked; the receipt rides ``TurnResult.effect_receipts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from kernel.models import (
    EffectReceipt,
    OperationExecutionContext,
    OperationResult,
    WorkflowModeConfig,
)
from kernel.turn_runner import (
    TurnHooks,
    _enforce_write_receipt,
    run_turn,
)

# ── Dispatch-layer enforcement (unit) ─────────────────────────────────


def test_enforce_write_receipt_passes_when_receipt_recorded() -> None:
    """A successful write dispatch with a receipt appended → no error."""
    receipts = [EffectReceipt(effect="write", capability_id="write_file")]
    assert (
        _enforce_write_receipt(
            tool_name="write_file",
            dispatched_effect="write",
            dispatched_succeeded=True,
            receipts=receipts,
            receipts_before=0,
        )
        is None
    )


def test_enforce_write_receipt_ignores_reads_and_failures() -> None:
    """Read effects and failed dispatches are outside the invariant."""
    assert (
        _enforce_write_receipt(
            tool_name="fetch_page",
            dispatched_effect="read",
            dispatched_succeeded=True,
            receipts=[],
            receipts_before=0,
        )
        is None
    )
    assert (
        _enforce_write_receipt(
            tool_name="write_file",
            dispatched_effect="write",
            dispatched_succeeded=False,
            receipts=[],
            receipts_before=0,
        )
        is None
    )


def test_enforce_write_receipt_flags_successful_write_without_receipt() -> None:
    """The drift case: declared write + success + no receipt → kernel error."""
    marker = _enforce_write_receipt(
        tool_name="write_file",
        dispatched_effect="external_write",
        dispatched_succeeded=True,
        receipts=[],
        receipts_before=0,
    )
    assert marker is not None
    assert marker.startswith("KERNEL ERROR")
    assert "write_file" in marker


def test_enforce_write_receipt_only_counts_this_dispatch() -> None:
    """A receipt from an earlier dispatch does not satisfy this one."""
    receipts = [EffectReceipt(effect="write", capability_id="write_file")]
    marker = _enforce_write_receipt(
        tool_name="write_file",
        dispatched_effect="write",
        dispatched_succeeded=True,
        receipts=receipts,
        receipts_before=1,  # the existing receipt predates this dispatch
    )
    assert marker is not None


# ── Parent-turn validation (scripted turns) ───────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    """Fake model completion for tests (mirrors test_turn_runner)."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _write_contract_catalog() -> dict[str, WorkflowModeConfig]:
    """A chat mode with an authored write-effect binding.

    ``mode_completion_effects`` includes authored operation ids even when
    the op is unknown to the catalog ("can't prove it's read-only"), so
    the contract is non-empty regardless of the fixture op registry.
    """
    return {
        "writer": WorkflowModeConfig(
            mode_id="writer",
            start_stage_id="none",
            mode_kind="chat",
            routable=True,
            authored_allowed_operation_ids=["write_file"],
            allowed_operation_ids=["write_file"],
        ),
    }


@pytest.mark.asyncio
async def test_parent_turn_no_receipt_blocked_as_no_tools() -> None:
    """Parent-turn analog of the 2026-08-06 remember-child incident.

    A mode with a write contract whose model reports success without a
    single tool call is blocked as ``no_tools``; the unverified prose is
    replaced with a deterministic failure message.
    """

    async def hallucinating(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done — I wrote the file for you.")

    hooks = TurnHooks(complete=hallucinating)
    with (
        patch(
            "kernel.turn_runner.get_workflow_catalog",
            return_value=_write_contract_catalog(),
        ),
        patch(
            "kernel.operations.registry.get_workflow_catalog",
            return_value=_write_contract_catalog(),
            create=True,
        ),
        patch(
            "kernel.ipsar_fsm.get_workflow_catalog",
            return_value=_write_contract_catalog(),
        ),
    ):
        result = await run_turn(mode_id="writer", query="write the file", hooks=hooks)

    assert result.blocked is True
    assert result.block_origin == "no_tools"
    assert "unverified" in result.response_text
    assert "I wrote the file" not in result.response_text


@pytest.mark.asyncio
async def test_parent_turn_chat_mode_without_contract_is_exempt() -> None:
    """Chat modes have no completion contract — a clean exit is complete."""

    async def chat(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Just a chat answer.")

    hooks = TurnHooks(complete=chat)
    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.blocked is False
    assert result.response_text == "Just a chat answer."


@pytest.mark.asyncio
async def test_parent_turn_with_receipt_is_not_blocked() -> None:
    """A write-contract mode whose write dispatched (receipt) completes."""

    async def stub_execute_operation(
        operation_id: str, _ctx: OperationExecutionContext
    ) -> OperationResult:
        return OperationResult(
            output_text=f"Executed {operation_id}",
            output_json={"artifact_id": "doc-1"},
        )

    fake_write_op = SimpleNamespace(
        effect="write",
        mode_allowlist=(),
        description="write a file",
        parameters_schema={"type": "object", "properties": {}},
        max_calls_per_turn=0,
        effective_provider_id=lambda: "test",
    )

    call_count = 0

    async def writing(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "a.md", "content": "x"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Wrote it.")

    hooks = TurnHooks(complete=writing, execute_operation=stub_execute_operation)
    with (
        patch(
            "kernel.turn_runner.get_workflow_catalog",
            return_value=_write_contract_catalog(),
        ),
        patch(
            "kernel.turn_runner.get_operation_catalog",
            return_value={"write_file": fake_write_op},
        ),
        patch(
            "kernel.ipsar_fsm.get_workflow_catalog",
            return_value=_write_contract_catalog(),
        ),
    ):
        result = await run_turn(mode_id="writer", query="write the file", hooks=hooks)

    # ``write_file`` resolved via the stubbed operation dispatch and
    # produced a receipt; the completion contract is satisfied.
    assert any(r.capability_id == "write_file" for r in result.effect_receipts)
    assert result.block_origin != "no_tools"
