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
from kernel.turn_hooks import ModelHooks, OperationHooks, ToolHooks
from kernel.turn_phases._dispatch import enforce_write_receipt as _enforce_write_receipt
from kernel.turn_runner import (
    TurnHooks,
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
    the op is unknown to the catalog (operations are in-process; an
    unresolvable op id is assumed write under the default cutover), so the
    contract is non-empty regardless of the fixture op registry.
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

    hooks = TurnHooks(
        models=ModelHooks(
            complete=hallucinating,
        ),
    )
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

    hooks = TurnHooks(
        models=ModelHooks(
            complete=chat,
        ),
    )
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

    hooks = TurnHooks(
        models=ModelHooks(
            complete=writing,
        ),
        operations=OperationHooks(
            execute_operation=stub_execute_operation,
        ),
    )
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


@pytest.mark.asyncio
async def test_degraded_write_operation_produces_no_receipt() -> None:
    """A write op returning ``status="degraded"`` does NOT produce a receipt.

    The ``OperationResult.status`` contract lets an operation signal a soft
    failure (truncated, empty, persisted-with-warning) without raising. The
    turn runner must not record an ``EffectReceipt`` for a degraded write,
    so the completion contract treats it as an unverified write — a
    hallucinated "I wrote it" cannot slip through as success.
    """

    async def stub_execute_operation(
        operation_id: str, _ctx: OperationExecutionContext
    ) -> OperationResult:
        return OperationResult(
            output_text=f"Executed {operation_id} (truncated).",
            output_json={"artifact_id": "doc-1"},
            status="degraded",
            error_detail="continuation exhausted: 3 empty passes",
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

    hooks = TurnHooks(
        models=ModelHooks(
            complete=writing,
        ),
        operations=OperationHooks(
            execute_operation=stub_execute_operation,
        ),
    )
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

    # A degraded write must NOT produce a receipt — the completion contract
    # is not satisfied by a truncated/partial result.
    assert not any(r.capability_id == "write_file" for r in result.effect_receipts)


@pytest.mark.asyncio
async def test_parent_turn_failed_mcp_write_still_blocked_as_no_tools() -> None:
    """A failed MCP write must not satisfy the completion contract.

    Regression for the 2026-08-07 review finding: the parent guard used
    to key on ``total_tool_calls == 0``, but a failed MCP dispatch still
    increments that counter (the server returns ``is_error=True`` without
    raising). A write-contract mode whose only write attempt failed must
    still be blocked as ``no_tools`` — the receipt ledger is empty.
    """
    from kernel.tools.mcp_client import MCPToolResult

    class _FailingWritePool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(
                tool_id=tool_id,
                output_text="provider unavailable",
                is_error=True,
            )

        def tool_spec(self, tool_id: str) -> Any:
            return None

        async def __aenter__(self) -> _FailingWritePool:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    async def claims_write(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "lisa-storage.write_file",
                        "arguments": '{"path": "a.md", "content": "x"}',
                    },
                }
            ],
        )

    async def then_claims_done(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done — I wrote the file for you.")

    call_count = 0

    async def model(**_kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return await claims_write()
        return await then_claims_done()

    hooks = TurnHooks(
        models=ModelHooks(
            complete=model,
        ),
        tools=ToolHooks(
            mcp_tool_pool=_FailingWritePool(),
        ),
    )
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

    # The write was attempted (total_tool_calls == 1) but failed; the
    # receipt ledger is empty, so the turn is blocked as no_tools.
    assert result.tool_call_count == 1
    assert result.blocked is True
    assert result.block_origin == "no_tools"
    assert "unverified" in result.response_text
    assert "I wrote the file" not in result.response_text


@pytest.mark.asyncio
async def test_parent_turn_dense_content_with_readonly_work_accepted() -> None:
    """Dense content after read-only work is accepted for chat modes.

    Regression for the 2026-08-14 work-gate: a ``chat``-kind mode with a
    write contract whose model dispatched a read-only tool (so
    ``total_tool_calls > 0``) and then produced dense content without a
    write receipt is the honest case the softening exists to accept —
    the model did real investigation but failed to land the write. The
    turn must NOT be blocked as ``no_tools``.
    """
    from kernel.tools.mcp_client import MCPToolResult

    class _ReadOnlyPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(tool_id=tool_id, output_text="read result")

        def tool_spec(self, tool_id: str) -> Any:
            return SimpleNamespace(read_only_hint=True)

        async def __aenter__(self) -> _ReadOnlyPool:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    call_count = 0

    async def model(**_kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "lisa-storage.read_file",
                            "arguments": '{"path": "a.md"}',
                        },
                    }
                ],
            )
        # Second completion: dense prose but no write tool call.
        return _StubCompletion(
            content=(
                "I investigated the file and gathered the context. The module "
                "exposes three entry points and the call graph confirms the "
                "boundary holds. Here is the synthesis of what I found across "
                "the codebase and the web — the analysis is complete."
            )
        )

    hooks = TurnHooks(
        models=ModelHooks(
            complete=model,
        ),
        tools=ToolHooks(
            mcp_tool_pool=_ReadOnlyPool(),
        ),
    )
    # The mode has a real write contract (``write_file``) AND a read-only
    # tool. The model did read-only work but no write; a ``chat``-kind mode
    # that did real work and produced dense content is accepted, not blocked.
    catalog = {
        "writer": WorkflowModeConfig(
            mode_id="writer",
            start_stage_id="none",
            mode_kind="chat",
            routable=True,
            authored_allowed_operation_ids=["write_file"],
            allowed_operation_ids=["write_file"],
            authored_allowed_tool_ids=["lisa-storage.read_file"],
            allowed_tool_ids=["lisa-storage.read_file"],
        ),
    }
    # Lower the dense-content threshold for this test so the short prose
    # fixture qualifies as dense; the production default (1000) is
    # exercised by the zero-tool-calls test below.
    from kernel import config

    mp = pytest.MonkeyPatch()
    mp.setattr(config.settings, "chat_mode_dense_content_min_chars", 100)
    try:
        with (
            patch(
                "kernel.turn_runner.get_workflow_catalog",
                return_value=catalog,
            ),
            patch(
                "kernel.operations.registry.get_workflow_catalog",
                return_value=catalog,
                create=True,
            ),
            patch(
                "kernel.ipsar_fsm.get_workflow_catalog",
                return_value=catalog,
            ),
        ):
            result = await run_turn(mode_id="writer", query="investigate", hooks=hooks)
    finally:
        mp.undo()

    assert result.blocked is False
    assert result.block_origin == ""
    # The dense prose survived (not replaced by the no_tools failure text).
    assert "investigated the file" in result.response_text


@pytest.mark.asyncio
async def test_parent_turn_failed_write_attempt_action_mode_still_blocked() -> None:
    """A failed write attempt keeps the strict block for any mode kind.

    Regression for the 2026-08-14 pack-leak removal, updated by the
    2026-08-27 read-receipt fix: the kernel must not infer "this mode's
    writes land in the thread" from tool-name prefixes — only
    ``mode_kind`` is consulted for softening, and a write ATTEMPT is
    never softened (the model may claim success on the failed write).
    An ``action``-kind mode whose model attempted ``write_file`` (the
    dispatch failed, no receipt) after read-only work is still blocked
    as ``no_tools`` — reads do not rescue a failed write.
    """
    from kernel.tools.mcp_client import MCPToolResult

    class _FailingWritePool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> MCPToolResult:
            # ``write_file`` fails; the read tool is not called.
            return MCPToolResult(
                tool_id=tool_id,
                output_text="provider unavailable",
                is_error=True,
            )

        def tool_spec(self, tool_id: str) -> Any:
            return None

        async def __aenter__(self) -> _FailingWritePool:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    call_count = 0

    async def model(**_kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "lisa-storage.write_file",
                            "arguments": '{"path": "a.md", "content": "x"}',
                        },
                    }
                ],
            )
        return _StubCompletion(
            content=(
                "I investigated the file and gathered the context. The module "
                "exposes three entry points and the call graph confirms the "
                "boundary holds. Here is the synthesis of what I found."
            )
        )

    hooks = TurnHooks(
        models=ModelHooks(
            complete=model,
        ),
        tools=ToolHooks(
            mcp_tool_pool=_FailingWritePool(),
        ),
    )
    # Same read-tool surface as the accepted case, but the model
    # ATTEMPTED the write (``write_file``) and the dispatch failed: the
    # strict block applies regardless of mode kind, content density, or
    # the read receipts on the ledger.
    catalog = {
        "runner": WorkflowModeConfig(
            mode_id="runner",
            start_stage_id="none",
            mode_kind="action",
            routable=True,
            authored_allowed_operation_ids=["write_file"],
            allowed_operation_ids=["write_file"],
            authored_allowed_tool_ids=["lisa-storage.read_file"],
            allowed_tool_ids=["lisa-storage.read_file"],
        ),
    }
    from kernel import config

    mp = pytest.MonkeyPatch()
    mp.setattr(config.settings, "chat_mode_dense_content_min_chars", 100)
    try:
        with (
            patch(
                "kernel.turn_runner.get_workflow_catalog",
                return_value=catalog,
            ),
            patch(
                "kernel.operations.registry.get_workflow_catalog",
                return_value=catalog,
                create=True,
            ),
            patch(
                "kernel.ipsar_fsm.get_workflow_catalog",
                return_value=catalog,
            ),
        ):
            result = await run_turn(mode_id="runner", query="investigate", hooks=hooks)
    finally:
        mp.undo()

    assert result.blocked is True
    assert result.block_origin == "no_tools"
    assert "unverified" in result.response_text


@pytest.mark.asyncio
async def test_parent_turn_dense_content_zero_tool_calls_blocked() -> None:
    """Zero-tool dense content is blocked (the 2026-08-14 proceed-loop).

    A write-contract mode whose model made ZERO tool calls and only
    emitted "I'll start the research once you confirm" prose is a
    hallucinated success — the model announced work it never started.
    Dense content must NOT soften the block when no tool was dispatched.
    This is the proceed-loop regression: the researcher repeatedly said
    it would begin, the dense-content acceptance treated that as
    success, and the turn ended with zero work done.
    """
    # Announcement prose well under the 1000-char dense threshold,
    # matching the "I'll start once you say ok" pattern from the
    # incident logs. It must be blocked by the work-gate (zero tool
    # dispatches), not accepted as dense content.
    announcement = (
        "I'll start the research now. Once you confirm I'll gather the "
        "context across the web and codebase, synthesize the findings, "
        "and write them to a markdown document for you to review."
    )

    async def announcing(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content=announcement)

    hooks = TurnHooks(
        models=ModelHooks(
            complete=announcing,
        ),
    )
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
        result = await run_turn(mode_id="writer", query="do the work", hooks=hooks)

    assert result.blocked is True
    assert result.block_origin == "no_tools"
    assert "unverified" in result.response_text
    assert "I'll start the research" not in result.response_text


@pytest.mark.asyncio
async def test_parent_turn_zero_tool_calls_backstop_synthesizes_dispatch() -> None:
    """Zero-tool prose on a chat write-contract mode triggers the backstop.

    Regression for the 2026-08-24 cloudservers stall: a ``chat``-kind mode
    with a side-effecting completion contract whose model returned prose
    with ZERO tool calls on its first iteration used to block as
    ``no_tools`` — the chat-mode softening cannot rescue it because every
    softening branch is gated on ``_did_work`` (the 2026-08-14 proceed-loop
    guard, intentionally strict on zero-tool turns). The write-contract
    empty-completion backstop arms ``forced_work_tool`` with the mode's
    first action tool and synthesizes a real dispatch instead of accepting
    the hallucinated prose. After the synth dispatch the turn has
    ``total_tool_calls > 0`` and the chat-mode softening can accept the
    follow-up response (or the failed-write block fires honestly).
    """
    from kernel.tools.mcp_client import MCPToolResult

    class _Pool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(tool_id=tool_id, output_text="tool ran")

        def tool_spec(self, tool_id: str) -> Any:
            return SimpleNamespace(read_only_hint=True)

        async def __aenter__(self) -> _Pool:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    call_count = 0

    async def model(**_kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Zero tool calls: announcement prose. The backstop must arm.
            return _StubCompletion(content="I'll check the servers for you now.")
        # Second completion (after the synthesized dispatch returned a
        # tool result): dense prose that the chat-mode softening accepts.
        return _StubCompletion(
            content=(
                "I checked the servers and gathered the status. The bind "
                "VM is running and the bind-tls VM is stopped. Here is the "
                "synthesis of the current state across the fleet — done."
            )
        )

    hooks = TurnHooks(
        models=ModelHooks(
            complete=model,
        ),
        tools=ToolHooks(
            mcp_tool_pool=_Pool(),
        ),
    )
    catalog = {
        "writer": WorkflowModeConfig(
            mode_id="writer",
            start_stage_id="none",
            mode_kind="chat",
            routable=True,
            authored_allowed_operation_ids=["write_file"],
            allowed_operation_ids=["write_file"],
        ),
    }
    from kernel import config

    mp = pytest.MonkeyPatch()
    mp.setattr(config.settings, "chat_mode_dense_content_min_chars", 100)
    try:
        with (
            patch(
                "kernel.turn_runner.get_workflow_catalog",
                return_value=catalog,
            ),
            patch(
                "kernel.operations.registry.get_workflow_catalog",
                return_value=catalog,
                create=True,
            ),
            patch(
                "kernel.ipsar_fsm.get_workflow_catalog",
                return_value=catalog,
            ),
        ):
            result = await run_turn(mode_id="writer", query="check the servers", hooks=hooks)
    finally:
        mp.undo()

    # The backstop synthesized a dispatch (call_count > 1) and the
    # follow-up dense prose was accepted by the chat-mode softening —
    # NOT blocked as no_tools.
    assert call_count >= 2, "backstop did not synthesize a dispatch"
    assert result.blocked is False
    assert result.block_origin == ""
    assert "checked the servers" in result.response_text
    # A real tool call was dispatched (the synth counts as a tool call).
    assert result.tool_call_count >= 1


# ── Regression: read-only MCP tool must not emit a write receipt ──────


@pytest.mark.asyncio
async def test_read_only_mcp_tool_does_not_satisfy_write_contract() -> None:
    """A server-declared read-only MCP tool is not a completion requirement.

    Regression for the 2026-08-07 recall-block: the old ``mode_completion_effects``
    "include unknown" fallback added every unresolvable authored tool to the
    mode's write contract, so a read-only MCP alias (``memory.search`` /
    ``lisa-storage.read_file``) polluted the action set and blocked legitimate
    recall-only turns as "hallucinated success". The resolver now consults the
    pool's ``read_only_hint``: a server-declared read-only MCP tool is excluded
    from the contract, so a turn that only calls it has an empty write contract
    and completes cleanly (no false ``no_tools`` block).
    """
    from kernel.tools.mcp_client import MCPToolResult

    class _ReadOnlyPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(tool_id=tool_id, output_text="read result")

        def tool_spec(self, tool_id: str) -> Any:
            return SimpleNamespace(read_only_hint=True)

        async def __aenter__(self) -> _ReadOnlyPool:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    async def reads_then_claims(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "lisa-storage.read_file",
                        "arguments": '{"path": "a.md"}',
                    },
                }
            ],
        )

    async def then_claims_done(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done — I read the file for you.")

    call_count = 0

    async def model(**_kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return await reads_then_claims()
        return await then_claims_done()

    hooks = TurnHooks(
        models=ModelHooks(
            complete=model,
        ),
        tools=ToolHooks(
            mcp_tool_pool=_ReadOnlyPool(),
        ),
    )
    # ``read_file`` is an unresolvable authored tool (not in the in-process
    # catalog). With the pool declaring ``read_only_hint=True``, the resolver
    # excludes it from the write contract — the mode has NO write-contract
    # tools, so the turn completes cleanly instead of being blocked.
    read_only_catalog = {
        "writer": WorkflowModeConfig(
            mode_id="writer",
            start_stage_id="none",
            mode_kind="chat",
            routable=True,
            authored_allowed_tool_ids=["read_file"],
            allowed_tool_ids=["read_file"],
        ),
    }
    with (
        patch(
            "kernel.turn_runner.get_workflow_catalog",
            return_value=read_only_catalog,
        ),
        patch(
            "kernel.operations.registry.get_workflow_catalog",
            return_value=read_only_catalog,
            create=True,
        ),
        patch(
            "kernel.ipsar_fsm.get_workflow_catalog",
            return_value=read_only_catalog,
        ),
    ):
        result = await run_turn(mode_id="writer", query="read the file", hooks=hooks)

    # The read-only MCP tool emits a READ receipt (never a write one —
    # the 2026-08-07 over-count fix), and because it is excluded from the
    # (now empty) write contract, the turn is NOT blocked.
    assert result.blocked is False
    assert result.block_origin == ""
    read_receipts = [
        r for r in result.effect_receipts if r.capability_id == "lisa-storage.read_file"
    ]
    assert read_receipts, "read-only MCP tool produced no receipt"
    assert all(r.effect == "read" for r in read_receipts)


# ── Resolver: op id mislisted as a tool id (2026-09-07 shell incident) ─


def _shell_incident_catalog() -> dict[str, WorkflowModeConfig]:
    """The shell mode as authored in the 2026-09-07 bash pack.

    ``allowed_tool_ids`` listed ``layers_list`` — a read-effect OPERATION
    from the bwrap pack — alongside the real ``bash_run`` tool. With the
    tool catalog unaware of ``layers_list``, the resolver's conservative
    write default gave shell a phantom write contract.
    """
    return {
        "shell": WorkflowModeConfig(
            mode_id="shell",
            start_stage_id="none",
            mode_kind="action",
            routable=True,
            authored_allowed_tool_ids=["bash_run", "layers_list"],
            allowed_tool_ids=["bash_run", "layers_list"],
        ),
    }


def test_read_effect_operation_mislisted_as_tool_excluded_from_contract() -> None:
    """A read-effect operation listed under allowed_tool_ids is not a write contract.

    Regression for the 2026-09-07 shell block (turns 114/116/118): the bash
    pack's shell mode listed the bwrap pack's read-only ``layers_list``
    OPERATION in ``allowed_tool_ids``. Operations and tools live in separate
    catalogs, so the tool-catalog lookup missed and the conservative write
    default classified the read op as a required write. Every summarize-only
    shell turn was then blocked as a hallucinated success, and the
    write-contract backstop synthesized a doomed ``layers_list`` dispatch on
    top. The resolver must consult the operation catalog first: an id that
    names a resolvable read/execute operation can never be a
    write-completion requirement.
    """
    from kernel.models import OperationDefinition
    from kernel.operations import registry as ops_registry

    op_catalog = {
        "layers_list": OperationDefinition(
            operation_id="layers_list",
            description="list overlayfs layers",
            effect="read",
        ),
    }
    # The tool catalog does NOT know ``layers_list`` (it is an operation);
    # ``bash_run`` resolves as an execute-effect tool.
    tool_catalog: dict[str, Any] = {}

    with (
        patch.object(ops_registry, "get_operation_catalog", return_value=op_catalog),
        patch("kernel.tools.registry.get_tool_catalog", return_value=tool_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=_shell_incident_catalog()),
    ):
        effects = ops_registry.mode_completion_effects("shell")

    assert "layers_list" not in effects, (
        "read-effect operation mislisted as a tool leaked into the write contract"
    )
    # ``bash_run`` is also unresolvable in this minimal fixture (it lives in
    # the MCP tool catalog in production), so the conservative default keeps
    # it as the mode's real completion requirement.
    assert "bash_run" in effects


def test_write_effect_operation_mislisted_as_tool_stays_in_contract() -> None:
    """The exclusion is effect-scoped: a WRITE-effect op stays a requirement.

    Guarding the guard: the 1b pass must only demote read/execute
    capabilities. An operation with declared ``write`` effect that a mode
    mislists under ``allowed_tool_ids`` keeps its completion requirement —
    otherwise a mislisted write op would silently drop out of the contract
    and unverified-success turns would pass.
    """
    from kernel.models import OperationDefinition
    from kernel.operations import registry as ops_registry

    op_catalog = {
        "remember_store": OperationDefinition(
            operation_id="remember_store",
            description="store a memory record",
            effect="write",
        ),
    }
    catalog = {
        "remember": WorkflowModeConfig(
            mode_id="remember",
            start_stage_id="none",
            mode_kind="action",
            routable=True,
            authored_allowed_tool_ids=["remember_store"],
            allowed_tool_ids=["remember_store"],
        ),
    }
    with (
        patch.object(ops_registry, "get_operation_catalog", return_value=op_catalog),
        patch("kernel.tools.registry.get_tool_catalog", return_value={}),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        effects = ops_registry.mode_completion_effects("remember")

    assert "remember_store" in effects


# ── Routing-misfire prose surfacing (2026-09-02 executor misroute) ────


def _misroute_tool_catalog() -> dict[str, Any]:
    """A tool catalog entry for ``coding_apply_patch`` that REQUIRES params.

    The synthesized-call required-params guard consults this definition;
    without it the guard cannot detect the missing-params synthesis and
    the test exercises nothing. Mirrors the real ``apply_patch.tool.yml``
    shape (a write-effect tool whose params are user-supplied).
    """
    from kernel.models import ToolDefinition
    from pydantic import BaseModel

    class _ApplyPatchParams(BaseModel):
        path: str
        patch: str

    return {
        "coding_apply_patch": ToolDefinition(
            tool_id="coding_apply_patch",
            description="apply a bounded patch",
            parameters_schema=_ApplyPatchParams,
        ),
    }


@pytest.mark.asyncio
async def test_routing_misfire_zero_dispatch_prose_surfaced_not_blocked() -> None:
    """A conversational misroute into a write-contract mode keeps its prose.

    Regression for the 2026-09-02 turn-70 executor block: the user asked a
    conversational question, routing put it in front of ``executor`` (a
    write-contract mode), the model answered entirely in prose with zero
    tool calls, the write-contract empty-completion backstop synthesized
    ``coding_apply_patch`` with ``{}``, and the synthesized-call
    required-params guard aborted the dispatch ("surfacing model prose").
    But the post-loop ``_parent_no_tools`` check then REPLACED that prose
    with the no-verified-write block — directly contradicting the guard's
    contract. The kernel must surface the model's prose with the no-write
    warning footnote instead: answering a question is honest work even
    though the receipt ledger shows zero tool calls.
    """
    from kernel.tools.mcp_client import MCPToolResult

    from kernel import config

    class _EmptyPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(tool_id=tool_id, output_text="tool ran")

        def tool_spec(self, tool_id: str) -> Any:
            return None

        async def __aenter__(self) -> _EmptyPool:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    prose = (
        "Reviewing the two files you already have: the miner covers conn, "
        "dns, http, ssl, ssh, notice, and weird logs with gzip support, and "
        "the quickref walks through the zeek policy replacement. To extend "
        "them I would add a daily report wrapper and a new-since-yesterday "
        "diff, but tell me which direction you prefer."
    )

    async def prose_only(**_kwargs: Any) -> _StubCompletion:
        # Zero tool calls, zero receipts: pure conversational prose from a
        # write-contract mode the routing chose.
        return _StubCompletion(content=prose)

    hooks = TurnHooks(
        models=ModelHooks(
            complete=prose_only,
        ),
        tools=ToolHooks(
            mcp_tool_pool=_EmptyPool(),
        ),
    )
    catalog = {
        "executor": WorkflowModeConfig(
            mode_id="executor",
            start_stage_id="none",
            mode_kind="chat",
            routable=True,
            authored_allowed_tool_ids=["coding_apply_patch"],
            allowed_tool_ids=["coding_apply_patch"],
        ),
    }
    mp = pytest.MonkeyPatch()
    mp.setattr(config.settings, "chat_mode_dense_content_min_chars", 1000)
    try:
        with (
            patch(
                "kernel.turn_runner.get_workflow_catalog",
                return_value=catalog,
            ),
            patch(
                "kernel.operations.registry.get_workflow_catalog",
                return_value=catalog,
                create=True,
            ),
            patch(
                "kernel.ipsar_fsm.get_workflow_catalog",
                return_value=catalog,
            ),
            patch(
                "kernel.turn_runner.get_tool_catalog",
                return_value=_misroute_tool_catalog(),
            ),
        ):
            result = await run_turn(mode_id="executor", query="what would you extend?", hooks=hooks)
    finally:
        mp.undo()

    # The prose is surfaced with the warning footnote — NOT replaced by
    # the no_tools failure text.
    assert result.block_origin != "no_tools"
    assert "no verified write" not in result.response_text
    assert "daily report wrapper" in result.response_text


@pytest.mark.asyncio
async def test_routing_misfire_empty_prose_yields_not_surfaces() -> None:
    """Empty prose on an aborted synth dispatch yields for redirect.

    When the synthesized-call guard aborts on a zero-dispatch turn but
    the model's completion carries NO usable text, there is nothing to
    surface. The empty-content post-loop path yields an honest redirect
    question ("model returned an empty response…") — the turn is NOT
    blocked as no_tools and NOT silently accepted.
    """
    from kernel.tools.mcp_client import MCPToolResult

    from kernel import config

    class _EmptyPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(tool_id=tool_id, output_text="tool ran")

        def tool_spec(self, tool_id: str) -> Any:
            return None

        async def __aenter__(self) -> _EmptyPool:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    async def empty_prose(**_kwargs: Any) -> _StubCompletion:
        # Zero tool calls AND zero content: nothing to surface.
        return _StubCompletion(content="")

    hooks = TurnHooks(
        models=ModelHooks(
            complete=empty_prose,
        ),
        tools=ToolHooks(
            mcp_tool_pool=_EmptyPool(),
        ),
    )
    catalog = {
        "executor": WorkflowModeConfig(
            mode_id="executor",
            start_stage_id="none",
            mode_kind="chat",
            routable=True,
            authored_allowed_tool_ids=["coding_apply_patch"],
            allowed_tool_ids=["coding_apply_patch"],
        ),
    }
    mp = pytest.MonkeyPatch()
    mp.setattr(config.settings, "chat_mode_dense_content_min_chars", 1000)
    try:
        with (
            patch(
                "kernel.turn_runner.get_workflow_catalog",
                return_value=catalog,
            ),
            patch(
                "kernel.operations.registry.get_workflow_catalog",
                return_value=catalog,
                create=True,
            ),
            patch(
                "kernel.ipsar_fsm.get_workflow_catalog",
                return_value=catalog,
            ),
            patch(
                "kernel.turn_runner.get_tool_catalog",
                return_value=_misroute_tool_catalog(),
            ),
        ):
            result = await run_turn(mode_id="executor", query="do the work", hooks=hooks)
    finally:
        mp.undo()

    # Nothing to surface: the empty-content path yields for redirect.
    assert result.blocked is False
    assert result.yielded is not None


# ── Parent-turn pre-cap write-nudge (2026-08-11) ──────────────────────


@pytest.mark.asyncio
async def test_parent_turn_pre_cap_write_nudge_fires_before_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent turn with unmet action tools gets a write-nudge one
    iteration before the stuck-in-stage cap would pause it.

    Regression for the 2026-08-10/11 ``no_tools`` blocks: capture modes
    (``researcher``/``reviewer``/``executor``) read-looped to the
    stuck-in-stage cap without ever calling their write tools, then ended
    blocked as ``no_tools`` (hallucinated success). The pre-cap nudge
    fires at ``stage_nudge_cap - write_deadline_margin`` to give the
    model a final chance to call a write tool before the cap freezes the
    turn.

    The stub model always calls a read-only tool (``fetch_page``) and
    never the write tool, so ``turns_in_stage`` climbs to the cap. The
    nudge prose ("zero writes") must appear in the system prompt at the
    pre-cap iteration.
    """
    from kernel import config

    # Tighten the caps so the test runs in a few iterations, not 300.
    # ``stage_nudge_cap=4`` + ``write_deadline_margin=1`` → nudge fires
    # at ``turns_in_stage == 3`` (the 4th iteration). The global
    # iteration cap stays well above the stage cap so the stage cap is
    # what binds.
    monkeypatch.setattr(config.settings, "stage_nudge_cap", 4)
    monkeypatch.setattr(config.settings, "write_deadline_margin", 1)
    monkeypatch.setattr(config.settings, "max_tool_iterations", 50)

    captured_prompts: list[str] = []

    async def stub_execute_operation(
        operation_id: str, _ctx: OperationExecutionContext
    ) -> OperationResult:
        return OperationResult(output_text=f"Executed {operation_id}")

    async def read_loop(**kwargs: Any) -> _StubCompletion:
        # Capture the system prompt each iteration to inspect the nudge.
        messages = kwargs.get("messages", [])
        if messages and isinstance(messages, list):
            sys_msg = str(messages[0].get("content", "")) if messages[0] else ""
            captured_prompts.append(sys_msg)
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

    hooks = TurnHooks(
        models=ModelHooks(
            complete=read_loop,
        ),
        operations=OperationHooks(
            execute_operation=stub_execute_operation,
        ),
    )
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
        await run_turn(mode_id="writer", query="write the file", hooks=hooks)

    # The nudge fires exactly once at the pre-cap iteration. Assert the
    # nudge prose ("zero writes") appears in at least one captured
    # system prompt, and that it fired before the cap paused the turn.
    nudge_iterations = [i for i, p in enumerate(captured_prompts) if "zero writes" in p]
    assert nudge_iterations, (
        "parent-turn pre-cap write-nudge did not fire; captured prompts "
        f"did not contain 'zero writes'. prompts={captured_prompts!r}"
    )


@pytest.mark.asyncio
async def test_parent_turn_write_nudge_disabled_when_margin_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``write_deadline_margin=0`` disables the parent-turn pre-cap
    nudge — the turn ends at the stuck-in-stage cap with no nudge
    prose in the system prompt."""
    from kernel import config

    monkeypatch.setattr(config.settings, "stage_nudge_cap", 3)
    monkeypatch.setattr(config.settings, "write_deadline_margin", 0)
    monkeypatch.setattr(config.settings, "max_tool_iterations", 50)

    captured_prompts: list[str] = []

    async def stub_execute_operation(
        operation_id: str, _ctx: OperationExecutionContext
    ) -> OperationResult:
        return OperationResult(output_text=f"Executed {operation_id}")

    async def read_loop(**kwargs: Any) -> _StubCompletion:
        messages = kwargs.get("messages", [])
        if messages and isinstance(messages, list):
            sys_msg = str(messages[0].get("content", "")) if messages[0] else ""
            captured_prompts.append(sys_msg)
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

    hooks = TurnHooks(
        models=ModelHooks(
            complete=read_loop,
        ),
        operations=OperationHooks(
            execute_operation=stub_execute_operation,
        ),
    )
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
        await run_turn(mode_id="writer", query="write the file", hooks=hooks)

    assert not any("zero writes" in p for p in captured_prompts), (
        "parent-turn write-nudge fired despite write_deadline_margin=0"
    )


# ── needs_guidance backstops (2026-08-15) ─────────────────────────────


def _needs_guidance_model() -> Any:
    """A stub model entry with ``needs_guidance=True``."""

    @dataclass(frozen=True, slots=True)
    class _GuidanceModel:
        model_id: str = "weak-model"
        reasoning_model: bool = False
        thinking_model: bool = False
        cost_tier: str = "standard"
        structured_reliability: float = 0.0
        needs_guidance: bool = True
        capabilities: tuple = ()
        provider: str = "test"

    return _GuidanceModel()


def _strong_model() -> Any:
    """A stub model entry with ``needs_guidance=False`` (default)."""

    @dataclass(frozen=True, slots=True)
    class _StrongModel:
        model_id: str = "strong-model"
        reasoning_model: bool = True
        thinking_model: bool = False
        cost_tier: str = "frontier"
        structured_reliability: float = 1.0
        needs_guidance: bool = False
        capabilities: tuple = ()
        provider: str = "test"

    return _StrongModel()


@pytest.mark.asyncio
async def test_midpoint_nudge_fires_for_guidance_model_gather_without_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The midpoint write-nudge fires for a guidance-needing model that
    gathered inputs (read-effect tools) but never called its write-effect
    output tool.

    Regression for the 2026-08-15 deepseek cap-pause: the model called
    ``web_search`` / ``fetch_page`` (read-effect) repeatedly but never
    ``markdown_forge_document`` (write-effect). The midpoint nudge (gated
    on zero write-action receipts) fires at the write-deadline margin —
    the model has read receipts, not write receipts, so the gate passes.
    The ``needs_guidance`` flag does not gate the midpoint nudge (it
    fires for all models), but it arms the forced-work backstop that
    synthesizes the write call if the model ignores the nudge.
    """
    from kernel import config

    monkeypatch.setattr(config.settings, "stage_nudge_cap", 4)
    monkeypatch.setattr(config.settings, "write_deadline_margin", 1)
    monkeypatch.setattr(config.settings, "max_tool_iterations", 50)

    captured_prompts: list[str] = []

    async def stub_execute_operation(
        operation_id: str, _ctx: OperationExecutionContext
    ) -> OperationResult:
        return OperationResult(output_text=f"Executed {operation_id}")

    call_count = 0

    async def gather_loop(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        messages = kwargs.get("messages", [])
        if messages and isinstance(messages, list):
            sys_msg = str(messages[0].get("content", "")) if messages[0] else ""
            captured_prompts.append(sys_msg)
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"call_{call_count}",
                    "function": {
                        "name": "fetch_page",
                        "arguments": '{"url": "https://example.com"}',
                    },
                }
            ],
        )

    def resolve_model(_model_id: str) -> Any:
        return _needs_guidance_model()

    hooks = TurnHooks(
        models=ModelHooks(
            complete=gather_loop,
        ),
        operations=OperationHooks(
            execute_operation=stub_execute_operation,
        ),
    )
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
        patch(
            "kernel.turn_runner._default_resolve_model",
            side_effect=resolve_model,
        ),
    ):
        await run_turn(
            mode_id="writer",
            query="write the file",
            hooks=hooks,
            preferred_model_id="weak-model",
        )

    # The midpoint nudge ("zero writes") fires — the model did read work
    # but has zero write receipts. This is the existing nudge; the
    # ``needs_guidance`` flag does not change it.
    assert any("zero writes" in p for p in captured_prompts), (
        "midpoint write-nudge did not fire for a gather-without-output turn. "
        f"prompts={captured_prompts!r}"
    )


# ── Stalled-write rescue (the 2026-09-02 write-announcement loop) ──────


def _announcement_writer_catalog() -> dict[str, WorkflowModeConfig]:
    """A chat mode with a write contract and an ambient read tool.

    Mirrors the production ``default`` mode: ``lisa-storage.read_file`` is
    ambient (reads work) and ``write_file`` is the action tool the model
    keeps announcing but never calling.
    """
    return {
        "writer": WorkflowModeConfig(
            mode_id="writer",
            start_stage_id="none",
            mode_kind="chat",
            routable=True,
            authored_allowed_operation_ids=["write_file"],
            allowed_operation_ids=["write_file"],
            authored_allowed_tool_ids=["lisa-storage.read_file"],
            allowed_tool_ids=["lisa-storage.read_file"],
        ),
    }


class _ReadThenWritePool:
    """MCP pool stub: reads succeed; writes record their payload."""

    def __init__(self) -> None:
        self.write_calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
        return []

    async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> Any:
        from kernel.tools.mcp_client import MCPToolResult

        if "write" in tool_id or "append" in tool_id:
            self.write_calls.append((tool_id, dict(arguments)))
            return MCPToolResult(
                tool_id=tool_id,
                output_text='{"ok": true}',
                is_error=False,
            )
        return MCPToolResult(tool_id=tool_id, output_text="read result", is_error=False)

    def tool_spec(self, tool_id: str) -> Any:
        # read_file is server-declared read-only; write tools are not.
        return SimpleNamespace(read_only_hint=("write" not in tool_id and "append" not in tool_id))

    async def __aenter__(self) -> _ReadThenWritePool:
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_stalled_write_rescue_directs_announced_write_to_dispatch() -> None:
    """Announced-but-unwritten prose after read work injects a one-shot
    dispatch directive and the follow-up write actually dispatches.

    Regression for the 2026-09-02 write-announcement loop: glm-5.2:cloud
    ended eight consecutive turns with "Writing both files now" while the
    receipt ledger held only read receipts; the completion contract
    accepted each turn (read receipts + no write attempt = honest
    read-only) and the user had to re-ask "continue" every time. The
    rescue detects the announcement shape and rides the next iteration's
    lifecycle briefing with a directive naming the action tool.
    """
    from kernel.tools.mcp_client import MCPToolResult  # noqa: F401

    pool = _ReadThenWritePool()
    call_count = 0
    captured_prompts: list[str] = []

    async def announcing_model(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        messages = kwargs.get("messages", [])
        if messages:
            captured_prompts.append(str(messages[0].get("content", "")))
        if call_count == 1:
            # Read work first (a real tool call), then announce.
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_read",
                        "function": {
                            "name": "lisa-storage.read_file",
                            "arguments": '{"path": "zeek/local.zeek"}',
                        },
                    }
                ],
            )
        if call_count == 2:
            # The announcement: prose that claims a write with NO tool call.
            return _StubCompletion(
                content=(
                    "Now I have everything. Writing both files now — "
                    "the Python analysis script and the cheat sheet."
                )
            )
        # After the rescue directive: actually dispatch the write.
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": "call_write",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "zeek/log-mining.py", "content": "#!/usr/bin/env python3"}',
                    },
                }
            ],
        )

    hooks = TurnHooks(
        models=ModelHooks(
            complete=announcing_model,
        ),
        tools=ToolHooks(
            mcp_tool_pool=pool,
        ),
    )
    with (
        patch(
            "kernel.turn_runner.get_workflow_catalog",
            return_value=_announcement_writer_catalog(),
        ),
        patch(
            "kernel.operations.registry.get_workflow_catalog",
            return_value=_announcement_writer_catalog(),
            create=True,
        ),
        patch(
            "kernel.ipsar_fsm.get_workflow_catalog",
            return_value=_announcement_writer_catalog(),
        ),
        patch(
            "kernel.turn_runner.get_tool_catalog",
            return_value={},
        ),
    ):
        result = await run_turn(mode_id="writer", query="write the scripts", hooks=hooks)

    # The write actually dispatched — the loop is broken.
    assert pool.write_calls, "the rescue must lead to a real write dispatch"
    assert result.blocked is False
    # The rescue directive was injected into the briefing exactly once.
    rescue_prompts = [p for p in captured_prompts if "STOP ANNOUNCING AND DISPATCH" in p]
    assert rescue_prompts, "rescue directive missing from the briefing"
    assert len(rescue_prompts) == 1, "rescue directive must be one-shot"
    assert "write_file" in rescue_prompts[0]


@pytest.mark.asyncio
async def test_stalled_write_rescue_does_not_fire_for_dense_honest_answers() -> None:
    """A genuine synthesized answer (no announcement language) must NOT
    trigger the rescue — honest read-only chat work stays honest."""
    from kernel import config

    pool = _ReadThenWritePool()
    captured_prompts: list[str] = []

    async def honest_model(**kwargs: Any) -> _StubCompletion:
        messages = kwargs.get("messages", [])
        if messages:
            captured_prompts.append(str(messages[0].get("content", "")))
        if not captured_prompts or len(captured_prompts) == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_read",
                        "function": {
                            "name": "lisa-storage.read_file",
                            "arguments": '{"path": "zeek/local.zeek"}',
                        },
                    }
                ],
            )
        return _StubCompletion(
            content=(
                "I reviewed the zeek configuration and gathered the "
                "context. The policy loads the base protocol scripts and "
                "the intel framework is commented out pending the do_notice "
                "path fix. The capture-loss estimate is within normal "
                "bounds for this deployment and no further configuration "
                "changes are required right now."
            )
        )

    hooks = TurnHooks(
        models=ModelHooks(
            complete=honest_model,
        ),
        tools=ToolHooks(
            mcp_tool_pool=pool,
        ),
    )
    mp = pytest.MonkeyPatch()
    mp.setattr(config.settings, "chat_mode_dense_content_min_chars", 100)
    try:
        with (
            patch(
                "kernel.turn_runner.get_workflow_catalog",
                return_value=_announcement_writer_catalog(),
            ),
            patch(
                "kernel.operations.registry.get_workflow_catalog",
                return_value=_announcement_writer_catalog(),
                create=True,
            ),
            patch(
                "kernel.ipsar_fsm.get_workflow_catalog",
                return_value=_announcement_writer_catalog(),
            ),
            patch("kernel.turn_runner.get_tool_catalog", return_value={}),
        ):
            result = await run_turn(mode_id="writer", query="review the config", hooks=hooks)
    finally:
        mp.undo()

    assert result.blocked is False
    assert pool.write_calls == [], "honest answers must not trigger a forced write"
    assert not [p for p in captured_prompts if "STOP ANNOUNCING" in p]


@pytest.mark.asyncio
async def test_stalled_write_rescue_is_one_shot_per_turn() -> None:
    """The rescue fires once; a model that ignores it is not nudged again.

    The latch (``stalled_write_rescue_fired``) prevents the kernel from
    looping a genuinely write-averse model through repeated directives —
    after one rescue the turn completes honestly whatever the model does.
    """
    call_count = 0
    captured_prompts: list[str] = []

    async def stalling_model(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        messages = kwargs.get("messages", [])
        if messages:
            captured_prompts.append(str(messages[0].get("content", "")))
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_read",
                        "function": {
                            "name": "lisa-storage.read_file",
                            "arguments": '{"path": "a.md"}',
                        },
                    }
                ],
            )
        # Every subsequent completion: announce, never dispatch.
        return _StubCompletion(content="Writing both files now. No more stalling.")

    hooks = TurnHooks(
        models=ModelHooks(
            complete=stalling_model,
        ),
        tools=ToolHooks(
            mcp_tool_pool=_ReadThenWritePool(),
        ),
    )
    with (
        patch(
            "kernel.turn_runner.get_workflow_catalog",
            return_value=_announcement_writer_catalog(),
        ),
        patch(
            "kernel.operations.registry.get_workflow_catalog",
            return_value=_announcement_writer_catalog(),
            create=True,
        ),
        patch(
            "kernel.ipsar_fsm.get_workflow_catalog",
            return_value=_announcement_writer_catalog(),
        ),
        patch("kernel.turn_runner.get_tool_catalog", return_value={}),
    ):
        result = await run_turn(mode_id="writer", query="write them", hooks=hooks)

    rescue_prompts = [p for p in captured_prompts if "STOP ANNOUNCING AND DISPATCH" in p]
    assert len(rescue_prompts) == 1, (
        f"rescue must fire exactly once per turn, fired {len(rescue_prompts)}"
    )
    # The stalling prose was still accepted (chat softening) — no loop.
    assert result.blocked is False


@pytest.mark.asyncio
async def test_stalled_write_rescue_fires_on_silent_read_loop_at_threshold() -> None:
    """A silent read loop (no announcement language) triggers the structured
    write rescue when reaching stage_nudge_threshold.

    Regression for the 2026-09-04 incident (executor read-loop burn): 30
    fetch_page calls with no write attempts and no announcement prose burned
    to the cap because read receipts exempted the turn from every write guard.
    The structured trigger fires on read-work + 0 write attempts at threshold,
    injecting the 'COMMIT TO A WRITE OR CONCLUDE' directive.
    """
    from kernel import config

    pool = _ReadThenWritePool()
    call_count = 0
    captured_prompts: list[str] = []

    async def read_loop_model(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        messages = kwargs.get("messages", [])
        if messages:
            captured_prompts.append(str(messages[0].get("content", "")))
        if call_count < 3:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": f"call_read_{call_count}",
                        "function": {
                            "name": "lisa-storage.read_file",
                            "arguments": '{"path": "a.md"}',
                        },
                    }
                ],
            )
        if call_count == 3:
            # Silent natural completion with non-dense content (no announcement).
            return _StubCompletion(content="Still looking.")
        # Follow-up after rescue directive: dispatch the write.
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": "call_write",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "b.md", "content": "data"}',
                    },
                }
            ],
        )

    hooks = TurnHooks(
        models=ModelHooks(
            complete=read_loop_model,
        ),
        tools=ToolHooks(
            mcp_tool_pool=pool,
        ),
    )
    mp = pytest.MonkeyPatch()
    mp.setattr(config.settings, "stage_nudge_threshold", 3)
    try:
        with (
            patch(
                "kernel.turn_runner.get_workflow_catalog",
                return_value=_announcement_writer_catalog(),
            ),
            patch(
                "kernel.operations.registry.get_workflow_catalog",
                return_value=_announcement_writer_catalog(),
                create=True,
            ),
            patch(
                "kernel.ipsar_fsm.get_workflow_catalog",
                return_value=_announcement_writer_catalog(),
            ),
            patch("kernel.turn_runner.get_tool_catalog", return_value={}),
        ):
            result = await run_turn(mode_id="writer", query="find and write", hooks=hooks)
    finally:
        mp.undo()

    assert pool.write_calls, "the rescue must lead to a real write dispatch"
    assert result.blocked is False
    commit_prompts = [p for p in captured_prompts if "COMMIT TO A WRITE OR CONCLUDE" in p]
    assert len(commit_prompts) == 1, "structured directive must fire once"
    assert "write_file" in commit_prompts[0]
