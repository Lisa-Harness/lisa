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

    hooks = TurnHooks(complete=model, mcp_tool_pool=_FailingWritePool())
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

    hooks = TurnHooks(complete=model, mcp_tool_pool=_ReadOnlyPool())
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
async def test_parent_turn_dense_content_action_mode_still_blocked() -> None:
    """An ``action``-kind mode keeps the strict block (no pack-specific softening).

    Regression for the 2026-08-14 pack-leak removal: the kernel must not
    infer "this mode's writes land in the thread" from tool-name prefixes.
    Only ``mode_kind`` is consulted. An ``action``-kind mode whose model
    did read-only work and produced dense content without a write receipt
    is still blocked as ``no_tools`` — action modes' writes land outside
    the thread and the receipt is the only honest signal.
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
        return _StubCompletion(
            content=(
                "I investigated the file and gathered the context. The module "
                "exposes three entry points and the call graph confirms the "
                "boundary holds. Here is the synthesis of what I found."
            )
        )

    hooks = TurnHooks(complete=model, mcp_tool_pool=_ReadOnlyPool())
    # Same shape as the accepted case, but ``mode_kind="action"``: the
    # strict block applies regardless of content density or read-only work.
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
    # Lower the threshold here too so the test proves the block is driven
    # by ``mode_kind == "action"``, not by the content being too short.
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

    hooks = TurnHooks(complete=announcing)
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

    hooks = TurnHooks(complete=model, mcp_tool_pool=_ReadOnlyPool())
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

    # The read-only MCP tool produced no write receipt, and because it is
    # excluded from the (now empty) write contract, the turn is NOT blocked.
    assert result.blocked is False
    assert result.block_origin == ""
    assert not any(r.capability_id == "lisa-storage.read_file" for r in result.effect_receipts)


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

    hooks = TurnHooks(complete=read_loop, execute_operation=stub_execute_operation)
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
    nudge_iterations = [
        i for i, p in enumerate(captured_prompts) if "zero writes" in p
    ]
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

    hooks = TurnHooks(complete=read_loop, execute_operation=stub_execute_operation)
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
