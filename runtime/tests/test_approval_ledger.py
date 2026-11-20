"""Durable approval evidence: the append-only transitions ledger.

``FileApprovalService`` appends one row per state transition (requested /
approved / denied / consumed / expired) to ``approvals.jsonl`` — the
evidence stream that answers "who authorized what, when" — beside its
authoritative working store. The policy auto-allow is recorded with
``decision_source: "policy"`` so an auto-approval is never misattributed to
an operator. The kernel-side invocation-stamping tests live in
``kernel/tests/test_approval_invocation_stamping.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from kernel.approval import (
    ApprovalIdentity,
    ApprovalInvocation,
    build_approval_invocation,
)
from runtime.approval_ledger import ApprovalLedger


def _identity() -> ApprovalIdentity:
    return ApprovalIdentity(user_slug="adam", mode_id="coding")


def _invocation(turn_index: int | None = None, **overrides: object) -> ApprovalInvocation:
    inv = build_approval_invocation(
        target_kind="tool",
        target_id="apply_patch",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"path": "src/x.py"},
        identity=_identity(),
        pack_config={},
        **overrides,  # type: ignore[arg-type]
    )
    return inv.model_copy(update={"turn_index": turn_index, "thread_store_slug": "adam"})


def _ledger_rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "approvals.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


# ── Ledger transitions ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ledger_records_each_transition(tmp_path: Path) -> None:
    from runtime.approvals import FileApprovalService

    store = FileApprovalService(
        tmp_path / "approvals.json", ledger_path=tmp_path / "approvals.jsonl"
    )
    identity = _identity()
    inv = _invocation(turn_index=7)
    recorded = await store.request(inv)
    receipt = await store.approve(recorded.request_id, identity)
    assert receipt is not None
    consumed = await store.consume(recorded.request_id, inv)
    assert consumed is not None

    rows = [json.loads(line) for line in (tmp_path / "approvals.jsonl").read_text().splitlines()]
    assert [r["state"] for r in rows] == ["requested", "approved", "consumed"]
    assert all(r["turn_index"] == 7 for r in rows)
    assert all(r["user_slug"] == "adam" for r in rows)
    assert all(r["thread_store_slug"] == "adam" for r in rows)
    assert rows[1]["decision_by"] == "adam"
    assert rows[1]["decision_source"] == "operator"
    assert [r["sequence"] for r in rows] == [1, 2, 3]


@pytest.mark.asyncio
async def test_ledger_records_deny_with_reason(tmp_path: Path) -> None:
    from runtime.approvals import FileApprovalService

    store = FileApprovalService(
        tmp_path / "approvals.json", ledger_path=tmp_path / "approvals.jsonl"
    )
    identity = _identity()
    inv = _invocation(turn_index=8)
    recorded = await store.request(inv)
    assert await store.deny(recorded.request_id, identity, reason="not today")

    rows = [json.loads(line) for line in (tmp_path / "approvals.jsonl").read_text().splitlines()]
    assert [r["state"] for r in rows] == ["requested", "denied"]
    assert rows[1]["reason"] == "not today"
    assert rows[1]["decision_source"] == "operator"


@pytest.mark.asyncio
async def test_ledger_records_expiry_observation(tmp_path: Path) -> None:
    from runtime.approvals import FileApprovalService

    store = FileApprovalService(
        tmp_path / "approvals.json", ledger_path=tmp_path / "approvals.jsonl"
    )
    identity = _identity()
    stale = _invocation(
        turn_index=9,
        now=datetime.now(UTC) - timedelta(minutes=60),
        ttl=timedelta(minutes=30),
    )
    recorded = await store.request(stale)
    assert await store.approve(recorded.request_id, identity) is None

    rows = [json.loads(line) for line in (tmp_path / "approvals.jsonl").read_text().splitlines()]
    assert [r["state"] for r in rows] == ["requested", "expired"]
    assert rows[1]["decision_source"] == "expiry"


@pytest.mark.asyncio
async def test_deduped_request_writes_no_second_row(tmp_path: Path) -> None:
    from runtime.approvals import FileApprovalService

    store = FileApprovalService(
        tmp_path / "approvals.json", ledger_path=tmp_path / "approvals.jsonl"
    )
    inv = _invocation(turn_index=7)
    first = await store.request(inv)
    # Same action re-issued while pending: returns the existing record.
    second = await store.request(inv)
    assert second.request_id == first.request_id

    rows = [json.loads(line) for line in (tmp_path / "approvals.jsonl").read_text().splitlines()]
    assert [r["state"] for r in rows] == ["requested"]


@pytest.mark.asyncio
async def test_policy_auto_allow_is_not_recorded_as_operator(tmp_path: Path) -> None:
    """An auto-approval is evidence with ``decision_source: policy``."""
    from runtime.approvals import FileApprovalService
    from runtime.policy import PolicyDocument, PolicyRule
    from runtime.policy_service import PolicyApprovalService

    policy = PolicyDocument(
        rules=[
            PolicyRule(
                id="allow-reads",
                description="read-only is safe",
                target="*",
                effect="read",
                verdict="allow",
            )
        ]
    )
    store = FileApprovalService(
        tmp_path / "approvals.json", ledger_path=tmp_path / "approvals.jsonl"
    )
    svc = PolicyApprovalService(store, policy=policy, data_dir=tmp_path)

    inv = _invocation(turn_index=3)
    inv = inv.model_copy(update={"effect": "read"})
    await svc.request(inv)

    rows = [json.loads(line) for line in (tmp_path / "approvals.jsonl").read_text().splitlines()]
    assert [r["state"] for r in rows] == ["requested", "approved"]
    assert rows[1]["decision_source"] == "policy"


@pytest.mark.asyncio
async def test_no_ledger_path_keeps_previous_behavior(tmp_path: Path) -> None:
    """Hosts that wire no ledger keep the working store exactly as before."""
    from runtime.approvals import FileApprovalService

    store = FileApprovalService(tmp_path / "approvals.json")
    identity = _identity()
    inv = _invocation(turn_index=7)
    recorded = await store.request(inv)
    assert await store.approve(recorded.request_id, identity) is not None
    assert not (tmp_path / "approvals.jsonl").exists()


@pytest.mark.asyncio
async def test_ledger_append_failure_never_blocks_transition(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The working store commits even when the ledger cannot be written."""

    from runtime.approvals import FileApprovalService

    store = FileApprovalService(
        tmp_path / "approvals.json", ledger_path=tmp_path / "blocked" / "x" / "approvals.jsonl"
    )
    # A file where a directory must be makes the append fail.
    (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        inv = _invocation(turn_index=7)
        recorded = await store.request(inv)
    assert recorded.request_id
    assert any("approval ledger append failed" in r.message for r in caplog.records)


def _execution_invocation(turn_id: str) -> ApprovalInvocation:
    """Return an invocation stamped with one execution id (§4)."""
    return _invocation(turn_index=7).model_copy(
        update={"turn_id": turn_id, "tool_call_id": "toolu-1", "dispatch_id": "disp-1"}
    )


def test_ledger_row_carries_the_requesting_execution_id(tmp_path: Path) -> None:
    """The durable row names the execution that asked, not just its position.

    ``turn_index`` is a position in one thread, so it cannot separate two
    overlapping executions; the row needs the id for the ledger to be joinable
    to the same execution's decision rows and thread entry.
    """
    ledger = ApprovalLedger(tmp_path / "approvals.jsonl")
    ledger.record_transition(invocation=_execution_invocation("turn-exec-1"), state="pending")

    rows = _ledger_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["state"] == "pending"
    assert row["turn_id"] == "turn-exec-1"
    # The pre-existing attribution keys still land beside the new one.
    assert row["turn_index"] == 7
    assert row["tool_call_id"] == "toolu-1"
    assert row["dispatch_id"] == "disp-1"


def test_an_off_turn_row_records_an_empty_execution_id(tmp_path: Path) -> None:
    """A request from no execution records the empty id rather than inventing one."""
    ledger = ApprovalLedger(tmp_path / "approvals.jsonl")
    ledger.record_transition(invocation=_execution_invocation(""), state="pending")

    rows = _ledger_rows(tmp_path)
    assert rows[0]["turn_id"] == ""
