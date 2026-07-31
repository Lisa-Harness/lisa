"""Tests for the recovery pack command handlers.

Covers:
- ``/stalls`` lists open recovery items
- ``/stall <id>`` shows detail for one item
- ``/retry <id>`` settles safe items, prompts for confirm items
- ``/retry-safe`` batches all safe items
- ``/skip <id>`` settles as skipped
- ``/cancel <id>`` settles as cancelled
- ``get_host_commands`` returns all six specs
"""

from __future__ import annotations

import pytest

from kernel.recovery import RecoveryIdentity, RecoveryRecord
from sdk.packs.recovery.commands import (
    get_host_commands,
    handle_cancel,
    handle_retry,
    handle_retry_safe,
    handle_skip,
    handle_stall,
    handle_stalls,
)


def _identity() -> RecoveryIdentity:
    return RecoveryIdentity(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        memory_scope="personal",
        session_plan_digest="digest",
    )


class _FakeService:
    """In-memory recovery service for pack command tests."""

    def __init__(self) -> None:
        self.records: dict[str, RecoveryRecord] = {}

    async def begin(self, record: RecoveryRecord) -> RecoveryRecord:
        self.records[record.work_item_id] = record
        return record

    async def settle(
        self,
        work_item_id: str,
        identity: RecoveryIdentity,
        *,
        state: str,
        last_error: str = "",
        result_preview: str = "",
        failure_scope: str | None = None,
        retry_policy: str | None = None,
        blocks_workflow: bool | None = None,
    ) -> RecoveryRecord | None:
        record = self.records.get(work_item_id)
        if record is None or record.identity != identity:
            return None
        update: dict[str, object] = {
            "state": state,
            "last_error": last_error,
            "result_preview": result_preview,
        }
        if retry_policy is not None:
            update["retry_policy"] = retry_policy
        if blocks_workflow is not None:
            update["blocks_workflow"] = blocks_workflow
        updated = record.model_copy(update=update)
        self.records[work_item_id] = updated
        return updated

    async def get(self, work_item_id: str, identity: RecoveryIdentity) -> RecoveryRecord | None:
        record = self.records.get(work_item_id)
        if record is None or record.identity != identity:
            return None
        return record

    async def list_open(self, identity: RecoveryIdentity) -> list[RecoveryRecord]:
        return [
            r for r in self.records.values() if r.identity == identity and r.open_for_operator()
        ]


class _FakeRuntime:
    """Minimal runtime stub with a recovery_service and session identity."""

    def __init__(self, service: _FakeService) -> None:
        self.recovery_service = service
        self.session_plan = type(
            "Plan",
            (),
            {"manifest": type("M", (), {"name": "workspace"}), "plan_digest": "digest"},
        )()
        self.config = type(
            "Cfg",
            (),
            {
                "user_slug": "user",
                "surface_type": "personal_dm",
                "surface_id": "surface",
                "memory_scope": "personal",
            },
        )()


def _make_record(
    *,
    target_id: str = "op_a",
    state: str = "stalled",
    retry_policy: str = "safe",
    effect: str = "read",
) -> RecoveryRecord:
    return RecoveryRecord(
        identity=_identity(),
        target_kind="operation",
        target_id=target_id,
        state=state,
        retry_policy=retry_policy,
        effect=effect,
        last_error="timeout" if state == "stalled" else "",
    )


# ── get_host_commands ─────────────────────────────────────────────────


def test_get_host_commands_returns_all_six_specs() -> None:
    specs = get_host_commands()
    prefixes = {s.prefix for s in specs}
    assert prefixes == {"/stalls", "/stall", "/retry", "/retry-safe", "/skip", "/cancel"}


# ── /stalls ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stalls_lists_open_items() -> None:
    service = _FakeService()
    await service.begin(_make_record(target_id="op_a"))
    runtime = _FakeRuntime(service)

    result = await handle_stalls(runtime, "/stalls", "/stalls")

    assert "Open recovery work items:" in result
    assert "op_a" in result


@pytest.mark.asyncio
async def test_stalls_empty_returns_no_items_message() -> None:
    service = _FakeService()
    runtime = _FakeRuntime(service)

    result = await handle_stalls(runtime, "/stalls", "/stalls")

    assert result == "No stalled or indeterminate work items."


# ── /stall <id> ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stall_shows_detail_for_existing_item() -> None:
    service = _FakeService()
    record = await service.begin(_make_record(target_id="op_b"))
    runtime = _FakeRuntime(service)

    result = await handle_stall(
        runtime, f"/stall {record.work_item_id}", f"/stall {record.work_item_id}"
    )

    assert "Work item:" in result
    assert "op_b" in result
    assert "state: stalled" in result


@pytest.mark.asyncio
async def test_stall_missing_id_returns_not_found() -> None:
    service = _FakeService()
    runtime = _FakeRuntime(service)

    result = await handle_stall(runtime, "/stall nonexistent", "/stall nonexistent")

    assert "No work item found" in result


# ── /retry <id> ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_safe_item_settles_as_succeeded() -> None:
    service = _FakeService()
    record = await service.begin(_make_record(target_id="op_c", retry_policy="safe"))
    runtime = _FakeRuntime(service)

    result = await handle_retry(
        runtime, f"/retry {record.work_item_id}", f"/retry {record.work_item_id}"
    )

    assert "Retried work item" in result
    settled = await service.get(record.work_item_id, _identity())
    assert settled is not None
    assert settled.state == "succeeded"


@pytest.mark.asyncio
async def test_retry_confirm_item_prompts_for_approval() -> None:
    service = _FakeService()
    record = await service.begin(
        _make_record(target_id="op_d", retry_policy="confirm", effect="write")
    )
    runtime = _FakeRuntime(service)

    result = await handle_retry(
        runtime, f"/retry {record.work_item_id}", f"/retry {record.work_item_id}"
    )

    assert "requires explicit approval" in result
    settled = await service.get(record.work_item_id, _identity())
    assert settled is not None
    assert settled.state == "stalled"


@pytest.mark.asyncio
async def test_retry_never_item_is_rejected() -> None:
    service = _FakeService()
    record = await service.begin(_make_record(target_id="op_e", retry_policy="never"))
    runtime = _FakeRuntime(service)

    result = await handle_retry(
        runtime, f"/retry {record.work_item_id}", f"/retry {record.work_item_id}"
    )

    assert "retry_policy='never'" in result


# ── /retry-safe ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_safe_batch_settles_all_safe_items() -> None:
    service = _FakeService()
    r1 = await service.begin(_make_record(target_id="safe_a", retry_policy="safe"))
    r2 = await service.begin(_make_record(target_id="safe_b", retry_policy="safe"))
    await service.begin(_make_record(target_id="confirm_c", retry_policy="confirm"))
    runtime = _FakeRuntime(service)

    result = await handle_retry_safe(runtime, "/retry-safe", "/retry-safe")

    assert "2/2" in result
    s1 = await service.get(r1.work_item_id, _identity())
    s2 = await service.get(r2.work_item_id, _identity())
    assert s1 is not None and s1.state == "succeeded"
    assert s2 is not None and s2.state == "succeeded"


@pytest.mark.asyncio
async def test_retry_safe_no_safe_items_returns_message() -> None:
    service = _FakeService()
    await service.begin(_make_record(target_id="confirm_x", retry_policy="confirm"))
    runtime = _FakeRuntime(service)

    result = await handle_retry_safe(runtime, "/retry-safe", "/retry-safe")

    assert "No safe" in result


# ── /skip <id> ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skip_settles_as_skipped() -> None:
    service = _FakeService()
    record = await service.begin(_make_record(target_id="op_f"))
    runtime = _FakeRuntime(service)

    result = await handle_skip(
        runtime, f"/skip {record.work_item_id}", f"/skip {record.work_item_id}"
    )

    assert "Skipped work item" in result
    settled = await service.get(record.work_item_id, _identity())
    assert settled is not None
    assert settled.state == "skipped"


# ── /cancel <id> ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_settles_as_cancelled() -> None:
    service = _FakeService()
    record = await service.begin(_make_record(target_id="op_g"))
    runtime = _FakeRuntime(service)

    result = await handle_cancel(
        runtime, f"/cancel {record.work_item_id}", f"/cancel {record.work_item_id}"
    )

    assert "Cancelled work item" in result
    settled = await service.get(record.work_item_id, _identity())
    assert settled is not None
    assert settled.state == "cancelled"
