"""Branch-coverage tests for recovery command guard paths and rendering.

Complements ``test_recovery_commands.py`` with the error/edge branches:

- missing or incomplete service → "Recovery service is not available…"
- usage errors (``/stall``, ``/retry``, ``/skip``, ``/cancel`` with no id)
- service.get/settle raising → "<Cmd> failed: {Type}: {msg}"
- get/settle returning None → "No (open) work item found with id …"
- ``_retry_guard_message``: not-operator-open and confirm/effectful prompt
- ``_render_record_summary``/``_render_record_detail``: ``stage_id or '-'``
  fallback, last_error/result_preview/settled_at append lines
- ``_bulk_dismiss`` (skip-all/cancel-all): service errors, nothing to
  dismiss, settle returning None, settle raising per item
- ``/retry-safe``: list error, empty list, settle failure counted
"""

from __future__ import annotations

from typing import Any

import pytest
from kernel.recovery import RecoveryIdentity, RecoveryRecord
from sdk.packs.recovery.commands import (
    handle_cancel,
    handle_cancel_all,
    handle_retry,
    handle_retry_safe,
    handle_skip,
    handle_skip_all,
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
    )


class _FakeService:
    """In-memory recovery service (same pattern as test_recovery_commands)."""

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
        retry_policy: str | None = None,
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

    def __init__(self, service: Any) -> None:
        self.services = type("Services", (), {"recovery_service": service})()
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


class _ExplodingService:
    """Service whose every method raises the configured error."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def list_open(self, _identity: RecoveryIdentity) -> list[Any]:
        raise self._error

    async def get(self, _work_item_id: str, _identity: RecoveryIdentity) -> Any:
        raise self._error

    async def settle(self, *_args: Any, **_kwargs: Any) -> Any:
        raise self._error


# ── service availability guards ───────────────────────────────────────


@pytest.mark.asyncio
async def test_all_handlers_report_missing_recovery_service() -> None:
    runtime = _FakeRuntime(None)

    assert "not available" in await handle_stalls(runtime, "/stalls", "/stalls")
    assert "not available" in await handle_stall(runtime, "/stall abc", "/stall abc")
    assert "not available" in await handle_retry(runtime, "/retry abc", "/retry abc")
    assert "not available" in await handle_retry_safe(runtime, "/retry-safe", "/retry-safe")
    assert "not available" in await handle_skip(runtime, "/skip abc", "/skip abc")
    assert "not available" in await handle_cancel(runtime, "/cancel abc", "/cancel abc")
    assert "not available" in await handle_skip_all(runtime, "/skip-all", "/skip-all")
    assert "not available" in await handle_cancel_all(runtime, "/cancel-all", "/cancel-all")


@pytest.mark.asyncio
async def test_service_without_expected_methods_reports_unavailable() -> None:
    """A non-None service lacking ``list_open``/``get``/``settle`` counts as
    unavailable (the ``hasattr`` guard branch)."""
    runtime = _FakeRuntime(object())

    assert "not available" in await handle_stalls(runtime, "/stalls", "/stalls")
    assert "not available" in await handle_stall(runtime, "/stall abc", "/stall abc")
    assert "not available" in await handle_retry(runtime, "/retry abc", "/retry abc")
    assert "not available" in await handle_retry_safe(runtime, "/retry-safe", "/retry-safe")
    assert "not available" in await handle_skip(runtime, "/skip abc", "/skip abc")
    assert "not available" in await handle_cancel(runtime, "/cancel abc", "/cancel abc")
    assert "not available" in await handle_skip_all(runtime, "/skip-all", "/skip-all")
    assert "not available" in await handle_cancel_all(runtime, "/cancel-all", "/cancel-all")


# ── usage errors ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_id_commands_return_usage_when_id_missing_or_blank() -> None:
    runtime = _FakeRuntime(_FakeService())

    assert await handle_stall(runtime, "/stall", "/stall") == "Usage: /stall <work_item_id>"
    assert await handle_stall(runtime, "/stall  ", "/stall  ") == "Usage: /stall <work_item_id>"
    assert await handle_retry(runtime, "/retry", "/retry") == "Usage: /retry <work_item_id>"
    assert await handle_skip(runtime, "/skip", "/skip") == "Usage: /skip <work_item_id>"
    assert await handle_cancel(runtime, "/cancel", "/cancel") == "Usage: /cancel <work_item_id>"


# ── service raising ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stalls_list_error_reports_failure() -> None:
    runtime = _FakeRuntime(_ExplodingService(RuntimeError("db down")))

    result = await handle_stalls(runtime, "/stalls", "/stalls")

    assert result == "Failed to list stalled work: RuntimeError: db down"


@pytest.mark.asyncio
async def test_stall_get_error_reports_failure() -> None:
    runtime = _FakeRuntime(_ExplodingService(ValueError("bad id")))

    result = await handle_stall(runtime, "/stall abc", "/stall abc")

    assert result == "Failed to read work item: ValueError: bad id"


@pytest.mark.asyncio
async def test_retry_get_error_reports_failure() -> None:
    runtime = _FakeRuntime(_ExplodingService(ValueError("bad id")))

    result = await handle_retry(runtime, "/retry abc", "/retry abc")

    assert result == "Failed to read work item: ValueError: bad id"


@pytest.mark.asyncio
async def test_retry_settle_error_reports_failure() -> None:
    """get succeeds but the replay-placeholder settle raises."""

    class _GetOkSettleRaises:
        async def get(self, work_item_id: str, identity: RecoveryIdentity) -> Any:
            return _make_record(target_id="op_r")

        async def settle(self, *_args: Any, **_kwargs: Any) -> Any:
            raise OSError("store write failed")

    runtime = _FakeRuntime(_GetOkSettleRaises())

    result = await handle_retry(runtime, "/retry abc", "/retry abc")

    assert result == "Retry failed to settle: OSError: store write failed"


@pytest.mark.asyncio
async def test_retry_safe_list_error_reports_failure() -> None:
    runtime = _FakeRuntime(_ExplodingService(RuntimeError("db down")))

    result = await handle_retry_safe(runtime, "/retry-safe", "/retry-safe")

    assert result == "Failed to list stalled work: RuntimeError: db down"


@pytest.mark.asyncio
async def test_retry_safe_settle_errors_reduce_settled_count() -> None:
    service = _FakeService()
    ok = await service.begin(_make_record(target_id="safe_ok"))
    await service.begin(_make_record(target_id="safe_bad"))

    class _FlakySettleService(_FakeService):
        def __init__(self, wrapped: _FakeService) -> None:
            super().__init__()
            self._wrapped = wrapped
            self.records = wrapped.records

        async def settle(self, *args: Any, **kwargs: Any) -> Any:
            if args[0] == ok.work_item_id:
                return await self._wrapped.settle(*args, **kwargs)
            raise RuntimeError("transient store error")

    runtime = _FakeRuntime(_FlakySettleService(service))

    result = await handle_retry_safe(runtime, "/retry-safe", "/retry-safe")

    assert result == "Retried 1/2 safe work items."


@pytest.mark.asyncio
async def test_skip_settle_error_reports_failure() -> None:
    runtime = _FakeRuntime(_ExplodingService(OSError("io err")))

    result = await handle_skip(runtime, "/skip abc", "/skip abc")

    assert result == "Skip failed: OSError: io err"


@pytest.mark.asyncio
async def test_cancel_settle_error_reports_failure() -> None:
    runtime = _FakeRuntime(_ExplodingService(OSError("io err")))

    result = await handle_cancel(runtime, "/cancel abc", "/cancel abc")

    assert result == "Cancel failed: OSError: io err"


@pytest.mark.asyncio
async def test_bulk_dismiss_list_error_reports_failure() -> None:
    runtime = _FakeRuntime(_ExplodingService(RuntimeError("db down")))

    skip_result = await handle_skip_all(runtime, "/skip-all", "/skip-all")
    cancel_result = await handle_cancel_all(runtime, "/cancel-all", "/cancel-all")

    assert skip_result == "Failed to list stalled work: RuntimeError: db down"
    assert cancel_result == "Failed to list stalled work: RuntimeError: db down"


# ── not-found branches ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skip_unknown_id_reports_no_open_item() -> None:
    service = _FakeService()
    await service.begin(_make_record(target_id="op_present"))
    runtime = _FakeRuntime(service)

    result = await handle_skip(runtime, "/skip missing-id", "/skip missing-id")

    assert result == "No open work item found with id 'missing-id' for this session."


@pytest.mark.asyncio
async def test_skip_already_settled_reports_no_open_item() -> None:
    """settle() → None also when the record exists but belongs to another
    identity — the fake service returns None for the identity mismatch,
    matching the real store's scoping behavior."""
    service = _FakeService()
    runtime = _FakeRuntime(service)

    result = await handle_cancel(runtime, "/cancel missing-id", "/cancel missing-id")

    assert result == "No open work item found with id 'missing-id' for this session."


# ── _retry_guard_message branches ─────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_unknown_id_reports_not_found() -> None:
    service = _FakeService()
    runtime = _FakeRuntime(service)

    result = await handle_retry(runtime, "/retry ghost", "/retry ghost")

    assert result == "No work item found with id 'ghost' for this session."


@pytest.mark.asyncio
async def test_retry_not_open_for_operator_reports_state() -> None:
    """A settled record is not operator-open → guard reports its state."""
    service = _FakeService()
    record = await service.begin(
        _make_record(target_id="op_done", state="succeeded", retry_policy="safe")
    )
    runtime = _FakeRuntime(service)

    result = await handle_retry(
        runtime, f"/retry {record.work_item_id}", f"/retry {record.work_item_id}"
    )

    assert "is not open for retry" in result
    assert "'succeeded'" in result


@pytest.mark.asyncio
async def test_retry_confirm_effectful_prompt_lists_alternative_paths() -> None:
    """The confirm/effectful guard names both escape paths: re-issue the
    original request, or dismiss via /skip or /cancel."""
    service = _FakeService()
    record = await service.begin(
        _make_record(target_id="op_effectful", retry_policy="confirm", effect="write")
    )
    runtime = _FakeRuntime(service)

    result = await handle_retry(
        runtime, f"/retry {record.work_item_id}", f"/retry {record.work_item_id}"
    )

    assert "effectful" in result
    assert "re-issue your original request" in result
    assert f"/skip {record.work_item_id}" in result
    assert f"/cancel {record.work_item_id}" in result


# ── _bulk_dismiss remaining branches ──────────────────────────────────


@pytest.mark.asyncio
async def test_skip_all_nothing_dismissible_returns_message() -> None:
    """Only running/settled records → the dismissible filter leaves none."""
    service = _FakeService()
    await service.begin(_make_record(target_id="op_running", state="running"))
    runtime = _FakeRuntime(service)

    result = await handle_skip_all(runtime, "/skip-all", "/skip-all")

    assert result == "No stalled or indeterminate work items to dismiss."


@pytest.mark.asyncio
async def test_cancel_all_settle_returning_none_is_not_counted() -> None:
    """Concurrently-settled item: settle returns None → excluded from count."""
    service = _FakeService()
    await service.begin(_make_record(target_id="op_x", state="indeterminate"))

    class _AllNoneSettleService(_FakeService):
        def __init__(self, wrapped: _FakeService) -> None:
            super().__init__()
            self.records = wrapped.records

        async def settle(self, *_args: Any, **_kwargs: Any) -> Any:
            return None

    runtime = _FakeRuntime(_AllNoneSettleService(service))

    result = await handle_cancel_all(runtime, "/cancel-all", "/cancel-all")

    assert result == "Cancelled 0/1 work items."


@pytest.mark.asyncio
async def test_skip_all_settle_error_leaves_record_unsettled() -> None:
    service = _FakeService()
    okay = await service.begin(_make_record(target_id="ok_a"))
    broken = await service.begin(_make_record(target_id="broken_b"))

    class _PartialFailureService(_FakeService):
        def __init__(self, wrapped: _FakeService) -> None:
            super().__init__()
            self._wrapped = wrapped
            self.records = wrapped.records

        async def settle(self, *args: Any, **kwargs: Any) -> Any:
            if args[0] == broken.work_item_id:
                raise RuntimeError("settle blew up")
            return await self._wrapped.settle(*args, **kwargs)

    runtime = _FakeRuntime(_PartialFailureService(service))

    result = await handle_skip_all(runtime, "/skip-all", "/skip-all")

    assert result == "Skipped 1/2 work items."
    settled_ok = await service.get(okay.work_item_id, _identity())
    settled_broken = await service.get(broken.work_item_id, _identity())
    assert settled_ok is not None and settled_ok.state == "skipped"
    assert settled_broken is not None and settled_broken.state == "stalled"


# ── rendering helpers (covered through the public handlers) ───────────


@pytest.mark.asyncio
async def test_stalls_summary_renders_dash_for_missing_stage_and_appends_error() -> None:
    """``stage_id`` empty → ``stage=-``; a non-empty ``last_error`` appends
    the error continuation line to the summary."""
    service = _FakeService()
    record = await service.begin(_make_record(target_id="op_stageless"))
    assert not record.stage_id
    runtime = _FakeRuntime(service)

    result = await handle_stalls(runtime, "/stalls", "/stalls")

    assert "stage=-" in result
    assert "error: timeout" in result


@pytest.mark.asyncio
async def test_stalls_summary_omits_error_line_when_empty() -> None:
    service = _FakeService()
    await service.begin(_make_record(target_id="op_clean", state="indeterminate"))
    runtime = _FakeRuntime(service)

    result = await handle_stalls(runtime, "/stalls", "/stalls")

    assert "op_clean" in result
    assert "error:" not in result


@pytest.mark.asyncio
async def test_stall_detail_appends_error_result_preview_and_settled_at() -> None:
    """The detail block appends last_error / result_preview / settled_at
    lines only when those fields are non-empty / non-None."""
    service = _FakeService()
    record = await service.begin(_make_record(target_id="op_full", effect="write"))
    detailed = record.model_copy(
        update={"result_preview": "partial payload", "settled_at": "2026-10-01T00:00:00Z"}
    )
    service.records[record.work_item_id] = detailed
    runtime = _FakeRuntime(service)

    result = await handle_stall(
        runtime, f"/stall {record.work_item_id}", f"/stall {record.work_item_id}"
    )

    assert "last_error: timeout" in result
    assert "result_preview: partial payload" in result
    assert "settled_at: 2026-10-01T00:00:00Z" in result


@pytest.mark.asyncio
async def test_stall_detail_omits_optional_lines_when_empty() -> None:
    service = _FakeService()
    record = await service.begin(
        _make_record(target_id="op_bare", state="indeterminate")  # last_error=""
    )
    runtime = _FakeRuntime(service)

    result = await handle_stall(
        runtime, f"/stall {record.work_item_id}", f"/stall {record.work_item_id}"
    )

    assert "op_bare" in result
    assert "last_error:" not in result
    assert "result_preview:" not in result
    assert "settled_at:" not in result
