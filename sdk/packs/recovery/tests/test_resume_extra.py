"""Branch-coverage tests for ``handle_resume`` beyond the core paths.

Complements ``test_resume.py`` with:

- yielded entry with an empty ``yield_question`` → bare "Resumed from turn" line
- stalled+confirm records surfaced alongside the yield question
- ``list_open`` raising → warning swallowed, plain resume text returned
- service ``None`` / missing ``list_open`` → no stall summary appended
- open records that are not stalled+confirm → no stall summary
- stalled+confirm records on the pending-reply path (summary appended)
"""

from __future__ import annotations

from typing import Any

import pytest
from kernel.recovery import RecoveryIdentity, RecoveryRecord
from sdk.packs.recovery.commands import handle_resume


def _identity() -> RecoveryIdentity:
    return RecoveryIdentity(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        memory_scope="personal",
    )


def _stalled_confirm_record(target_id: str = "op_stalled") -> RecoveryRecord:
    return RecoveryRecord(
        identity=_identity(),
        target_kind="operation",
        target_id=target_id,
        state="stalled",
        retry_policy="confirm",
        effect="write",
        last_error="awaiting approval",
    )


class _FakeService:
    """In-memory recovery service returning a preset open list (or raising)."""

    def __init__(
        self,
        open_records: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._open_records = list(open_records or [])
        self._error = error

    async def list_open(self, identity: Any) -> list[Any]:
        if self._error is not None:
            raise self._error
        return list(self._open_records)


class _FakeRuntime:
    """Minimal runtime stub mirroring the grouped HostRuntime surface."""

    def __init__(self, service: Any) -> None:
        self.paths = type("Paths", (), {"thread_path": "thread"})()
        self.services = type("Services", (), {"recovery_service": service})()
        self.session = type("Session", (), {"pending_resume_message": None})()
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


def _entry(
    *,
    turn: int = 1,
    yielded: bool = False,
    yield_question: str = "",
) -> dict[str, Any]:
    return {"turn": turn, "yielded": yielded, "yield_question": yield_question}


def _patched_resume(monkeypatch: pytest.MonkeyPatch, entries: list[dict[str, Any]]) -> None:
    def _fake_load(**_kwargs: Any) -> list[dict[str, Any]]:
        return list(entries)

    # handle_resume imports load_recent_thread_entries lazily from
    # runtime.thread; patch it there so the in-function import sees the stub.
    monkeypatch.setattr(
        "runtime.thread.load_recent_thread_entries",
        _fake_load,
        raising=False,
    )


# ── yield-question variants ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_yielded_entry_without_question_returns_bare_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A yielded entry with an empty ``yield_question`` renders the bare
    "Resumed from turn N." line with no "Question:" suffix."""
    _patched_resume(monkeypatch, [_entry(turn=3, yielded=True, yield_question="")])

    runtime = _FakeRuntime(_FakeService())
    result = await handle_resume(runtime, "/resume", "/resume")

    assert result == "Resumed from turn 3."
    assert runtime.session.pending_resume_message is None


# ── stalled-record surfacing ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_surfaces_stalled_confirm_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stalled + confirm-policy records are listed under the yield question
    with their last_error and the resolution hint."""
    _patched_resume(monkeypatch, [_entry(turn=2, yielded=True, yield_question="Continue?")])
    service = _FakeService([_stalled_confirm_record("op_stalled")])

    runtime = _FakeRuntime(service)
    result = await handle_resume(runtime, "/resume", "/resume")

    assert "Resumed from turn 2" in result
    assert "Question: Continue?" in result
    assert "Stalled work items needing your decision:" in result
    assert "op_stalled (operation): awaiting approval" in result
    assert "Use /retry, /skip, or /cancel to resolve." in result


@pytest.mark.asyncio
async def test_resume_ignores_non_stalled_or_non_confirm_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open records that are not (stalled AND confirm) produce no summary."""
    non_matching = [
        _stalled_confirm_record("op_safe").model_copy(update={"retry_policy": "safe"}),
        _stalled_confirm_record("op_indeterminate").model_copy(update={"state": "indeterminate"}),
    ]
    _patched_resume(monkeypatch, [_entry(turn=2, yielded=True, yield_question="Continue?")])

    runtime = _FakeRuntime(_FakeService(non_matching))
    result = await handle_resume(runtime, "/resume", "/resume")

    assert result == "Resumed from turn 2.\n\nQuestion: Continue?"


@pytest.mark.asyncio
async def test_resume_swallows_list_open_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising ``list_open`` is logged and swallowed — the resume text is
    still returned without a stall summary."""
    _patched_resume(monkeypatch, [_entry(turn=4, yielded=True, yield_question="Pick one.")])

    runtime = _FakeRuntime(_FakeService(error=RuntimeError("store offline")))
    result = await handle_resume(runtime, "/resume", "/resume")

    assert result == "Resumed from turn 4.\n\nQuestion: Pick one."


@pytest.mark.asyncio
async def test_resume_without_recovery_service_skips_stall_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No service at all → the stalled-record block is silently skipped."""
    _patched_resume(monkeypatch, [_entry(turn=5, yielded=True, yield_question="Yes or no?")])

    runtime = _FakeRuntime(None)
    result = await handle_resume(runtime, "/resume", "/resume")

    assert result == "Resumed from turn 5.\n\nQuestion: Yes or no?"


@pytest.mark.asyncio
async def test_resume_service_without_list_open_skips_stall_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-None service attribute lacking ``list_open`` counts as absent."""
    _patched_resume(monkeypatch, [_entry(turn=6, yielded=True, yield_question="Yes or no?")])

    runtime = _FakeRuntime(object())
    result = await handle_resume(runtime, "/resume", "/resume")

    assert result == "Resumed from turn 6.\n\nQuestion: Yes or no?"


@pytest.mark.asyncio
async def test_resume_reply_includes_stall_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """``/resume <reply>`` stashes the reply AND still appends the stalled
    record summary for the operator."""
    _patched_resume(monkeypatch, [_entry(turn=7, yielded=True, yield_question="Confirm?")])

    runtime = _FakeRuntime(_FakeService([_stalled_confirm_record("op_gate")]))
    result = await handle_resume(runtime, "/resume ship it", "/resume ship it")

    assert runtime.session.pending_resume_message == "ship it"
    assert "Resumed from turn 7. Your reply will be sent next." in result
    assert "op_gate (operation): awaiting approval" in result
