"""Tests for the recovery pack ``/resume`` handler.

Covers ``handle_resume``:
- Surfaces the ``yield_question`` when the thread tail has a yielded entry.
- Returns the no-yield message when no yielded entry is present.
- Stashes a ``/resume <reply>`` into ``runtime.pending_resume_message``.
"""

from __future__ import annotations

from typing import Any

import pytest

from sdk.packs.recovery.commands import handle_resume


class _FakeService:
    """In-memory recovery service with an empty open list by default."""

    async def list_open(self, identity: Any) -> list[Any]:
        return []


class _FakeRuntime:
    """Minimal runtime stub: thread_path + recovery_service + resume sink."""

    def __init__(self, service: _FakeService) -> None:
        self.recovery_service = service
        self.thread_path = "thread"  # value is irrelevant; loader is mocked
        self.pending_resume_message: str | None = None
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


# ── /resume with a yielded entry ──────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_surfaces_yield_question_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the thread tail has a ``yielded=True`` entry, the handler
    surfaces that entry's ``yield_question``."""
    entries = [
        _entry(turn=1),
        _entry(turn=2, yielded=True, yield_question="Which option do you prefer?"),
    ]

    def _fake_load(**_kwargs: Any) -> list[dict[str, Any]]:
        return list(entries)

    # handle_resume imports load_recent_thread_entries lazily from
    # host.thread; patch it there so the in-function import sees the stub.
    monkeypatch.setattr(
        "host.thread.load_recent_thread_entries",
        _fake_load,
        raising=False,
    )

    runtime = _FakeRuntime(_FakeService())
    result = await handle_resume(runtime, "/resume", "/resume")

    assert "Resumed from turn 2" in result
    assert "Which option do you prefer?" in result
    assert runtime.pending_resume_message is None


# ── /resume with no yielded entry ─────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_returns_no_yield_message_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no entry in the tail has ``yielded=True``, the handler reports
    there is nothing to resume."""
    entries = [_entry(turn=1), _entry(turn=2)]

    def _fake_load(**_kwargs: Any) -> list[dict[str, Any]]:
        return list(entries)

    monkeypatch.setattr(
        "host.thread.load_recent_thread_entries",
        _fake_load,
        raising=False,
    )

    runtime = _FakeRuntime(_FakeService())
    result = await handle_resume(runtime, "/resume", "/resume")

    assert result == "No yielded turn to resume."
    assert runtime.pending_resume_message is None


# ── /resume <reply> stashes the reply ─────────────────────────────────


@pytest.mark.asyncio
async def test_resume_with_reply_sets_pending_resume_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/resume <reply>`` stashes the reply in
    ``runtime.pending_resume_message`` for the next turn to replay."""
    entries = [
        _entry(turn=1, yielded=True, yield_question="Pick A or B?"),
    ]

    def _fake_load(**_kwargs: Any) -> list[dict[str, Any]]:
        return list(entries)

    monkeypatch.setattr(
        "host.thread.load_recent_thread_entries",
        _fake_load,
        raising=False,
    )

    runtime = _FakeRuntime(_FakeService())
    result = await handle_resume(runtime, "/resume option A", "/resume option A")

    assert "Resumed from turn 1" in result
    assert runtime.pending_resume_message == "option A"