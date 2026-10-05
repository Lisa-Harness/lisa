"""Tests for thread records of operator host commands that settle workflow state.

The 2026-10-05 stale-yield loop: ``/cancel``/``/cancel-all`` settled the
recovery items backing a paused approval (the operator's explicit cleanup at
02:57:51Z), but no thread entry recorded the decision — the thread tail still
ended at "paused awaiting your decision (request 4803a869)". Every later
turn re-assembled that stale tail, and the model re-yielded asking the
operator to decide an already-cancelled approval. State-mutating operator
commands must leave the same durable record ``/approve`` does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from runtime.handle_message import _dispatch_host_command


class _FakeTransport:
    """Transport capture for dispatch tests."""

    def __init__(self) -> None:
        self.responses: list[str] = []
        self.errors: list[str] = []

    async def on_response(self, text: str, data: Any = None) -> None:
        self.responses.append(text)

    async def on_error(self, text: str, exc: Any = None) -> None:
        self.errors.append(text)

    async def on_stage(self, label: str, data: dict[str, Any]) -> None:
        pass

    async def on_debug(self, label: str, data: dict[str, Any]) -> None:
        pass

    async def on_token_update(self, payload: dict[str, Any]) -> None:
        pass


def _read_entries(thread_base: Path) -> list[dict[str, Any]]:
    from sdk.thread._backend import _segment_paths

    paths = _segment_paths(thread_base)
    if not paths.current.exists():
        return []
    lines = paths.current.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _fake_runtime(tmp_path: Path) -> Any:
    """Minimal runtime stub: paths + services + config + session."""

    class _FakeRecoveryService:
        def __init__(self) -> None:
            self.settled: list[tuple[str, str]] = []

        async def list_open(self, identity: Any) -> list[Any]:
            return []

        async def settle(self, work_item_id: str, identity: Any, **kwargs: Any) -> Any:
            self.settled.append((work_item_id, str(kwargs.get("state") or "")))
            return None

    from types import SimpleNamespace

    async def _ensure_pool_ready() -> None:
        return None

    return SimpleNamespace(
        paths=SimpleNamespace(thread_path=tmp_path / "thread", data_dir=tmp_path),
        config=SimpleNamespace(
            user_slug="user",
            thread_segment_turns=256,
            surface_type="personal_dm",
            surface_id="surface",
            memory_scope="personal",
        ),
        session=SimpleNamespace(
            current_mode_id="default",
            auto_approved_this_turn=0,
            restart_required=False,
            restart_reason="",
            state_snapshot={},
        ),
        services=SimpleNamespace(
            recovery_service=_FakeRecoveryService(),
            approval_service=None,
        ),
        session_plan=SimpleNamespace(
            manifest=SimpleNamespace(
                name="workspace", context=SimpleNamespace(history=SimpleNamespace(digest=None))
            ),
            plan_digest="digest",
        ),
        start_mode_id="",
        host_commands=SimpleNamespace(),
        ensure_pool_ready=_ensure_pool_ready,
    )


@pytest.mark.asyncio
async def test_state_settling_pack_command_records_thread_entry(tmp_path: Path) -> None:
    """A pack command that settles workflow state must append a thread entry."""
    runtime = _fake_runtime(tmp_path)

    async def _cancel_all_handler(runtime: Any, stripped: str, message: str) -> str:
        await runtime.services.recovery_service.settle(
            "item-1", None, state="cancelled", last_error="cancelled by operator"
        )
        return "Cancelled 1/1 work items."

    import runtime.handle_message as hm

    class _Spec:
        prefix = "/cancel-all"
        handler = staticmethod(_cancel_all_handler)
        help_text = ""

    class _Entry:
        specs: ClassVar[list[Any]] = [_Spec()]

    monkeypatch_entry = _Entry()
    original = hm._pack_host_commands
    hm._pack_host_commands = lambda: [monkeypatch_entry]
    try:
        result = await _dispatch_host_command(
            runtime, "/cancel-all", "/cancel-all", "/cancel-all", _FakeTransport()
        )
    finally:
        hm._pack_host_commands = original

    assert result is not None
    entries = _read_entries(tmp_path / "thread")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["query"] == "/cancel-all"
    assert "Cancelled 1/1 work items." in entry["response_text"]
    provenance = entry["provenance"]
    assert provenance["source"] == "operator-host-command"
    assert provenance["state_change"] == "recovery_settled"


@pytest.mark.asyncio
async def test_deny_command_records_thread_entry(tmp_path: Path) -> None:
    """/deny settles an approval; the decision must land in the thread."""
    runtime = _fake_runtime(tmp_path)

    class _FakeApprovalService:
        async def deny(self, request_id: str, identity: Any, *, reason: str = "") -> bool:
            return True

        async def get(self, request_id: str) -> Any:
            return None

    runtime.services.approval_service = _FakeApprovalService()

    import runtime.handle_message as hm

    original = hm._dispatch_approval_command

    async def _fake_approval_dispatch(
        runtime: Any, stripped: str, lowered: str, transport: Any
    ) -> tuple[str | None, Any]:
        return "Denied approval request deadbeef.", None

    hm._dispatch_approval_command = _fake_approval_dispatch
    try:
        result = await _dispatch_host_command(
            runtime, "/deny deadbeef", "/deny deadbeef", "/deny deadbeef", _FakeTransport()
        )
    finally:
        hm._dispatch_approval_command = original

    assert result is not None
    entries = _read_entries(tmp_path / "thread")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["query"] == "/deny deadbeef"
    assert "Denied approval request" in entry["response_text"]
    assert entry["provenance"]["source"] == "operator-host-command"


@pytest.mark.asyncio
async def test_read_only_command_records_no_entry(tmp_path: Path) -> None:
    """``/stalls`` is an operator inspection, not a decision — no entry.

    Only state-settling commands record; read-only commands keep the quiet
    behavior so the thread tail reflects decisions, not every glance.
    """
    runtime = _fake_runtime(tmp_path)

    import runtime.handle_message as hm

    async def _stalls_handler(runtime: Any, stripped: str, message: str) -> str:
        return "No stalled or indeterminate work items."

    class _Spec:
        prefix = "/stalls"
        handler = staticmethod(_stalls_handler)
        help_text = ""

    class _Entry:
        specs: ClassVar[list[Any]] = [_Spec()]

    original = hm._pack_host_commands
    hm._pack_host_commands = lambda: [_Entry()]
    try:
        result = await _dispatch_host_command(
            runtime, "/stalls", "/stalls", "/stalls", _FakeTransport()
        )
    finally:
        hm._pack_host_commands = original

    assert result is not None
    assert _read_entries(tmp_path / "thread") == []