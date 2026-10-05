"""Tests for thread-derived context items pushed into the retrieval lane."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from kernel.models import ContextItem
from runtime.thread_digest import build_thread_context_items
from sdk.thread.redaction import ThreadRedactionPolicy


def _write_entry(
    thread_path: Path,
    *,
    query: str = "",
    response_text: str = "",
    task_id: str = "",
    timestamp_utc: str = "2026-10-02T16:39:06+00:00",
    block_origin: str = "",
    provenance_extra: dict[str, object] | None = None,
    yielded: bool = False,
    yield_question: str = "",
) -> None:
    from sdk.thread._backend import _segment_paths

    paths = _segment_paths(thread_path)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, object] = dict(provenance_extra or {})
    if task_id:
        provenance["task_id"] = task_id
    if block_origin:
        provenance["block_origin"] = block_origin
    entry: dict[str, object] = {
        "mode_id": "executor",
        "timestamp_utc": timestamp_utc,
        "query": query,
        "response_text": response_text,
    }
    if yielded:
        entry["yielded"] = True
        entry["yield_question"] = yield_question
    if provenance:
        entry["provenance"] = provenance
    with paths.current.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_digest_settings_disable_and_resize_the_push(tmp_path: Path) -> None:
    thread_path = tmp_path / "thread"
    _write_entry(thread_path, query="turn one", response_text="one", task_id="t")
    _write_entry(thread_path, query="turn two", response_text="two", task_id="t")

    disabled = SimpleNamespace(enabled=False)
    assert build_thread_context_items(thread_path, settings=disabled) == []

    tight = SimpleNamespace(
        enabled=True,
        recent_entries=1,
        recent_body_chars=200,
        task_view_budget_bytes=2048,
        max_open_objective_lines=1,
    )
    items = build_thread_context_items(thread_path, settings=tight)
    history = next(item for item in items if item.item_id == "thread:recent_history")
    # Newest-first truncation plus the recorded-at timestamp and the
    # past/present closing directive.
    assert history.body.startswith("- [recorded 2026-10-02T16:39:06+00:00] turn two: two")
    assert history.body.count("\n- ") == 0  # only the newest exchange fits
    assert "not the live present" in history.body


def test_digest_returns_empty_without_thread(tmp_path: Path) -> None:
    assert build_thread_context_items(None) == []
    assert build_thread_context_items(tmp_path / "missing") == []


def test_digest_derives_task_state_history_and_objectives(tmp_path: Path) -> None:
    thread_path = tmp_path / "thread"
    _write_entry(thread_path, query="older turn", response_text="older answer")
    _write_entry(
        thread_path,
        query="fix the router bug",
        response_text="patched the routing logic",
        task_id="task-1",
        block_origin="task_completion",
    )

    items = build_thread_context_items(thread_path)

    by_id = {item.item_id: item for item in items}
    assert set(by_id) == {
        "thread:task_state",
        "thread:recent_history",
        "thread:open_objectives",
    }
    for item in items:
        assert isinstance(item, ContextItem)
        assert item.pack == ""
        assert item.namespace == "thread_history"
        assert item.metadata["provenance"] == "system_observed"
    task_state = by_id["thread:task_state"]
    assert "task-1" in task_state.body
    assert "recorded observation" in task_state.body
    assert "fix the router bug" in task_state.body
    assert task_state.metadata["selection"] == "latest_recorded"
    history = by_id["thread:recent_history"]
    assert "older turn" in history.body
    assert "fix the router bug" in history.body
    objectives = by_id["thread:open_objectives"]
    assert "blocked turn (task_completion)" in objectives.body
    assert "fix the router bug" in objectives.body


def test_digest_redacts_recorded_text(tmp_path: Path) -> None:
    thread_path = tmp_path / "thread"
    _write_entry(
        thread_path,
        query="I've hit my iteration budget in explainer",
        response_text="confidential payload",
        task_id="secret-task",
    )
    policy = ThreadRedactionPolicy.from_patterns(["confidential", "secret-task"])

    items = build_thread_context_items(thread_path, redaction=policy)

    body = "\n".join(item.body for item in items)
    # The default cap-fire pattern is not configured here; the custom
    # patterns must strikethrough both the response text and the task id.
    assert "confidential payload" not in body
    assert "secret-task" not in body
    assert "iteration budget" in body  # not in the configured pattern set


def test_digest_skips_open_objectives_when_turns_are_clean(tmp_path: Path) -> None:
    thread_path = tmp_path / "thread"
    _write_entry(thread_path, query="plain turn", response_text="done", task_id="t")

    items = build_thread_context_items(thread_path)

    assert [item.item_id for item in items] == [
        "thread:task_state",
        "thread:recent_history",
    ]


# ── past/present framing on every digest item ──────────────────────────


def test_digest_items_frame_recorded_state_as_past(tmp_path: Path) -> None:
    """Every digest body must close with the recorded-not-current directive.

    The recent_conversation lane and the readout pages already carry a
    recorded-history footer, but the thread digest items did not — the model
    read "status: awaiting_approval" as the live present (2026-10-05
    turn-510 stale-yield loop: it re-derived "approve request 4803a869"
    from a recorded blocker that no longer existed).
    """
    thread_path = tmp_path / "thread"
    _write_entry(thread_path, query="warm the build cache", response_text="started")
    _write_entry(
        thread_path,
        query="retry the check",
        response_text="Approval required for run shell command(s).",
        task_id="task-1",
        provenance_extra={"pending_approval_request_id": "4803a8691a87431f857c0043b875ed7b"},
    )

    items = build_thread_context_items(thread_path)

    for item in items:
        assert "recorded" in item.body.lower()
        assert "not verified live state" in item.body or "current turn" in item.body


def test_task_state_names_the_recorded_turn_and_the_live_baseline(tmp_path: Path) -> None:
    """Task state must anchor: recorded at turn N — verify against live tools."""
    thread_path = tmp_path / "thread"
    _write_entry(
        thread_path,
        query="fix the router bug",
        response_text="patched the routing logic",
        task_id="task-1",
    )

    items = build_thread_context_items(thread_path)
    task_state = next(item for item in items if item.item_id == "thread:task_state")

    assert "recorded at turn" in task_state.body
    assert "Live state is what tools return" in task_state.body


def test_recent_history_lines_carry_recorded_timestamps(tmp_path: Path) -> None:
    """History lines must carry their recorded-at timestamps, newest first."""
    thread_path = tmp_path / "thread"
    _write_entry(
        thread_path,
        query="first turn",
        response_text="first answer",
        timestamp_utc="2026-10-05T01:00:00+00:00",
    )
    _write_entry(
        thread_path,
        query="second turn",
        response_text="second answer",
        timestamp_utc="2026-10-05T02:00:00+00:00",
    )

    items = build_thread_context_items(thread_path)
    history = next(item for item in items if item.item_id == "thread:recent_history")

    assert history.body.index("second turn") < history.body.index("first turn")
    assert "[recorded 2026-10-05T02:00:00+00:00]" in history.body
    assert "[recorded 2026-10-05T01:00:00+00:00]" in history.body


# ── live approval reconciliation ───────────────────────────────────────


def _live_state(resolved: dict[str, str], stalled: set[str] | None = None) -> SimpleNamespace:
    """Build the live-state snapshot the host resolves before each turn.

    ``resolved`` maps approval request_id → live approval-store outcome
    ("consumed", "approved", "denied", "cancelled", "expired"); ``stalled``
    is the set of work_item_ids still open in the recovery store.
    """
    return SimpleNamespace(resolved_approvals=resolved, open_work_items=stalled or set())


def test_digest_marks_recorded_blocker_resolved_by_live_approval_state(tmp_path: Path) -> None:
    """A recorded awaiting-approval entry must be annotated when live state resolved it.

    2026-10-05 incident: approval 4803a869 was policy-approved, executed,
    killed by the 30s wrapper, and the continuation turn died with the
    process — the thread tail stayed "paused awaiting your decision" while
    the store said consumed. The model re-yielded the same stale question
    every turn. The digest must reconcile the recorded blocker against the
    live approval store and annotate the superseded state.
    """
    thread_path = tmp_path / "thread"
    _write_entry(
        thread_path,
        query="The approved action `bash_run` executed successfully and returned output",
        response_text=(
            "Approval required for run shell command(s).\n\nThis action is paused until "
            "you approve or deny it (request 4803a8691a87431f857c0043b875ed7b)."
        ),
        task_id="task-1",
        provenance_extra={"pending_approval_request_id": "4803a8691a87431f857c0043b875ed7b"},
    )

    items = build_thread_context_items(
        thread_path,
        live_state=_live_state({"4803a8691a87431f857c0043b875ed7b": "consumed"}),
    )

    by_id = {item.item_id: item for item in items}
    task_state = by_id["thread:task_state"]
    assert "request 4803a8691a87431f857c0043b875ed7b" in task_state.body
    assert "superseded (live approval store: consumed)" in task_state.body
    objectives = by_id["thread:open_objectives"]
    assert "superseded (live approval store: consumed)" in objectives.body
    # The recorded blocker text stays visible for audit, but the
    # reconciliation note must distinguish it from live state.
    assert "awaiting user" in objectives.body


def test_digest_without_live_state_leaves_recorded_blocker_unannotated(tmp_path: Path) -> None:
    """Without a live snapshot the digest preserves today's behavior."""
    thread_path = tmp_path / "thread"
    _write_entry(
        thread_path,
        query="paused for approval",
        response_text="Approval required for run shell command(s).",
        task_id="task-1",
        provenance_extra={"pending_approval_request_id": "deadbeef00000000"},
    )

    items = build_thread_context_items(thread_path)
    task_state = next(item for item in items if item.item_id == "thread:task_state")

    assert "superseded" not in task_state.body


def test_digest_live_state_unknown_request_stays_unannotated(tmp_path: Path) -> None:
    """A live snapshot that does not cover the request id changes nothing."""
    thread_path = tmp_path / "thread"
    _write_entry(
        thread_path,
        query="paused for approval",
        response_text="Approval required.",
        task_id="task-1",
        provenance_extra={"pending_approval_request_id": "cafe1111"},
    )

    items = build_thread_context_items(
        thread_path,
        live_state=_live_state({"deadbeef00000000": "consumed"}),
    )

    task_state = next(item for item in items if item.item_id == "thread:task_state")
    assert "superseded" not in task_state.body


# ── live-state resolution helpers ──────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_live_workflow_state_reads_only_named_requests(tmp_path: Path) -> None:
    """The resolver snapshots only the request ids the thread tail names."""
    from runtime.thread_digest import resolve_live_workflow_state

    class FakeStore:
        def __init__(self) -> None:
            self.asked: list[str] = []

        async def get(self, request_id: str) -> SimpleNamespace:
            self.asked.append(request_id)
            return SimpleNamespace(state="consumed")

    service = FakeStore()
    runtime = SimpleNamespace(services=SimpleNamespace(approval_service=service))

    live = await resolve_live_workflow_state(
        runtime, request_ids=["4803a8691a87431f857c0043b875ed7b"]
    )

    assert live is not None
    assert live.resolved_approvals == {"4803a8691a87431f857c0043b875ed7b": "consumed"}
    assert service.asked == ["4803a8691a87431f857c0043b875ed7b"]


@pytest.mark.asyncio
async def test_resolve_live_workflow_state_degrades_to_none(tmp_path: Path) -> None:
    """No ids, no service, or a failing store all degrade to None."""
    from runtime.thread_digest import resolve_live_workflow_state

    assert await resolve_live_workflow_state(SimpleNamespace(), request_ids=[]) is None
    assert (
        await resolve_live_workflow_state(
            SimpleNamespace(services=SimpleNamespace(approval_service=None)),
            request_ids=["x"],
        )
        is None
    )

    class BrokenStore:
        async def get(self, request_id: str) -> None:
            raise RuntimeError("store offline")

    live = await resolve_live_workflow_state(
        SimpleNamespace(services=SimpleNamespace(approval_service=BrokenStore())),
        request_ids=["x"],
    )
    assert live is None


def test_recorded_approval_request_ids_collects_tail_blockers(tmp_path: Path) -> None:
    """The tail collector returns the newest-first distinct request ids."""
    from runtime.thread_digest import recorded_approval_request_ids

    thread_path = tmp_path / "thread"
    _write_entry(
        thread_path,
        query="older pause",
        response_text="Approval required (request 11111111).",
        provenance_extra={"pending_approval_request_id": "11111111"},
    )
    _write_entry(thread_path, query="clean turn", response_text="done")
    _write_entry(
        thread_path,
        query="newer pause",
        response_text="Approval required (request 22222222).",
        provenance_extra={"pending_approval_request_id": "22222222"},
    )
    _write_entry(
        thread_path,
        query="repeat pause",
        response_text="Approval required (request 22222222).",
        provenance_extra={"pending_approval_request_id": "22222222"},
    )

    assert recorded_approval_request_ids(thread_path) == ["22222222", "11111111"]
    assert recorded_approval_request_ids(None) == []
    assert recorded_approval_request_ids(tmp_path / "missing") == []
