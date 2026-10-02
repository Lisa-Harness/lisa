"""Tests for thread-derived context items pushed into the retrieval lane."""

from __future__ import annotations

import json
from pathlib import Path

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
    if provenance:
        entry["provenance"] = provenance
    with paths.current.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
