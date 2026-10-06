"""Tests for replaying the latest turn and its unfinished task from the thread."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from runtime.host_commands import register_builtin_host_commands
from runtime.thread import (
    append_thread_entry,
    build_open_task_messages,
    project_thread_messages,
    set_thread_redaction_policy,
)
from sdk.thread import ThreadReader, ThreadRedactionPolicy, ThreadStore, result_object_path

register_builtin_host_commands()


def _append(thread_path: Path, query: str, **kwargs: object) -> None:
    append_thread_entry(
        thread_path=thread_path,
        user_slug="tester",
        mode_id=str(kwargs.pop("mode_id", "worker")),
        query=query,
        response_text=str(kwargs.pop("response_text", "")),
        **kwargs,  # type: ignore[arg-type]
    )


def _read_turn(turn_query: str, output: str, call_id: str = "c1") -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": f"## Pre-turn Briefing\n\n...\n\n## User Query\n\n{turn_query}",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": output},
    ]


def test_task_view_recovers_raw_and_approved_results_from_runtime_records(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    original = "FULL FILE DATA " * 800
    transcript = _read_turn("update parser", original)
    projected = project_thread_messages(transcript, "task-1", thread_path=thread)
    assert json.loads(projected[-1]["content"])["abbreviated"] is True
    _append(
        thread,
        "update parser",
        provenance={"task_id": "task-1", "pending_approval_request_id": "approval-1"},
        transcript=transcript,
        effect_receipts=[{"capability_id": "files.read", "effect": "read", "artifact_id": ""}],
    )
    _append(
        thread,
        "/approve approval-1",
        response_text="parser updated",
        provenance={"task_id": "task-1", "source": "host-approved"},
        effect_receipts=[
            {"capability_id": "files.write", "effect": "write", "artifact_id": "change-1"}
        ],
    )
    reader = ThreadReader(ThreadStore(thread))
    view = json.loads(reader.task_view("task-1"))
    assert view["status"] == "open"
    assert "blocker" not in view
    assert view["objective"]["text"] == "update parser"
    assert view["effects"] == [
        {"capability_id": "files.read", "effect": "read", "count": 1, "artifact_ids": []},
        {
            "capability_id": "files.write",
            "effect": "write",
            "count": 1,
            "artifact_ids": ["change-1"],
        },
    ]
    assert view["evidence"] == ["t0.m2.content", "t1.response_text"]
    restored = [
        json.loads(reader.read_content(ref, limit_chars=20000))["content"]
        for ref in view["evidence"]
    ]
    assert restored == [original, "parser updated"]
    assert ThreadStore(thread).load_range(start_turn=0, end_turn=0)[0]["transcript"] == transcript


def test_open_task_replays_paused_chain_verbatim(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    _append(thread, "unrelated earlier", response_text="done")
    _append(
        thread,
        "fix issue 6",
        response_text="budget paused",
        provenance={"source": "observed", "operator_exit": "paused"},
        transcript=_read_turn("fix issue 6", "FULL FILE A"),
    )
    _append(thread, "/models apply", mode_id="models", response_text="applied")
    _append(
        thread,
        "continue",
        response_text="paused again",
        provenance={"source": "observed", "operator_exit": "paused"},
        transcript=_read_turn("continue", "FULL FILE B"),
    )

    messages = build_open_task_messages(thread_path=thread)

    contents = [m.get("content") for m in messages]
    assert "unrelated earlier" not in contents
    assert "/models apply" not in contents
    assert contents[0] == "fix issue 6"
    assert "FULL FILE A" in contents and "FULL FILE B" in contents
    call_ids = [c["id"] for m in messages for c in m.get("tool_calls") or []]
    assert len(set(call_ids)) == 2
    assert all(cid.startswith("t") for cid in call_ids)


@pytest.mark.parametrize("failed", [False, True])
def test_approved_action_replays_as_original_tool_result(tmp_path: Path, failed: bool) -> None:
    thread = tmp_path / "threads" / "tester"
    _append(
        thread,
        "fix issue 6",
        response_text="approval required",
        provenance={
            "pending_approval_request_id": "approval-1",
            "pending_approval_tool_call_id": "call-2",
        },
        transcript=[
            {"role": "user", "content": "fix issue 6"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "file contents"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "write", "arguments": "{}"},
                    }
                ],
            },
        ],
    )
    _append(
        thread,
        "/approve approval-1",
        response_text="Approved action failed: invalid patch" if failed else "saved file",
        provenance={
            "source": "host-approved-failed" if failed else "host-approved",
            "finish_reason": "approved-action-failed" if failed else "approved-action",
            "approval_request_id": "approval-1",
            "target_id": "write",
        },
        effect_receipts=[]
        if failed
        else [{"capability_id": "write", "effect": "write", "artifact_id": "doc-1"}],
    )

    messages = build_open_task_messages(thread_path=thread)

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert messages[-1]["tool_call_id"] == messages[-2]["tool_calls"][0]["id"]
    expected = "Approved action failed: invalid patch" if failed else "saved file"
    assert messages[-1]["content"] == expected
    assert all("/approve" not in str(message.get("content", "")) for message in messages)


def test_open_task_replay_redacts_on_read(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    _append(
        thread,
        "read secrets",
        response_text="ok",
        transcript=_read_turn("read secrets", "token=SECRET123"),
    )
    set_thread_redaction_policy(ThreadRedactionPolicy.from_patterns((r"SECRET\d+",)))
    try:
        messages = build_open_task_messages(thread_path=thread)
    finally:
        set_thread_redaction_policy(None)

    assert "SECRET123" not in str(messages)
    assert "SECRET123" in thread.with_suffix(".current.jsonl").read_text(encoding="utf-8")


def test_open_task_replay_does_not_duplicate_transcript_response(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    _append(
        thread,
        "finish task",
        response_text="final answer",
        transcript=[
            {"role": "user", "content": "finish task"},
            {"role": "assistant", "content": "final answer"},
        ],
    )

    messages = build_open_task_messages(thread_path=thread)

    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_large_live_and_replayed_results_are_durable_and_lossless(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    original = "full output " * 10000
    transcript = _read_turn("read", original)
    live = project_thread_messages(transcript, "task-1", thread_path=thread)
    stub = json.loads(live[-1]["content"])
    assert stub["abbreviated"]
    assert len(live[-1]["content"].encode("utf-8")) < 8192
    assert transcript[-1]["content"] == original
    assert project_thread_messages(transcript, "task-1", thread_path=thread) == live
    _append(thread, "read", transcript=transcript, provenance={"task_id": "task-1"})
    replayed = build_open_task_messages(thread_path=thread)
    replay_stub = json.loads(
        next(message for message in replayed if message["role"] == "tool")["content"]
    )
    assert replay_stub["abbreviated"]
    assert replay_stub["ref"] == stub["ref"]
    stored = json.loads(result_object_path(thread, stub["ref"]).read_text())
    assert stored["task_id"] == "task-1"
    reader = ThreadReader(ThreadStore(thread))
    offset = 0
    restored: list[str] = []
    while True:
        page = json.loads(reader.read_content(stub["ref"], offset))
        restored.append(page["content"])
        if not page["has_more"]:
            break
        offset = page["next_offset_chars"]
    assert "".join(restored) == original
    raw = ThreadStore(thread).load_range(start_turn=0, end_turn=0)[0]
    assert raw["transcript"][-1]["content"] == original


def test_live_result_redaction_and_integrity_are_enforced(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    original = "secret " * 5000
    policy = ThreadRedactionPolicy.from_patterns(("secret",))
    set_thread_redaction_policy(policy)
    try:
        projected = project_thread_messages(_read_turn("read", original), thread_path=thread)
    finally:
        set_thread_redaction_policy(None)
    stub = json.loads(projected[-1]["content"])
    assert "secret" not in projected[-1]["content"]
    reader = ThreadReader(ThreadStore(thread), policy)
    assert "secret" not in reader.read_content(stub["ref"])
    assert json.loads(reader.find_content(stub["ref"], "secret"))["matches"] == []
    path = result_object_path(thread, stub["ref"])
    assert json.loads(path.read_text())["content"] == original
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_text('{"content":"tampered"}')
    with pytest.raises(ValueError, match="integrity"):
        reader.read_content(stub["ref"])


@pytest.mark.parametrize(
    "tool_name", ["thread_search", "thread_read_content", "thread_find_content"]
)
def test_bounded_retrieval_pages_remain_readable(tmp_path: Path, tool_name: str) -> None:
    thread = tmp_path / "threads" / "tester"
    content = "bounded requested evidence " * 400
    messages = _read_turn("read", content)
    messages[1]["tool_calls"][0]["function"]["name"] = f"lisa-thread.{tool_name}"
    projected = project_thread_messages(messages, thread_path=thread)
    assert projected[-1]["content"] == content


def test_tool_context_compaction_preserves_pairing_and_inventory(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    raw = []
    for index in range(200):
        raw.extend(_read_turn("read", f"result {index} " * 100, call_id=f"call-{index}")[1:])
    projected = project_thread_messages(raw, "task-1", thread_path=thread)
    tool_bytes = sum(
        len(message["content"].encode("utf-8"))
        for message in projected
        if message["role"] == "tool"
    )
    assert tool_bytes <= 65536
    calls = [call["id"] for message in projected for call in message.get("tool_calls") or []]
    results = [message["tool_call_id"] for message in projected if message["role"] == "tool"]
    assert calls == results
    assert len(calls) < 200
    marker = next(message for message in projected if "inventory_ref" in message.get("content", ""))
    inventory_ref = json.loads(marker["content"])["inventory_ref"]
    reader = ThreadReader(ThreadStore(thread))
    inventory_page = json.loads(reader.read_content(inventory_ref))
    assert inventory_page["content"].startswith('[{"tool_calls":')
    assert inventory_page["has_more"]
    assert json.loads(reader.find_content(inventory_ref, "call-0"))["matches"]
    inventory = json.loads(result_object_path(thread, inventory_ref).read_text())["content"]
    groups = json.loads(inventory)
    assert len(groups) + len(calls) == 200
    first_ref = groups[0]["results"][0]["ref"]
    first = json.loads(reader.read_content(first_ref))
    assert first["content"] == raw[1]["content"]
    assert raw[1]["content"].startswith("result 0 ")


def test_result_refs_cannot_follow_symlinks_or_cross_threads(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    projected = project_thread_messages(_read_turn("read", "data " * 3000), thread_path=first)
    ref = json.loads(projected[-1]["content"])["ref"]
    with pytest.raises(ValueError, match="not found"):
        ThreadReader(ThreadStore(second)).read_content(ref)
    outside = tmp_path / "outside.json"
    outside.write_text('{"content":"outside"}')
    second_path = result_object_path(second, ref)
    second_path.parent.mkdir()
    second_path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        ThreadReader(ThreadStore(second)).read_content(ref)


def test_reprojection_reuses_original_result_references_under_pressure(tmp_path: Path) -> None:
    thread = tmp_path / "thread"
    replayed = []
    original_refs = set()
    for index in range(200):
        turn = _read_turn("read", f"large result {index} " * 2000, call_id=f"call-{index}")[1:]
        preview = project_thread_messages(turn, "task-1", thread_path=thread)
        original_refs.add(json.loads(preview[-1]["content"])["ref"])
        replayed.extend(preview)
    projected = project_thread_messages(replayed, "task-1", thread_path=thread)
    for message in projected:
        if message.get("role") == "tool":
            assert json.loads(message["content"])["ref"] in original_refs
    marker = next(message for message in projected if "inventory_ref" in message.get("content", ""))
    inventory_ref = json.loads(marker["content"])["inventory_ref"]
    groups = json.loads(
        json.loads(result_object_path(thread, inventory_ref).read_text())["content"]
    )
    assert all(result["ref"] in original_refs for group in groups for result in group["results"])
    reader = ThreadReader(ThreadStore(thread))
    assert json.loads(reader.read_content(groups[0]["results"][0]["ref"]))["content"].startswith(
        "large result 0 "
    )
    assert project_thread_messages(projected, "task-1", thread_path=thread) == projected


class TestVerifiedFactsExtraction:
    """An approved tool result carrying ``output_json.fact`` attaches it."""

    def test_approval_results_extract_fact_only_from_fact_tool(self) -> None:
        from kernel.models import ToolResult
        from runtime.approval.orchestration import _fact_from_approval_result

        fact = {"claim": "fd 4 is the Unix socket", "at": "daemon.go:124", "revision": "0" * 40}
        carrying = ToolResult(
            tool_id="coding_record_fact",
            output_text="recorded",
            output_json={"fact": fact},
        )
        assert _fact_from_approval_result("coding_record_fact", carrying) == [fact]

    def test_other_tools_and_error_results_are_ignored(self) -> None:
        from kernel.models import ToolResult
        from runtime.approval.orchestration import _fact_from_approval_result

        fact = {"claim": "x", "at": "y:1", "revision": "0" * 40}
        assert (
            _fact_from_approval_result(
                "coding_commit", ToolResult(tool_id="coding_commit", output_json={"fact": fact})
            )
            is None
        )
        assert (
            _fact_from_approval_result(
                "coding_record_fact",
                ToolResult(tool_id="coding_record_fact", error="revision mismatch"),
            )
            is None
        )
        assert (
            _fact_from_approval_result(
                "coding_record_fact", ToolResult(tool_id="coding_record_fact")
            )
            is None
        )

    def test_fact_round_trips_through_approved_turn(self, tmp_path: Path) -> None:
        """The fact lands on the recorded approved turn and survives replay."""
        thread = tmp_path / "threads" / "tester"
        _append(
            thread,
            "/approve f1",
            response_text="Recorded fact on abc123: fd 4 is the Unix socket",
            provenance={"source": "host-approved", "target_id": "coding_record_fact"},
            effect_receipts=[
                {"capability_id": "coding_record_fact", "effect": "write", "artifact_id": "abc123"}
            ],
            verified_facts=[
                {"claim": "fd 4 is the Unix socket", "at": "daemon.go:124", "revision": "0" * 40}
            ],
        )

        from sdk.thread import ThreadStore

        entry = ThreadStore(thread).load_recent(limit=1)[0]
        assert entry["verified_facts"][0]["claim"] == "fd 4 is the Unix socket"
