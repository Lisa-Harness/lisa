"""Tests for replaying the latest turn and its unfinished task from the thread."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from runtime.host_commands import register_builtin_host_commands
from runtime.thread import (
    append_thread_entry,
    build_open_task_messages,
    project_thread_messages,
    set_thread_redaction_policy,
)
from runtime.thread_digest import build_thread_context_items
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


def test_resumed_turn_replays_latest_checkpoint_without_duplicating_history(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    transcript = _read_turn("original request", "first result")
    _append(
        thread,
        "original request",
        transcript=transcript,
        provenance={"kernel_turn_id": "same-turn", "pending_approval_request_id": "approval"},
    )
    _append(
        thread,
        "original request",
        transcript=[*transcript, {"role": "assistant", "content": "finished"}],
        provenance={"kernel_turn_id": "same-turn"},
        response_text="finished",
    )
    messages = build_open_task_messages(thread_path=thread)
    assert sum(message.get("content") == "first result" for message in messages) == 1
    assert sum(message.get("content") == "original request" for message in messages) == 1
    assert not any("approval" in str(message.get("content")) for message in messages)


def test_opaque_provider_context_survives_durable_thread_replay(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    transcript = _read_turn("check weather", "sunny")
    continuation = {
        "provider": "azure_openai",
        "model_id": "reasoning-model",
        "items": [{"type": "reasoning", "encrypted_content": "opaque", "summary": []}],
    }
    transcript[1]["provider_continuation"] = continuation
    _append(thread, "check weather", transcript=transcript)
    messages = build_open_task_messages(thread_path=thread)
    assistant = next(message for message in messages if message.get("role") == "assistant")
    assert assistant["provider_continuation"] == continuation


def test_task_view_recovers_raw_and_approved_results_from_runtime_records(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    original = "FULL FILE DATA " * 800
    transcript = _read_turn("update parser", original)
    projected = project_thread_messages(transcript, "task-1", thread_path=thread, budget_bytes=4096)
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


def _call(call_id: str, name: str, arguments: str, *, flat: bool) -> dict[str, object]:
    # Provider gateways record the flat OpenAI shape; synthesized calls nest under "function".
    if flat:
        return {"id": call_id, "type": "function", "name": name, "arguments": arguments}
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_approved_action_replays_as_original_tool_result(
    tmp_path: Path, failed: bool, flat: bool
) -> None:
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
                "tool_calls": [_call("call-1", "read", "{}", flat=flat)],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "file contents"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("call-2", "write", "{}", flat=flat)],
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


def test_open_task_replay_redacts_flat_tool_call_arguments(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    _append(
        thread,
        "write secrets",
        response_text="ok",
        transcript=[
            {"role": "user", "content": "write secrets"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("c1", "write", '{"token": "SECRET123"}', flat=True)],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "saved"},
        ],
    )
    set_thread_redaction_policy(ThreadRedactionPolicy.from_patterns((r"SECRET\d+",)))
    try:
        messages = build_open_task_messages(thread_path=thread)
    finally:
        set_thread_redaction_policy(None)

    assert "SECRET123" not in str(messages)


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
    replayed = build_open_task_messages(thread_path=thread, budget_bytes=65536)
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


def test_inventory_refs_expand_to_original_bytes_not_stub_json(tmp_path: Path) -> None:
    """Coherence by address (R1): a ref in any stub or inventory row must
    expand to the original permitted content in one hop — never to a nested
    preview envelope (a stub-of-a-stub).

    The bug: over-threshold results under projection pressure got *graduated*
    preview scales (the 600→120 head ramp, removed 2026-10-08; such stubs
    persist in old threads). ``_existing_result`` recognized
    only the two exact endpoint scale pairs, so a graduated stub that is
    later evicted by the working-set compactor is treated as fresh content
    and persisted as-is — the inventory ref then expands to the stub's own
    JSON instead of the original bytes. Reproduction: two 50KB results
    (pressure graduated stubs) + ``budget_bytes=800`` (inventory eviction).
    """
    thread = tmp_path / "threads" / "tester"
    transcript = _scan_turn(n_results=2, result_size=50_000)
    _append(
        thread,
        "scan",
        response_text="paused",
        provenance={"source": "observed", "operator_exit": "paused"},
        transcript=transcript,
    )

    replayed = build_open_task_messages(thread_path=thread, budget_bytes=800)

    reader = ThreadReader(ThreadStore(thread))
    refs: list[str] = []
    for message in replayed:
        content = str(message.get("content") or "")
        if not content.startswith("{"):
            continue
        stub = json.loads(content)
        if "ref" in stub and message.get("role") == "tool":
            refs.append(stub["ref"])
        if "inventory_ref" in stub:
            inventory = json.loads(reader.read_content(stub["inventory_ref"], limit_chars=8000))[
                "content"
            ]
            refs.extend(
                row_ref
                for group in json.loads(inventory)
                for row_ref in [r["ref"] for r in group.get("results", [])]
            )
    assert refs, "expected stub refs and/or inventory rows"
    reader_for_refs = ThreadReader(ThreadStore(thread))
    for ref in refs:
        offset = 0
        restored: list[str] = []
        while True:
            page = json.loads(reader_for_refs.read_content(ref, offset, limit_chars=20000))
            restored.append(page["content"])
            if not page["has_more"]:
                break
            offset = page["next_offset_chars"]
        assert "".join(restored) == "x" * 50_000, (
            "ref expanded to something other than the original bytes "
            "(stub-of-a-stub); coherence by address is broken"
        )


def _scan_turn(n_results: int, result_size: int) -> list[dict[str, object]]:
    calls = [
        {
            "id": f"c{i}",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }
        for i in range(n_results)
    ]
    return [
        {"role": "user", "content": "scan"},
        {"role": "assistant", "content": "", "tool_calls": calls},
        *[
            {"role": "tool", "tool_call_id": f"c{i}", "content": "x" * result_size}
            for i in range(n_results)
        ],
    ]


def _tool_bytes(messages: list[dict[str, Any]]) -> int:
    return sum(
        len(str(m.get("content") or "").encode("utf-8"))
        for m in messages
        if m.get("role") == "tool"
    )


def test_replay_working_set_is_bounded_by_budget_bytes(tmp_path: Path) -> None:
    """The resident working set is recomputed under a byte budget at the turn
    boundary (R2): six individually sub-threshold results (6KB each, below the
    8KB inline threshold so none stubs on its own) accumulate to 36KB verbatim
    with no budget; under ``budget_bytes`` they collapse to recoverable
    ref-stubs — residency bounded, coherence by address, nothing lost.
    """
    thread = tmp_path / "threads" / "tester"
    _append(
        thread,
        "scan",
        response_text="paused",
        provenance={"source": "observed", "operator_exit": "paused"},
        transcript=_scan_turn(n_results=6, result_size=6000),
    )

    messages = build_open_task_messages(thread_path=thread, budget_bytes=12_000)

    # Residency is bounded (verbatim replay was 36KB).
    assert _tool_bytes(messages) <= 12_000
    # Coherence by address: every collapsed result carries a ref that
    # re-expands losslessly to its full 6000 chars.
    refs = [
        json.loads(m["content"])["ref"]
        for m in messages
        if m.get("role") == "tool" and str(m.get("content") or "").startswith('{"ref"')
    ]
    assert refs, "expected collapsed results to enter as ref-stubs"
    reader = ThreadReader(ThreadStore(thread))
    for ref in refs:
        page = json.loads(reader.read_content(ref, limit_chars=9000))
        assert page["content"] == "x" * 6000


def test_replay_working_set_unbounded_without_budget(tmp_path: Path) -> None:
    """With no budget_bytes the prior verbatim behavior is unchanged."""
    thread = tmp_path / "threads" / "tester"
    _append(
        thread,
        "scan",
        response_text="paused",
        provenance={"source": "observed", "operator_exit": "paused"},
        transcript=_scan_turn(n_results=6, result_size=6000),
    )
    messages = build_open_task_messages(thread_path=thread)
    assert _tool_bytes(messages) == 36_000


def test_live_result_redaction_and_integrity_are_enforced(tmp_path: Path) -> None:
    thread = tmp_path / "threads" / "tester"
    original = "secret " * 5000
    policy = ThreadRedactionPolicy.from_patterns(("secret",))
    set_thread_redaction_policy(policy)
    try:
        projected = project_thread_messages(
            _read_turn("read", original), thread_path=thread, budget_bytes=4096
        )
    finally:
        set_thread_redaction_policy(None)
    stub = json.loads(projected[-1]["content"])
    assert "secret" not in projected[-1]["content"]
    reader = ThreadReader(ThreadStore(thread), policy)
    assert "secret" not in reader.read_content(stub["ref"])
    assert json.loads(reader.find_content(stub["ref"], "secret"))["matches"] == []
    path = result_object_path(thread, stub["ref"])
    assert json.loads(path.read_text())["content"] == original
    # Owner-only persistence: POSIX can assert the exact 0o600 mode; Windows
    # os.chmod cannot clear group/other write bits (only the read-only bit
    # maps), so there the assertion is the owner-writable bit (the strict
    # mode check stays POSIX-only).
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    else:
        assert path.stat().st_mode & 0o200
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
    projected = project_thread_messages(
        _read_turn("read", "data " * 3000), thread_path=first, budget_bytes=4096
    )
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
        preview = project_thread_messages(turn, "task-1", thread_path=thread, budget_bytes=4096)
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
        _append(
            thread, "continue audit", response_text="new work", provenance={"task_id": "task-2"}
        )

        facts = next(
            item
            for item in build_thread_context_items(thread)
            if item.item_id == "thread:verified_facts"
        )
        assert "fd 4 is the Unix socket" in facts.body
        assert "daemon.go:124" in facts.body
