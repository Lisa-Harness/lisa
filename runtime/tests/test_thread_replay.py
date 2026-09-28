"""Tests for replaying the latest turn and its unfinished task from the thread."""

from __future__ import annotations

from pathlib import Path

from runtime.host_commands import register_builtin_host_commands
from runtime.thread import (
    append_thread_entry,
    build_open_task_messages,
    set_thread_redaction_policy,
)
from sdk.thread import ThreadRedactionPolicy

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
