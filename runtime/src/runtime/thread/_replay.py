"""Replay the latest turn and its unfinished task as verbatim model messages.

The thread is the lossless record; this read path turns the most recent
entry — plus every immediately preceding entry that belongs to the same
unfinished task (paused, awaiting approval, or host-approved) — back into
user/assistant/tool messages at full fidelity. Redaction is applied on read,
matching every other thread projection.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from runtime.host_commands import is_host_command_query

_SCAN_LIMIT = 50


def _is_task_link(entry: dict[str, Any]) -> bool:
    provenance = entry.get("provenance") or {}
    return bool(
        provenance.get("operator_exit") == "paused"
        or provenance.get("pending_approval_request_id")
        or provenance.get("source") == "host-approved"
    )


def _open_task_span(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the newest entry plus the unfinished-task entries directly before it."""
    relevant = [
        e
        for e in entries
        if (e.get("provenance") or {}).get("source") == "host-approved"
        or not is_host_command_query(str(e.get("query") or ""))
    ]
    if not relevant:
        return []
    span = [relevant[-1]]
    for entry in reversed(relevant[:-1]):
        if not _is_task_link(entry):
            break
        span.insert(0, entry)
    return span


def _redact(value: str, redaction: Any) -> str:
    return redaction.redact(value) if redaction is not None else value


def _replay_message(message: dict[str, Any], prefix: str, redaction: Any) -> dict[str, Any]:
    replayed = copy.deepcopy(message)
    if isinstance(replayed.get("content"), str):
        replayed["content"] = _redact(replayed["content"], redaction)
    if replayed.get("tool_call_id"):
        replayed["tool_call_id"] = f"{prefix}{replayed['tool_call_id']}"
    for call in replayed.get("tool_calls") or []:
        call["id"] = f"{prefix}{call.get('id', '')}"
        function = call.get("function") or {}
        if isinstance(function.get("arguments"), str):
            function["arguments"] = _redact(function["arguments"], redaction)
    return replayed


def _entry_messages(entry: dict[str, Any], redaction: Any) -> list[dict[str, Any]]:
    # Tool-call ids are only unique within one turn; prefix them per turn.
    prefix = f"t{entry.get('turn', 'x')}_"
    messages = [{"role": "user", "content": _redact(str(entry.get("query") or ""), redaction)}]
    transcript = list(entry.get("transcript") or [])
    # The stored user message carries this turn's briefings; replay the plain query instead.
    if transcript and transcript[0].get("role") == "user":
        transcript = transcript[1:]
    messages.extend(_replay_message(m, prefix, redaction) for m in transcript)
    response = str(entry.get("response_text") or "")
    last_message = transcript[-1] if transcript else None
    if response and not (
        last_message is not None
        and last_message.get("role") == "assistant"
        and last_message.get("content") == response
    ):
        messages.append({"role": "assistant", "content": _redact(response, redaction)})
    return messages


def build_open_task_messages(*, thread_path: Path | None) -> list[dict[str, Any]]:
    """Return the latest turn and its unfinished task as replayable messages."""
    from runtime.thread import _thread_redaction, load_recent_thread_entries

    entries = load_recent_thread_entries(thread_path=thread_path, limit=_SCAN_LIMIT)
    messages: list[dict[str, Any]] = []
    for entry in _open_task_span(entries):
        messages.extend(_entry_messages(entry, _thread_redaction))
    return messages
