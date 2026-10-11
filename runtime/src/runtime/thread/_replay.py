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
_APPROVAL_SOURCES = frozenset({"host-approved", "host-approved-failed"})


def _is_task_link(entry: dict[str, Any]) -> bool:
    provenance = entry.get("provenance") or {}
    return bool(
        provenance.get("operator_exit") == "paused"
        or provenance.get("pending_approval_request_id")
        or provenance.get("source") in _APPROVAL_SOURCES
    )


def _open_task_span(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the newest entry plus the unfinished-task entries directly before it."""
    relevant = [
        e
        for e in entries
        if (e.get("provenance") or {}).get("source") in _APPROVAL_SOURCES
        or not is_host_command_query(str(e.get("query") or ""))
    ]
    latest_by_turn = {
        turn_id: entry
        for entry in relevant
        if (turn_id := (entry.get("provenance") or {}).get("kernel_turn_id"))
    }
    relevant = [
        entry
        for entry in relevant
        if not (turn_id := (entry.get("provenance") or {}).get("kernel_turn_id"))
        or latest_by_turn[turn_id] is entry
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
    if isinstance(replayed.get("content"), str) and replayed.get("role") != "tool":
        replayed["content"] = _redact(replayed["content"], redaction)
    if replayed.get("tool_call_id"):
        replayed["tool_call_id"] = f"{prefix}{replayed['tool_call_id']}"
    for call in replayed.get("tool_calls") or []:
        call["id"] = f"{prefix}{call.get('id', '')}"
        payload = _call_payload(call)
        if isinstance(payload.get("arguments"), str):
            payload["arguments"] = _redact(payload["arguments"], redaction)
    return replayed


def _call_payload(call: dict[str, Any]) -> dict[str, Any]:
    """Return the dict holding ``name``/``arguments`` for nested or flat tool calls."""
    function = call.get("function")
    return function if isinstance(function, dict) else call


def _entry_messages(
    entry: dict[str, Any],
    redaction: Any,
    *,
    approved_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    # Tool-call ids are only unique within one turn; prefix them per turn.
    prefix = f"t{entry.get('turn', 'x')}_"
    messages = [{"role": "user", "content": _redact(str(entry.get("query") or ""), redaction)}]
    transcript = list(entry.get("transcript") or [])
    # The stored user message carries this turn's briefings; replay the plain query instead.
    if transcript and transcript[0].get("role") == "user":
        transcript = transcript[1:]
    messages.extend(_replay_message(m, prefix, redaction) for m in transcript)
    if approved_result is not None:
        messages.append(_replay_message(approved_result, prefix, redaction))
        return messages
    response = str(entry.get("response_text") or "")
    last_message = transcript[-1] if transcript else None
    if response and not (
        last_message is not None
        and last_message.get("role") == "assistant"
        and last_message.get("content") == response
    ):
        messages.append({"role": "assistant", "content": _redact(response, redaction)})
    return messages


def _paired_approval_result(
    paused: dict[str, Any], approvals: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    provenance = paused.get("provenance") or {}
    request_id = str(provenance.get("pending_approval_request_id") or "")
    call_id = str(provenance.get("pending_approval_tool_call_id") or "")
    approved = approvals.get(request_id) if request_id and call_id else None
    if approved is None:
        return None
    calls = [
        call
        for message in paused.get("transcript") or []
        for call in message.get("tool_calls") or []
        if call.get("id") == call_id
    ]
    if len(calls) != 1:
        return None
    tool_name = str(_call_payload(calls[0]).get("name") or "")
    if tool_name != (approved.get("provenance") or {}).get("target_id"):
        return None
    failed = (approved.get("provenance") or {}).get("source") == "host-approved-failed"
    if not failed and not any(
        receipt.get("capability_id") == tool_name
        for receipt in approved.get("effect_receipts") or []
    ):
        return None
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": str(approved.get("response_text") or ""),
    }


def build_open_task_messages(
    *, thread_path: Path | None, budget_bytes: int | None = None
) -> list[dict[str, Any]]:
    """Return the latest turn and its unfinished task as replayable messages.

    ``budget_bytes`` bounds the resident tool-result bytes for the whole
    replayed working set (``None`` keeps the projection default): oldest
    results collapse into the evidence inventory by address, newest stay
    verbatim. Residency by allocation at the turn boundary (R2).
    """
    from runtime.thread import (
        _thread_redaction,
        load_recent_thread_entries,
        project_thread_messages,
    )

    entries = _open_task_span(
        load_recent_thread_entries(thread_path=thread_path, limit=_SCAN_LIMIT)
    )
    approvals = {
        str(provenance["approval_request_id"]): entry
        for entry in entries
        if (provenance := entry.get("provenance") or {}).get("source") in _APPROVAL_SOURCES
        and provenance.get("approval_request_id")
    }
    messages: list[dict[str, Any]] = []
    paired_ids: set[str] = set()
    for entry in entries:
        provenance = entry.get("provenance") or {}
        if provenance.get("source") in _APPROVAL_SOURCES:
            if provenance.get("approval_request_id") not in paired_ids:
                replay = _entry_messages(entry, _thread_redaction)
                messages.extend(
                    project_thread_messages(
                        replay,
                        str(provenance.get("task_id") or ""),
                        thread_path=thread_path,
                        budget_bytes=_entry_tool_bytes(replay),
                    )
                )
            continue
        approved_result = _paired_approval_result(entry, approvals)
        if approved_result is not None:
            paired_ids.add(str(provenance["pending_approval_request_id"]))
        replay = _entry_messages(entry, _thread_redaction, approved_result=approved_result)
        messages.extend(
            project_thread_messages(
                replay,
                str(provenance.get("task_id") or ""),
                thread_path=thread_path,
                budget_bytes=_entry_tool_bytes(replay),
            )
        )
    if budget_bytes is not None:
        # Turn-boundary working-set recompute (R2): the per-entry projection
        # above bounds each result on its own; this final pass bounds the
        # whole replayed set, collapsing the oldest tool groups into the
        # evidence inventory by address so the newest stays resident.
        messages = project_thread_messages(
            messages,
            str((entries[-1].get("provenance") or {}).get("task_id") or "") if entries else "",
            thread_path=thread_path,
            budget_bytes=budget_bytes,
        )
    return messages


def _entry_tool_bytes(messages: list[dict[str, Any]]) -> int:
    """Replay admits complete payloads; the kernel budgets the assembled request."""
    return sum(
        len(str(message.get("content") or "").encode("utf-8"))
        for message in messages
        if message.get("role") == "tool"
    )
