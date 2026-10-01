"""Grounded, byte-bounded task state derived from lossless thread entries."""

from __future__ import annotations

import json
from typing import Any, cast

from sdk.thread._search import entry_task_id, redact_value
from sdk.thread.redaction import ThreadRedactionPolicy

_APPROVAL_SOURCES = frozenset({"host-approved", "host-approved-failed"})
_OBJECTIVE_CHARS = 400
_LATEST_CHARS = 400
_MAX_MODES = 12
_MAX_EFFECTS = 12
_MAX_ARTIFACTS = 6
_MAX_EVIDENCE = 16
DEFAULT_TASK_VIEW_BYTES = 8192


def _text(value: Any, redaction: ThreadRedactionPolicy | None) -> str:
    permitted = redact_value(str(value or ""), redaction)
    return permitted if isinstance(permitted, str) else str(permitted)


def _preview(text: str, limit: int) -> dict[str, Any]:
    field: dict[str, Any] = {"text": text[:limit], "total_chars": len(text)}
    if len(text) > limit:
        field["abbreviated"] = True
    return field


def _status(entry: dict[str, Any]) -> str:
    provenance: dict[str, Any] = entry.get("provenance") or {}
    if provenance.get("pending_approval_request_id"):
        return "awaiting_approval"
    if provenance.get("operator_exit") == "paused":
        return "paused"
    if entry.get("yielded"):
        return "awaiting_user"
    return "open"


def _blocker(
    entry: dict[str, Any], redaction: ThreadRedactionPolicy | None
) -> dict[str, Any] | None:
    provenance: dict[str, Any] = entry.get("provenance") or {}
    request_id = provenance.get("pending_approval_request_id")
    if request_id:
        return {"kind": "approval", "request_id": _text(request_id, redaction)}
    if entry.get("yielded"):
        return {"kind": "question", "question": _text(entry.get("yield_question"), redaction)}
    return None


def _modes(
    scoped: list[tuple[int, dict[str, Any]]], redaction: ThreadRedactionPolicy | None
) -> list[str]:
    modes: list[str] = []
    for _, entry in scoped:
        for value in (
            entry.get("mode_id"),
            entry.get("transitioned_to"),
            entry.get("handoff_to"),
        ):
            name = _text(value, redaction)
            if name and name not in modes:
                modes.append(name)
    return modes[:_MAX_MODES]


def _effects(
    scoped: list[tuple[int, dict[str, Any]]], redaction: ThreadRedactionPolicy | None
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    artifacts: dict[tuple[str, str], list[str]] = {}
    for _, entry in scoped:
        receipts: list[Any] = entry.get("effect_receipts") or []
        for raw_receipt in receipts:
            if not isinstance(raw_receipt, dict):
                continue
            receipt = cast(dict[str, Any], raw_receipt)
            capability = str(receipt.get("capability_id") or "")
            if not capability:
                continue
            key = (capability, str(receipt.get("effect") or ""))
            counts[key] = counts.get(key, 0) + 1
            artifact_ids = artifacts.setdefault(key, [])
            artifact = _text(receipt.get("artifact_id"), redaction)
            if artifact and artifact not in artifact_ids:
                artifact_ids.append(artifact)
    return [
        {
            "capability_id": _text(capability, redaction),
            "effect": _text(effect, redaction),
            "count": count,
            "artifact_ids": artifacts[(capability, effect)][:_MAX_ARTIFACTS],
        }
        for (capability, effect), count in list(counts.items())[:_MAX_EFFECTS]
    ]


def _evidence(scoped: list[tuple[int, dict[str, Any]]]) -> list[str]:
    refs: list[str] = []
    for turn, entry in scoped:
        messages: list[Any] = entry.get("transcript") or []
        for index, raw_message in enumerate(messages):
            if not isinstance(raw_message, dict):
                continue
            message = cast(dict[str, Any], raw_message)
            if message.get("role") == "tool" and "content" in message:
                refs.append(f"t{turn}.m{index}.content")
        provenance: dict[str, Any] = entry.get("provenance") or {}
        if provenance.get("source") in _APPROVAL_SOURCES:
            refs.append(f"t{turn}.response_text")
    return refs


def _objective(
    scoped: list[tuple[int, dict[str, Any]]], redaction: ThreadRedactionPolicy | None
) -> tuple[str, int]:
    """Return the task's anchor request and its turn (skips host commands)."""
    for turn, entry in scoped:
        query = _text(entry.get("query"), redaction)
        if query and not query.lstrip().startswith("/"):
            return query, turn
    turn, entry = scoped[0]
    return _text(entry.get("query"), redaction), turn


def _compact_text(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return [_compact_text(item, limit) for item in cast(list[Any], value)]
    if isinstance(value, dict):
        record = cast(dict[str, Any], value)
        compacted: dict[str, Any] = {}
        for key, item in record.items():
            if key.endswith("_ref") or key in ("status", "kind"):
                compacted[key] = item
            elif key in ("task_id", "request_id") and isinstance(item, str) and len(item) > limit:
                compacted[key] = _preview(item, limit)
            else:
                compacted[key] = _compact_text(item, limit)
        original_text = record.get("text")
        if isinstance(original_text, str) and len(original_text) > limit:
            compacted["abbreviated"] = True
        return compacted
    return value


def _trim(view: dict[str, Any], budget: int) -> dict[str, Any]:
    text_limit = _OBJECTIVE_CHARS
    while len(json.dumps(view, ensure_ascii=False).encode("utf-8")) > budget:
        if view.get("evidence"):
            view["evidence"] = view["evidence"][1:]
            view["evidence_truncated"] = True
        elif view.get("effects"):
            view["effects"] = view["effects"][:-1]
            view["effects_truncated"] = True
        elif view.get("modes"):
            view["modes"] = view["modes"][:-1]
            view["modes_truncated"] = True
        else:
            text_limit //= 2
            view = _compact_text(view, text_limit)
            view["truncated"] = True
    return view


def build_task_view(
    entries: list[tuple[int, dict[str, Any]]],
    task_id: str = "",
    redaction: ThreadRedactionPolicy | None = None,
    *,
    budget_bytes: int = DEFAULT_TASK_VIEW_BYTES,
) -> dict[str, Any]:
    """Assemble one task's recorded state into a bounded, reference-carrying view."""
    ordered = list(entries)
    budget = max(1024, budget_bytes)
    if not ordered:
        return _trim({"task_id": _text(task_id, redaction), "status": "empty", "turns": 0}, budget)
    target = task_id
    if not target:
        for _, entry in reversed(ordered):
            target = entry_task_id(entry)
            if target:
                break
    scoped = (
        [(turn, entry) for turn, entry in ordered if entry_task_id(entry) == target]
        if target
        else [ordered[-1]]
    )
    if not scoped:
        return _trim(
            {
                "task_id": _text(target, redaction),
                "status": "empty",
                "turns": 0,
                "history_may_be_incomplete": ordered[0][0] > 0,
            },
            budget,
        )
    objective, objective_turn = _objective(scoped, redaction)
    latest_turn, latest_entry = scoped[-1]
    latest = _text(latest_entry.get("response_text"), redaction)
    evidence = _evidence(scoped)
    view: dict[str, Any] = {
        "task_id": _text(target, redaction),
        "task_id_ref": f"t{latest_turn}.task_id",
        "status": _status(latest_entry),
        "turns": len(scoped),
        "turn_range": [scoped[0][0], latest_turn],
        "history_may_be_incomplete": ordered[0][0] > 0,
        "objective": _preview(objective, _OBJECTIVE_CHARS),
        "objective_ref": f"t{objective_turn}.query",
        "modes": _modes(scoped, redaction),
        "effects": _effects(scoped, redaction),
        "evidence": evidence[-_MAX_EVIDENCE:],
        "evidence_truncated": len(evidence) > _MAX_EVIDENCE,
        "latest": _preview(latest, _LATEST_CHARS),
        "latest_ref": f"t{latest_turn}.response_text",
    }
    blocker = _blocker(latest_entry, redaction)
    if blocker is not None:
        if blocker["kind"] == "approval":
            blocker["request_id_ref"] = f"t{latest_turn}.pending_approval_request_id"
        else:
            blocker["question_ref"] = f"t{latest_turn}.yield_question"
        view["blocker"] = blocker
    return _trim(view, budget)
