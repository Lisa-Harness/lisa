"""Bounded, grounded task-scoped view assembled from recorded thread state.

The thread is the lossless record; this read path distils one task's entries
into a compact state summary — objective, status, modes, tool effects, and
evidence references — so a resumed model reads the task instead of replaying
its whole history. Every field is derived from recorded provenance, effect
receipts, and transcript stubs; nothing is inferred or summarized by a model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from sdk.thread._search import entry_task_id, redact_value
from sdk.thread.redaction import ThreadRedactionPolicy

_RESULT_REFERENCE = re.compile(r"r[a-f0-9]{64}")
_APPROVAL_SOURCES = frozenset({"host-approved", "host-approved-failed"})
_OBJECTIVE_CHARS = 400
_LATEST_CHARS = 400
_MAX_MODES = 12
_MAX_EFFECTS = 12
_MAX_TARGETS = 6
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
    provenance = entry.get("provenance") or {}
    if provenance.get("pending_approval_request_id"):
        return "awaiting_approval"
    if provenance.get("operator_exit") == "paused":
        return "paused"
    if entry.get("yielded"):
        return "awaiting_user"
    return "open"


def _blocker(entry: dict[str, Any]) -> dict[str, Any] | None:
    provenance = entry.get("provenance") or {}
    request_id = provenance.get("pending_approval_request_id")
    if request_id:
        return {"kind": "approval", "request_id": str(request_id)}
    if entry.get("yielded"):
        return {"kind": "question", "question": str(entry.get("yield_question") or "")}
    return None


def _modes(scoped: list[tuple[int, dict[str, Any]]]) -> list[str]:
    modes: list[str] = []
    for _, entry in scoped:
        for value in (
            entry.get("mode_id"),
            entry.get("transitioned_to"),
            entry.get("handoff_to"),
        ):
            name = str(value or "")
            if name and name not in modes:
                modes.append(name)
    return modes[:_MAX_MODES]


def _effects(
    scoped: list[tuple[int, dict[str, Any]]], redaction: ThreadRedactionPolicy | None
) -> list[dict[str, Any]]:
    order: list[str] = []
    counts: dict[str, int] = {}
    targets: dict[str, list[str]] = {}
    for _, entry in scoped:
        for receipt in entry.get("effect_receipts") or []:
            if not isinstance(receipt, dict):
                continue
            capability = str(receipt.get("capability_id") or "")
            if not capability:
                continue
            if capability not in counts:
                order.append(capability)
                counts[capability] = 0
                targets[capability] = []
            counts[capability] += 1
            target = _text(receipt.get("target_id"), redaction)
            if target and target not in targets[capability]:
                targets[capability].append(target)
    return [
        {
            "capability_id": capability,
            "count": counts[capability],
            "targets": targets[capability][:_MAX_TARGETS],
        }
        for capability in order[:_MAX_EFFECTS]
    ]


def _evidence(scoped: list[tuple[int, dict[str, Any]]]) -> list[str]:
    refs: list[str] = []
    for _, entry in scoped:
        for message in entry.get("transcript") or []:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or '"r' not in content:
                continue
            try:
                parsed = json.loads(content)
            except (ValueError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue
            for value in (parsed.get("ref"), parsed.get("inventory_ref")):
                if (
                    isinstance(value, str)
                    and _RESULT_REFERENCE.fullmatch(value)
                    and value not in refs
                ):
                    refs.append(value)
    return refs[-_MAX_EVIDENCE:]


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


def _trim(view: dict[str, Any], budget: int) -> dict[str, Any]:
    while len(json.dumps(view, ensure_ascii=False).encode("utf-8")) > budget:
        if view.get("evidence"):
            view["evidence"] = view["evidence"][1:]
            view["evidence_truncated"] = True
        elif len(view.get("effects") or []) > 1:
            view["effects"] = view["effects"][:-1]
            view["effects_truncated"] = True
        else:
            break
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
    if not ordered:
        return {"task_id": task_id, "status": "empty", "turns": 0}
    target = task_id or entry_task_id(ordered[-1][1])
    scoped = (
        [(turn, entry) for turn, entry in ordered if entry_task_id(entry) == target]
        if target
        else [ordered[-1]]
    )
    if not scoped:
        return {"task_id": target, "status": "empty", "turns": 0}
    objective, objective_turn = _objective(scoped, redaction)
    latest_turn, latest_entry = scoped[-1]
    latest = _text(latest_entry.get("response_text"), redaction)
    view: dict[str, Any] = {
        "task_id": target,
        "status": _status(latest_entry),
        "turns": len(scoped),
        "turn_range": [scoped[0][0], latest_turn],
        "objective": _preview(objective, _OBJECTIVE_CHARS),
        "objective_ref": f"t{objective_turn}.query",
        "modes": _modes(scoped),
        "effects": _effects(scoped, redaction),
        "evidence": _evidence(scoped),
        "latest": _preview(latest, _LATEST_CHARS),
        "latest_ref": f"t{latest_turn}.response_text",
    }
    blocker = _blocker(latest_entry)
    if blocker is not None:
        view["blocker"] = blocker
    return _trim(view, max(1024, budget_bytes))
