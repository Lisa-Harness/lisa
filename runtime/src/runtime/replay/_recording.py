"""Load one recorded turn from the thread store into a replayable form.

The loader is the read-only half of the replay engine: it dereferences the
recorded turn's ``turn_input.calls`` into :class:`ArchivedCall` pairs via the
slice-2 archive loader, and walks the recorded transcript's ``role="tool"``
messages into a ``tool_call_id -> output text`` map. Nothing here touches a
runtime; the engine (``_engine.py``) consumes the result.

Archive gaps (a call with no ``response_ref``, or an unloadable ref) raise
:class:`ReplayError` naming the turn: turns recorded before the model-call
archive existed have no responses and are refused, per the spec's default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from runtime.testing.archive import ArchivedCall, archive_loader


def _unwrap_envelope(snapshot: Any) -> dict[str, Any]:
    """Unwrap a persisted result-object envelope into its snapshot dict.

    Production persists model request/response snapshots as
    ``{"content": json.dumps(snapshot)}`` (see ``runtime.turn_continuation.
    persist_request``); ``load_result_object`` returns that envelope as-is.
    The archive consumes the snapshot dict itself, so unwrap here. Objects
    persisted without a ``content`` key are returned wrapped by
    ``load_result_object`` already and pass through unchanged.
    """
    if not isinstance(snapshot, dict):
        return {}
    content = snapshot.get("content")
    if not isinstance(content, str):
        return snapshot
    try:
        unwrapped = json.loads(content)
    except (TypeError, ValueError):
        return snapshot
    return unwrapped if isinstance(unwrapped, dict) else snapshot


def _text(content: Any) -> str:
    """Flatten message content (string or OpenAI-style parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else str(content)


@dataclass
class RecordedTurn:
    """One recorded turn, loaded and dereferenced for replay.

    ``calls`` holds the archived request/response pairs in recorded order
    (the archive gateway consumes them in that order). ``tool_results`` maps
    each recorded ``tool_call_id`` to the transcript's tool-output text, so a
    replayed dispatch can be answered from the recording. ``effect_receipts``
    is the recorded receipt list the report compares against.
    """

    turn_index: int
    turn_id: str
    mode_id: str
    query: str
    response_text: str
    calls: list[ArchivedCall] = field(default_factory=list)
    tool_results: dict[str, str] = field(default_factory=dict)
    effect_receipts: list[dict[str, Any]] = field(default_factory=list)
    run_ref: str = ""


def load_recorded_turn(store: Any, turn_index: int) -> RecordedTurn:
    """Load the recorded turn at ``turn_index`` from ``store``.

    ``store`` is a :class:`sdk.thread.ThreadStore` opened on the source
    thread's base path. Raises :class:`ReplayError` when the turn is missing
    or its archive has a gap (no ``response_ref``, or an unloadable ref).
    """
    from runtime.replay import ReplayError  # local: avoid a module-level cycle

    entries = store.load_turns({turn_index})
    entry = entries.get(turn_index)
    if entry is None:
        raise ReplayError(f"no recorded turn at index {turn_index}")

    turn_input = entry.get("turn_input") or {}
    if not isinstance(turn_input, dict):
        turn_input = {}
    try:
        calls = ArchivedCall.from_turn_input(
            turn_input.get("calls", []),
            load=lambda ref: _unwrap_envelope(archive_loader(store.base_path)(ref)),
        )
    except Exception as exc:
        raise ReplayError(f"turn {turn_index} cannot be replayed: archive gap: {exc}") from exc

    tool_results: dict[str, str] = {}
    for message in entry.get("transcript") or []:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id", ""))
        if tool_call_id:
            tool_results[tool_call_id] = _text(message.get("content"))

    return RecordedTurn(
        turn_index=turn_index,
        turn_id=str(entry.get("turn_id", "") or ""),
        mode_id=str(entry.get("mode_id", "") or ""),
        query=str(entry.get("query", "") or ""),
        response_text=str(entry.get("response_text", "") or ""),
        calls=calls,
        tool_results=tool_results,
        effect_receipts=list(entry.get("effect_receipts") or []),
        run_ref=str(entry.get("run_ref", "") or ""),
    )
