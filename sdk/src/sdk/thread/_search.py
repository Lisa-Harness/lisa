"""Deterministic search over permitted archive text and paired tool evidence."""

from __future__ import annotations

import heapq
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from sdk.thread.redaction import ThreadRedactionPolicy

_EXCERPT_CHARS = 240
_CONTEXT_CHARS = 80
_MAX_MATCHES = 3


@dataclass(frozen=True)
class SearchFilters:
    tool_name: str | None = None
    path: str | None = None
    task_id: str | None = None

    def accepts_tool(self, name: str, arguments: Any) -> bool:
        if self.tool_name and name != self.tool_name and name.rsplit(".", 1)[-1] != self.tool_name:
            return False
        if not self.path:
            return True
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (ValueError, TypeError):
            return False
        return any(self.path in value for value in _strings(parsed))


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def entry_task_id(entry: dict[str, Any]) -> str:
    return str(entry.get("task_id") or (entry.get("provenance") or {}).get("task_id") or "")


def _tool_calls(message: dict[str, Any]) -> Iterator[tuple[int, str, str, Any]]:
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or call
        yield (
            index,
            str(call.get("id") or ""),
            str(function.get("name") or ""),
            function.get("arguments"),
        )


def _values(
    entry: dict[str, Any], turn: int, filters: SearchFilters
) -> Iterator[tuple[str, Any, str]]:
    if not filters.tool_name and not filters.path:
        for field in ("query", "response_text", "mode_id"):
            yield f"t{turn}.{field}", entry.get(field) or "", ""
    calls: dict[str, tuple[str, Any]] = {}
    for index, message in enumerate(entry.get("transcript") or []):
        if not isinstance(message, dict):
            continue
        for call_index, call_id, name, arguments in _tool_calls(message):
            calls[call_id] = (name, arguments)
            if filters.accepts_tool(name, arguments):
                yield f"t{turn}.m{index}.c{call_index}.arguments", arguments, name
                yield f"t{turn}.m{index}.c{call_index}.name", name, name
        if message.get("role") == "tool":
            name, arguments = calls.get(str(message.get("tool_call_id") or ""), ("", None))
            name = str(message.get("name") or name)
            if filters.accepts_tool(name, arguments):
                yield f"t{turn}.m{index}.content", message.get("content") or "", name


def redact_value(value: Any, redaction: ThreadRedactionPolicy | None) -> Any:
    if isinstance(value, str):
        return redaction.redact(value) if redaction is not None else value
    if isinstance(value, list):
        return [redact_value(item, redaction) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, redaction) for key, item in value.items()}
    return value


def permitted_text(value: Any, redaction: ThreadRedactionPolicy | None) -> str:
    permitted = redact_value(value, redaction)
    return (
        permitted
        if isinstance(permitted, str)
        else json.dumps(
            permitted,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _match(text: str, query: str, ref: str, tool: str) -> dict[str, Any] | None:
    if not text:
        return None
    terms = tuple(dict.fromkeys(query.split()))
    hits = [found for term in terms if (found := re.search(re.escape(term), text, re.I))]
    if terms and not hits:
        return None
    phrase = re.search(re.escape(query), text, re.I) if query else None
    position = phrase.start() if phrase is not None else hits[0].start() if hits else 0
    offset = max(0, position - _CONTEXT_CHARS)
    return {
        "ref": ref,
        "tool_name": tool,
        "offset_chars": offset,
        "match_offset_chars": position,
        "excerpt": text[offset : offset + _EXCERPT_CHARS],
        "score": 1.0 if phrase is not None or not terms else len(hits) / len(terms),
    }


def _score_entry(
    turn: int,
    entry: dict[str, Any],
    query: str,
    filters: SearchFilters,
    redaction: ThreadRedactionPolicy | None,
) -> dict[str, Any] | None:
    if filters.task_id and entry_task_id(entry) != filters.task_id:
        return None
    matches = [
        match
        for ref, value, tool in _values(entry, turn, filters)
        if (
            match := _match(
                permitted_text(value, redaction), query, ref, str(redact_value(tool, redaction))
            )
        )
        is not None
    ]
    if not matches and (filters.tool_name or filters.path or filters.task_id):
        return None
    if not matches and not (entry.get("query") or entry.get("response_text")):
        return None
    matches.sort(key=lambda match: match["score"], reverse=True)
    return {
        "turn": turn,
        "score": matches[0]["score"] if matches else 0.0,
        "matches": matches[:_MAX_MATCHES],
    }


def search_entries(
    entries: Iterable[tuple[int, dict[str, Any]]],
    query: str,
    limit: int,
    filters: SearchFilters,
    redaction: ThreadRedactionPolicy | None,
) -> list[dict[str, Any]]:
    candidates = (
        scored
        for turn, entry in entries
        if (scored := _score_entry(turn, entry, query, filters, redaction)) is not None
    )
    return heapq.nlargest(limit, candidates, key=lambda match: (match["score"], match["turn"]))
