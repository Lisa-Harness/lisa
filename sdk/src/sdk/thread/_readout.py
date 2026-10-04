"""Bounded, restorable model-facing views of the lossless thread archive."""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from sdk.thread._backend import ThreadStore
from sdk.thread._results import load_result_object
from sdk.thread._search import SearchFilters, entry_task_id, redact_value, search_entries
from sdk.thread._taskview import DEFAULT_TASK_VIEW_BYTES, RECORDED_HISTORY_LABEL, build_task_view
from sdk.thread.redaction import ThreadRedactionPolicy

DEFAULT_READOUT_BYTES = 16384
MAX_READOUT_BYTES = 32768
_TASK_SCAN_LIMIT = 200
_INLINE_BYTES = 600
_HEAD_CHARS = 120
_TAIL_CHARS = 60
_CURSOR_RESERVE_BYTES = 256
_REASONING_CHARS = 800
_MAX_SEARCH_RESULTS = 20
_MAX_RECENT_ENTRIES = 50
_MAX_REFERENCE_CHARS = 128
_RANGE_CURSOR = re.compile(r"(range|recent)\.(\d{1,20})\.(\d{1,20})\.(\d{1,20})\.(-?\d{1,20})")
_SEARCH_CURSOR = re.compile(r"search\.(-?\d{1,20})\.(\d{1,20})\.([a-f0-9]{16})")
_REFERENCE = re.compile(r"t(\d{1,20})(?:\.m(\d{1,20}))?(?:\.c(\d{1,20}))?(?:\.([a-z_]+))?")
_ENTRY_FIELDS = ("mode_id", "query", "response_text", "handoff_to", "transitioned_to")
_MESSAGE_FIELDS = ("role", "name", "tool_call_id", "id", "content", "tool_calls", "is_error")


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _byte_size(value: Any) -> int:
    return len(_encode(value).encode("utf-8"))


def _budget(requested: int) -> int:
    if requested <= 0:
        raise ValueError("budget_bytes must be positive")
    return max(DEFAULT_READOUT_BYTES, min(requested, MAX_READOUT_BYTES))


def _page() -> dict[str, Any]:
    return {
        "scope": "recorded_history",
        "interpretation": RECORDED_HISTORY_LABEL,
        "entries": [],
        "next_cursor": None,
        "has_more": False,
    }


def _finish(page: dict[str, Any], cursor: str | None = None) -> str:
    page["next_cursor"] = cursor
    page["has_more"] = cursor is not None
    return _encode(page)


def _fits(page: dict[str, Any], budget: int) -> bool:
    return _byte_size(page) + _CURSOR_RESERVE_BYTES <= budget


@dataclass
class PageCache:
    """Bounded LRU for pages of immutable thread refs.

    ``t<turn>`` refs are append-only history and ``r<hash>`` refs are
    content-addressed by payload digest, so a page fetched for
    ``(ref, offset, limit)`` can never change while the process lives.
    The cache is read-path only; the lossless archive is never mutated.
    """

    def __init__(self, max_pages: int = 256, max_bytes: int = 524_288) -> None:
        self._max_pages = max(1, max_pages)
        self._max_bytes = max(1, max_bytes)
        self._entries: OrderedDict[tuple[str, int, int, str], dict[str, Any]] = OrderedDict()
        self._bytes = 0

    def get(self, key: tuple[str, int, int, str]) -> dict[str, Any] | None:
        page = self._entries.get(key)
        if page is None:
            return None
        self._entries.move_to_end(key)
        return page

    def put(self, key: tuple[str, int, int, str], page: dict[str, Any]) -> None:
        size = _byte_size(page)
        if size > self._max_bytes:
            return
        while self._entries and (
            len(self._entries) >= self._max_pages or self._bytes + size > self._max_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= _byte_size(evicted)
        self._entries[key] = page
        self._bytes += size
        self._entries.move_to_end(key)


@dataclass(frozen=True)
class ThreadReader:
    """Project archive content without mutating it or exposing raw reasoning fields."""

    store: ThreadStore
    redaction: ThreadRedactionPolicy | None = None
    page_cache: PageCache | None = None

    def _redact(self, value: Any) -> Any:
        return redact_value(value, self.redaction)

    def _preview(self, value: Any, ref: str) -> Any:
        permitted = self._redact(value)
        if _byte_size(permitted) <= _INLINE_BYTES:
            return permitted
        text = permitted if isinstance(permitted, str) else _encode(permitted)
        head_chars = min(_HEAD_CHARS, len(text))
        tail_chars = min(_TAIL_CHARS, len(text) - head_chars)
        return {
            "ref": ref,
            "total_chars": len(text),
            "abbreviated": True,
            "head": text[:head_chars],
            "tail": text[-tail_chars:] if tail_chars else "",
            "omitted_chars": len(text) - head_chars - tail_chars,
        }

    def _entry(self, entry: dict[str, Any], turn: int) -> dict[str, Any]:
        result: dict[str, Any] = {"turn": turn, "transcript_count": len(entry.get("transcript") or [])}
        result["timestamp_utc"] = self._preview(
            entry.get("timestamp_utc"), f"t{turn}.timestamp_utc"
        )
        result["provenance_source"] = self._preview(
            (entry.get("provenance") or {}).get("source"), f"t{turn}.provenance_source"
        )
        for field in _ENTRY_FIELDS:
            result[field] = self._preview(entry.get(field), f"t{turn}.{field}")
        caps = self._receipt_caps(entry)
        result["task_id"] = self._preview(entry_task_id(entry), f"t{turn}.task_id")
        result["receipt_capability_ids"] = self._preview(caps, f"t{turn}.receipt_capability_ids")
        source = entry.get("content_source")
        reasoning = entry.get("reasoning_text")
        if source and reasoning:
            result["content_source"] = self._preview(source, f"t{turn}.content_source")
            reasoning_text = str(self._redact(str(reasoning)))
            result["reasoning_excerpt"] = reasoning_text[-_REASONING_CHARS:]
            result["reasoning_omitted_chars"] = max(0, len(reasoning_text) - _REASONING_CHARS)
        return result

    @staticmethod
    def _receipt_caps(entry: dict[str, Any]) -> list[str]:
        return [
            str(receipt.get("capability_id") or "")
            for receipt in entry.get("effect_receipts") or []
            if isinstance(receipt, dict)
        ]

    def _message(self, message: Any, turn: int, index: int) -> dict[str, Any]:
        ref = f"t{turn}.m{index}"
        result: dict[str, Any] = {"ref": ref}
        if not isinstance(message, dict):
            result["content"] = self._preview(message, ref)
            return result
        for field in _MESSAGE_FIELDS:
            if field in message:
                result[field] = self._preview(message[field], f"{ref}.{field}")
        return result

    def read_range(
        self,
        start_turn: int,
        end_turn: int,
        *,
        include_transcript: bool = False,
        cursor: str | None = None,
        budget_bytes: int = DEFAULT_READOUT_BYTES,
    ) -> str:
        """Read a fixed archive window; cursors can continue inside a transcript."""
        if start_turn < 0 or end_turn < start_turn:
            raise ValueError("Require 0 <= start_turn <= end_turn")
        bounds: tuple[int, int] = (start_turn, min(end_turn, self.store.next_turn_index() - 1))
        if cursor is not None:
            parsed = self._range_cursor(cursor, "range")
            if parsed[0] != start_turn or parsed[1] > end_turn:
                raise ValueError("Cursor does not belong to this range")
            bounds = parsed[:2]
        return self._read_window(bounds, "range", include_transcript, cursor, _budget(budget_bytes))

    def read_recent(
        self,
        limit: int = 10,
        *,
        include_transcript: bool = False,
        cursor: str | None = None,
        budget_bytes: int = DEFAULT_READOUT_BYTES,
    ) -> str:
        """Read a recent window anchored at the first page's archive tail."""
        capped_limit = max(1, min(limit, _MAX_RECENT_ENTRIES))
        end = self.store.next_turn_index() - 1
        bounds = (max(0, end - capped_limit + 1), end)
        if cursor is not None:
            bounds = self._range_cursor(cursor, "recent")[:2]
        return self._read_window(
            bounds, "recent", include_transcript, cursor, _budget(budget_bytes)
        )

    @staticmethod
    def _range_cursor(cursor: str, kind: str) -> tuple[int, int, int, int]:
        cursor_match = _RANGE_CURSOR.fullmatch(cursor)
        if cursor_match is None or cursor_match[1] != kind:
            raise ValueError("Invalid thread cursor")
        start, end, turn, position = (int(value) for value in cursor_match.groups()[1:])
        if not start <= turn <= end or position < -1:
            raise ValueError("Invalid thread cursor position")
        return start, end, turn, position

    def _read_window(
        self,
        bounds: tuple[int, int],
        kind: str,
        include_transcript: bool,
        cursor: str | None,
        budget: int,
    ) -> str:
        start, end = bounds
        turn, position = start, -1
        if cursor is not None:
            _, _, turn, position = self._range_cursor(cursor, kind)
        if position >= 0 and not include_transcript:
            raise ValueError("Transcript cursor requires include_transcript=true")
        page = _page()
        for ordinal, entry in self.store.iter_range(start_turn=turn, end_turn=end):
            projected: dict[str, Any] = (
                self._entry(entry, ordinal)
                if position == -1
                else {"turn": ordinal, "transcript_start": position}
            )
            if include_transcript:
                projected["transcript_start"] = max(0, position)
                projected["transcript_count"] = len(entry.get("transcript") or [])
                transcript: list[Any] = []
                projected["transcript"] = transcript
            page["entries"].append(projected)
            if not _fits(page, budget):
                page["entries"].pop()
                return _finish(page, f"{kind}.{start}.{end}.{ordinal}.{position}")
            if include_transcript:
                next_position = self._append_messages(page, entry, ordinal, position, budget)
                if next_position is not None:
                    return _finish(page, f"{kind}.{start}.{end}.{ordinal}.{next_position}")
            position = -1
        return _finish(page)

    def _append_messages(
        self,
        page: dict[str, Any],
        entry: dict[str, Any],
        turn: int,
        position: int,
        budget: int,
    ) -> int | None:
        transcript = entry.get("transcript") or []
        if position > len(transcript):
            raise ValueError("Cursor exceeds transcript length")
        messages = page["entries"][-1]["transcript"]
        for index in range(max(0, position), len(transcript)):
            messages.append(self._message(transcript[index], turn, index))
            if not _fits(page, budget):
                messages.pop()
                return index
        return None

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        cursor: str | None = None,
        budget_bytes: int = DEFAULT_READOUT_BYTES,
        tool_name: str | None = None,
        path: str | None = None,
        task_id: str | None = None,
        start_turn: int = 0,
        end_turn: int | None = None,
    ) -> str:
        """Return bounded ranked matches with expandable query and response previews."""
        budget = _budget(budget_bytes)
        capped_limit = max(1, min(limit, _MAX_SEARCH_RESULTS))
        if start_turn < 0 or (end_turn is not None and end_turn < start_turn):
            raise ValueError("Invalid search turn range")
        inputs = [query, capped_limit, tool_name, path, task_id, start_turn, end_turn]
        signature = hashlib.sha256(_encode(inputs).encode("utf-8")).hexdigest()[:16]
        end = self.store.next_turn_index() - 1
        if end_turn is not None:
            end = min(end, end_turn)
        position = 0
        if cursor is not None:
            cursor_match = _SEARCH_CURSOR.fullmatch(cursor)
            if cursor_match is None or cursor_match[3] != signature:
                raise ValueError("Cursor does not belong to this search")
            end, position = int(cursor_match[1]), int(cursor_match[2])
        matches = search_entries(
            self.store.iter_range(start_turn=start_turn, end_turn=end),
            query,
            capped_limit,
            SearchFilters(tool_name, path, task_id),
            self.redaction,
        )
        if position > len(matches):
            raise ValueError("Cursor exceeds search results")
        page = _page()
        for index in range(position, len(matches)):
            candidate = matches[index]
            turn = candidate["turn"]
            raw = self.store.load_range(start_turn=turn, end_turn=turn)[0]
            projected = self._entry(raw, turn)
            projected["response_preview"] = projected.pop("response_text")
            projected["score"] = candidate["score"]
            projected["matches"] = candidate["matches"]
            page["entries"].append(projected)
            if not _fits(page, budget):
                page["entries"].pop()
                return _finish(page, f"search.{end}.{index}.{signature}")
        return _finish(page)

    def task_view(
        self,
        task_id: str = "",
        *,
        scan_limit: int = _TASK_SCAN_LIMIT,
        budget_bytes: int = DEFAULT_TASK_VIEW_BYTES,
    ) -> str:
        """Summarize one task's recorded state from a bounded recent scan."""
        if budget_bytes <= 0 or scan_limit <= 0:
            raise ValueError("budget_bytes and scan_limit must be positive")
        budget = max(1024, min(budget_bytes, MAX_READOUT_BYTES))
        end = self.store.next_turn_index() - 1
        if end < 0:
            return _encode(build_task_view([], task_id, self.redaction, budget_bytes=budget))
        start = max(0, end - max(1, min(scan_limit, _TASK_SCAN_LIMIT)) + 1)
        entries = list(self.store.iter_range(start_turn=start, end_turn=end))
        return _encode(build_task_view(entries, task_id, self.redaction, budget_bytes=budget))

    def read_content(
        self,
        ref: str,
        offset_chars: int = 0,
        limit_chars: int = 4000,
        *,
        budget_bytes: int = DEFAULT_READOUT_BYTES,
    ) -> str:
        """Page a complete permitted value; offsets count Unicode characters after redaction."""
        if len(ref) > _MAX_REFERENCE_CHARS or offset_chars < 0 or limit_chars <= 0:
            raise ValueError("Invalid reference, offset_chars, or limit_chars")
        key = (ref, offset_chars, limit_chars, "read")
        cached = self.page_cache.get(key) if self.page_cache is not None else None
        if cached is not None:
            cached = dict(cached)
            cached["cached"] = True
            return _encode(cached)
        value, source = self._resolve(ref)
        permitted = self._redact(value)
        text = permitted if isinstance(permitted, str) else _encode(permitted)
        if offset_chars > len(text):
            raise ValueError("offset_chars exceeds content length")
        budget = _budget(budget_bytes)
        lower, upper = 0, min(limit_chars, MAX_READOUT_BYTES, len(text) - offset_chars)
        while lower < upper:
            length = (lower + upper + 1) // 2
            candidate = self._content_page(ref, text, offset_chars, length, source)
            if _byte_size(candidate) <= budget:
                lower = length
            else:
                upper = length - 1
        page = self._content_page(ref, text, offset_chars, lower, source)
        if self.page_cache is not None:
            self.page_cache.put(key, page)
        return _encode(page)

    def find_content(
        self,
        ref: str,
        query: str,
        offset_chars: int = 0,
        limit: int = 10,
        *,
        budget_bytes: int = DEFAULT_READOUT_BYTES,
    ) -> str:
        """Find literal text inside one permitted value without paging through all of it."""
        if not query or offset_chars < 0 or len(ref) > _MAX_REFERENCE_CHARS:
            raise ValueError("Require a nonempty query, valid ref, and nonnegative offset")
        key = (ref, offset_chars, limit, "find:" + query)
        cached = self.page_cache.get(key) if self.page_cache is not None else None
        if cached is not None:
            cached = dict(cached)
            cached["cached"] = True
            return _encode(cached)
        raw, source = self._resolve(ref)
        value = self._redact(raw)
        text = value if isinstance(value, str) else _encode(value)
        if offset_chars > len(text):
            raise ValueError("offset_chars exceeds content length")
        budget = _budget(budget_bytes)
        pattern = re.compile(re.escape(query), re.I)
        page: dict[str, Any] = {
            "ref": ref,
            "source": source,
            "matches": [],
            "next_offset_chars": None,
            "has_more": False,
            "total_chars": len(text),
        }
        position = offset_chars
        while (found := pattern.search(text, position)) is not None:
            offset = max(0, found.start() - 80)
            item = {
                "offset_chars": offset,
                "match_offset_chars": found.start(),
                "excerpt": text[offset : offset + 240],
            }
            page["matches"].append(item)
            if _byte_size(page) + _CURSOR_RESERVE_BYTES > budget:
                page["matches"].pop()
                page["next_offset_chars"] = found.start()
                page["has_more"] = True
                break
            position = found.end()
            if len(page["matches"]) >= max(1, min(limit, _MAX_SEARCH_RESULTS)):
                if pattern.search(text, position) is not None:
                    page["next_offset_chars"] = position
                    page["has_more"] = True
                break
        if self.page_cache is not None:
            self.page_cache.put(key, page)
        return _encode(page)

    @staticmethod
    def _content_page(
        ref: str, text: str, offset: int, length: int, source: dict[str, Any]
    ) -> dict[str, Any]:
        next_offset = offset + length
        has_more = next_offset < len(text)
        return {
            "ref": ref,
            "source": source,
            "content": text[offset:next_offset],
            "offset_chars": offset,
            "total_chars": len(text),
            "next_offset_chars": next_offset if has_more else None,
            "has_more": has_more,
        }

    def _resolve(self, ref: str) -> tuple[Any, dict[str, Any]]:
        if ref.startswith("r"):
            return load_result_object(self.store.base_path, ref)["content"], {
                "scope": "stored_tool_result",
                "source_kind": "tool",
                "timestamp_utc": None,
                "interpretation": (
                    "Stored tool output; its time and relationship "
                    "to the active request are unknown."
                ),
            }
        match = _REFERENCE.fullmatch(ref)
        if match is None:
            raise ValueError("Invalid thread reference")
        turn = int(match[1])
        entries = self.store.load_range(start_turn=turn, end_turn=turn)
        if not entries:
            raise ValueError("Thread reference not found")
        entry = entries[0]
        field = match[4]
        provenance = (entry.get("provenance") or {}).get("source")
        source = {
            "scope": "recorded_history",
            "interpretation": RECORDED_HISTORY_LABEL,
            "turn": turn,
            "timestamp_utc": self._preview(entry.get("timestamp_utc"), f"t{turn}.timestamp_utc"),
            "provenance_source": self._preview(provenance, f"t{turn}.provenance_source"),
            "source_kind": "entry_metadata",
        }
        if match[3] is not None:
            if match[2] is None:
                raise ValueError("Call reference requires a message index")
            value = self._resolve_call(entry, int(match[2]), int(match[3]), field)
            source["source_kind"] = "assistant_tool_call"
            return value, source
        if match[2] is not None:
            index = int(match[2])
            value = self._resolve_message(entry, index, field)
            message = entry["transcript"][index]
            role = message.get("role") if isinstance(message, dict) else None
            source["source_kind"] = (
                role if role in ("user", "assistant", "tool", "system") else "message"
            )
            return value, source
        value = self._resolve_entry_field(entry, field)
        if field == "query":
            source["source_kind"] = "user"
        elif field == "response_text":
            source["source_kind"] = (
                "host" if str(provenance or "").startswith("host-") else "assistant"
            )
        return value, source

    def _resolve_entry_field(self, entry: dict[str, Any], field: str | None) -> Any:
        if field in ("timestamp_utc", "provenance_source"):
            return (
                entry.get(field)
                if field == "timestamp_utc"
                else (entry.get("provenance") or {}).get("source")
            )
        if field == "receipt_capability_ids":
            return self._receipt_caps(entry)
        if field == "task_id":
            return entry_task_id(entry)
        if field == "pending_approval_request_id":
            return (entry.get("provenance") or {}).get(field) or ""
        if field == "yield_question":
            return entry.get(field) or ""
        if field not in (*_ENTRY_FIELDS, "content_source") or field not in entry:
            raise ValueError("Entry field not found or not available for expansion")
        return entry[field]

    @staticmethod
    def _resolve_call(
        entry: dict[str, Any],
        message_index: int,
        call_index: int,
        field: str | None,
    ) -> Any:
        transcript = entry.get("transcript") or []
        if message_index >= len(transcript):
            raise ValueError("Transcript reference not found")
        calls = transcript[message_index].get("tool_calls") or []
        if call_index >= len(calls) or field not in ("name", "arguments"):
            raise ValueError("Call field reference not found")
        function = calls[call_index].get("function") or calls[call_index]
        return function.get(field)

    @staticmethod
    def _resolve_message(entry: dict[str, Any], index: int, field: str | None) -> Any:
        transcript = entry.get("transcript") or []
        if index >= len(transcript):
            raise ValueError("Transcript reference not found")
        message = transcript[index]
        if field is None:
            return (
                {key: value for key, value in message.items() if key in _MESSAGE_FIELDS}
                if isinstance(message, dict)
                else message
            )
        if field not in _MESSAGE_FIELDS or not isinstance(message, dict) or field not in message:
            raise ValueError("Message field not found")
        return message[field]
