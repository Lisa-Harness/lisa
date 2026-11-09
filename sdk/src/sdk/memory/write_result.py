"""Memory-write outcomes and the ``memory.add`` result contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from kernel.models import MemoryRecord, ToolResult

__all__ = [
    "MemoryWriteOutcome",
    "format_memory_add_result",
    "parse_memory_add_id",
    "parse_supersession_retired",
]


@dataclass(frozen=True, slots=True)
class MemoryWriteOutcome:
    """A stored record and its write's retirement outcome, or None without a target."""

    record: MemoryRecord
    supersession_retired: bool | None


def format_memory_add_result(outcome: MemoryWriteOutcome) -> str:
    """Render JSON metadata for supersession before the human-readable receipt."""
    message = f"Memory stored with ID {outcome.record.id}"
    if outcome.supersession_retired is None:
        return message
    metadata = json.dumps(
        {
            "memory_id": outcome.record.id,
            "supersession_retired": outcome.supersession_retired,
        },
        separators=(",", ":"),
    )
    return f"{metadata}\n{message}"


def _result_metadata(result: ToolResult) -> dict[str, Any]:
    try:
        metadata = json.loads(result.output_text.partition("\n")[0])
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {**metadata, **result.output_json}


def parse_memory_add_id(result: ToolResult) -> int | None:
    """Read the numeric ID from structured or human-readable add receipts."""
    metadata = _result_metadata(result)
    if "memory_id" in metadata:
        value = metadata["memory_id"]
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return None
        try:
            return int(value)
        except ValueError:
            return None
    if "ID " not in result.output_text:
        return None
    try:
        return int(result.output_text.split("ID ", 1)[1].rstrip("."))
    except (ValueError, IndexError):
        return None


def parse_supersession_retired(result: ToolResult) -> bool | None:
    """Read a confirmed retirement outcome, or return ``None`` if unreported."""
    retired = _result_metadata(result).get("supersession_retired")
    return retired if isinstance(retired, bool) else None
