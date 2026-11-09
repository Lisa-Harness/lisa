"""Memory-add retirement receipts are explicit and type-checked."""

from __future__ import annotations

import json
from typing import Any

import pytest
from kernel.models import MemoryRecord, ToolResult
from sdk.memory.write_result import (
    MemoryWriteOutcome,
    format_memory_add_result,
    parse_memory_add_id,
    parse_supersession_retired,
)


@pytest.mark.parametrize("retired", [True, False])
def test_format_supersession_receipt(retired: bool) -> None:
    record = MemoryRecord(id=17, created_at="2026-01-01T00:00:00Z")
    text = format_memory_add_result(MemoryWriteOutcome(record=record, supersession_retired=retired))

    assert json.loads(text.splitlines()[0]) == {
        "memory_id": 17,
        "supersession_retired": retired,
    }
    assert text.splitlines()[1] == "Memory stored with ID 17"
    result = ToolResult(output_text=text)
    assert parse_memory_add_id(result) == 17
    assert parse_supersession_retired(result) is retired


def test_format_plain_write_keeps_existing_receipt() -> None:
    record = MemoryRecord(id=17, created_at="2026-01-01T00:00:00Z")
    text = format_memory_add_result(MemoryWriteOutcome(record=record, supersession_retired=None))

    assert text == "Memory stored with ID 17"
    assert parse_supersession_retired(ToolResult(output_text=text)) is None


@pytest.mark.parametrize(
    ("text", "structured", "expected"),
    [
        ('{"supersession_retired":true}', {}, True),
        ('{"supersession_retired":false}', {}, False),
        ("Memory stored with ID 17", {}, None),
        ("{", {}, None),
        ("[]", {}, None),
        ('{"supersession_retired":"true"}', {}, None),
        ('{"supersession_retired":1}', {}, None),
        ('{"supersession_retired":null}', {}, None),
        ("", {"supersession_retired": True}, True),
        ("", {"supersession_retired": False}, False),
        ('{"supersession_retired":true}', {"supersession_retired": "invalid"}, None),
    ],
)
def test_parse_only_explicit_boolean_retirement(
    text: str, structured: dict[str, Any], expected: bool | None
) -> None:
    result = ToolResult(output_text=text, output_json=structured)

    assert parse_supersession_retired(result) is expected


@pytest.mark.parametrize(
    ("text", "structured", "expected"),
    [
        ("Memory stored with ID 42", {}, 42),
        ("Memory stored with ID 42.", {}, 42),
        ("no marker here", {}, None),
        ("Memory stored with ID abc", {}, None),
        ('{"memory_id":42,"supersession_retired":true}', {}, 42),
        ('{"memory_id":"42"}', {}, 42),
        ('{"memory_id":true}', {}, None),
        ('{"memory_id":"external-id"}', {}, None),
        ("", {"memory_id": 42}, 42),
        ('{"memory_id":42}', {"memory_id": "invalid"}, None),
    ],
)
def test_parse_memory_id(text: str, structured: dict[str, Any], expected: int | None) -> None:
    assert parse_memory_add_id(ToolResult(output_text=text, output_json=structured)) == expected
