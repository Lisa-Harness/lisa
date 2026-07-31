"""Tests for kernel/operations/structured_output.py — structured-output parsing.

Covers the 2026-07-25 incident: the JSON-only retry path in
``kernel/operations/declarative.py`` called ``parse_structured_payload`` with
an empty completion body, and ``model_validate_json("")`` leaked an opaque
Pydantic "Invalid JSON: EOF" ``ValidationError``. The fix adds a typed
``EmptyStructuredOutputError`` (a ``ValueError`` subclass) raised before any
JSON parse attempt, so callers can distinguish "provider returned nothing"
from "provider returned malformed JSON".
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from kernel.operations.structured_output import (
    EmptyStructuredOutputError,
    extract_json_object,
    parse_structured_payload,
)


class _SamplePayload(BaseModel):
    """Minimal schema used by every test in this file."""

    query: str
    results: list[str] = []


# ── parse_structured_payload: empty-content guard ───────────────────────


def test_parse_structured_payload_empty_string_raises_empty_error() -> None:
    """An empty completion body raises ``EmptyStructuredOutputError``, not a
    raw Pydantic ``ValidationError``. This is the 2026-07-25 incident's exact
    failure mode — the JSON-only retry returned no text."""
    with pytest.raises(EmptyStructuredOutputError) as exc_info:
        parse_structured_payload(_SamplePayload, "")
    assert "provider returned no text" in str(exc_info.value)


def test_parse_structured_payload_whitespace_only_raises_empty_error() -> None:
    """Whitespace-only content is treated the same as empty — no JSON to
    parse, so the typed empty-content error fires before ``model_validate_json``."""
    with pytest.raises(EmptyStructuredOutputError):
        parse_structured_payload(_SamplePayload, "   \n  \t ")


def test_parse_structured_payload_none_content_raises_empty_error() -> None:
    """``None`` (a non-str completion) is coerced to empty and raises the
    typed error, not an ``AttributeError`` or ``ValidationError``."""
    with pytest.raises(EmptyStructuredOutputError):
        parse_structured_payload(_SamplePayload, None)  # type: ignore[arg-type]


def test_empty_structured_output_error_is_value_error_subclass() -> None:
    """Callers catching ``ValueError`` (the historical broad catch) still
    catch the empty-content case, but they can also narrow to
    ``EmptyStructuredOutputError`` to distinguish it from a malformed-JSON
    ``ValidationError`` (which is a ``ValueError`` too)."""
    assert issubclass(EmptyStructuredOutputError, ValueError)


# ── parse_structured_payload: non-empty still works ─────────────────────


def test_parse_structured_payload_bare_json_succeeds() -> None:
    """A bare JSON object validates directly — unchanged behavior."""
    parsed = parse_structured_payload(_SamplePayload, '{"query": "test", "results": ["a"]}')
    assert parsed.query == "test"
    assert parsed.results == ["a"]


def test_parse_structured_payload_fenced_json_succeeds() -> None:
    """Fenced JSON (```json ... ```) is extracted and validates — unchanged
    behavior for the tolerant-parse path."""
    fenced = '```json\n{"query": "test"}\n```'
    parsed = parse_structured_payload(_SamplePayload, fenced)
    assert parsed.query == "test"


def test_parse_structured_payload_prose_wrapped_json_succeeds() -> None:
    """Prose surrounding a JSON object is tolerated via ``extract_json_object``."""
    prose = 'Here is the result:\n{"query": "test", "results": ["x"]}\nThat is all.'
    parsed = parse_structured_payload(_SamplePayload, prose)
    assert parsed.query == "test"
    assert parsed.results == ["x"]


def test_parse_structured_payload_malformed_non_empty_raises_validation_error() -> None:
    """Non-empty but invalid JSON still raises ``ValidationError`` — the
    empty guard does NOT mask genuine malformed-JSON failures."""
    with pytest.raises(ValidationError):
        parse_structured_payload(_SamplePayload, '{"query": "missing close bracket"')


# ── extract_json_object ────────────────────────────────────────────────


def test_extract_json_object_no_braces_returns_stripped_original() -> None:
    """When there is no ``{...}`` block, the original stripped string is
    returned so callers can distinguish "no JSON found" from "empty"."""
    assert extract_json_object("just prose, no json here") == "just prose, no json here"


def test_extract_json_object_strips_fences() -> None:
    """Markdown fences are stripped before extraction."""
    assert extract_json_object('```json\n{"a": 1}\n```') == '{"a": 1}'
