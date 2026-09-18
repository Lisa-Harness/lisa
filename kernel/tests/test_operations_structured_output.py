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
    REASONING_TAIL_CONTENT_SOURCE,
    EmptyStructuredOutputError,
    extract_json_object,
    parse_structured_payload,
    reasoning_tail_excerpt,
)


class _SamplePayload(BaseModel):
    """Minimal schema used by every test in this file."""

    query: str
    results: list[str] = []


class _StubCompletion:
    """Minimal completion stub for extractor tests."""

    def __init__(
        self,
        content: str = "",
        reasoning_content: str = "",
        finish_reason: str | None = None,
    ) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.finish_reason = finish_reason


# ── reasoning_tail_excerpt: visible content wins ───────────────────────


def test_reasoning_tail_excerpt_visible_content_wins() -> None:
    """When the model produced visible text, it is returned unchanged and
    no caveat is built — reasoning is ignored entirely."""
    completion = _StubCompletion(
        content="Here is the answer.",
        reasoning_content="I should think about this first.",
    )
    visible, caveat = reasoning_tail_excerpt(completion)
    assert visible == "Here is the answer."
    assert caveat == ""


def test_reasoning_tail_excerpt_bare_string_passthrough() -> None:
    """A bare-string completion (test stubs) passes through with no caveat."""
    visible, caveat = reasoning_tail_excerpt("plain text")
    assert visible == "plain text"
    assert caveat == ""


# ── reasoning_tail_excerpt: reasoning-tail caveat ────────────────────────


def test_reasoning_tail_excerpt_empty_content_builds_labeled_caveat() -> None:
    """Empty visible content with reasoning builds a caveat that quotes the
    reasoning tail with an explicit not-an-answer label (the 2026-09-18
    rambling fix)."""
    completion = _StubCompletion(
        content="",
        reasoning_content="step one\nstep two\nthe answer is probably 42",
        finish_reason="stop",
    )
    visible, caveat = reasoning_tail_excerpt(completion)
    assert visible == ""
    assert "no visible answer" in caveat
    assert "not a response" in caveat
    assert "the answer is probably 42" in caveat


def test_reasoning_tail_excerpt_length_finish_reason_adds_truncation_clause() -> None:
    """``finish_reason == "length"`` means reasoning consumed the completion
    budget — the caveat says so."""
    completion = _StubCompletion(
        content="",
        reasoning_content="thinking and thinking and",
        finish_reason="length",
    )
    _, caveat = reasoning_tail_excerpt(completion)
    assert "consumed the completion budget" in caveat


def test_reasoning_tail_excerpt_clips_long_reasoning_to_tail() -> None:
    """Reasoning longer than ``max_chars`` is clipped to the TAIL (the tail
    ends closest to the intended answer) with an ellipsis prefix; the quote
    is bounded and ends with the reasoning's own tail."""
    head = "early thinking. " * 50  # > 400 chars of noise
    tail = "final consideration. the answer is 42."
    completion = _StubCompletion(
        content="",
        reasoning_content=head + tail,
        finish_reason="stop",
    )
    _, caveat = reasoning_tail_excerpt(completion, max_chars=120)
    quote_line = [line for line in caveat.splitlines() if line.startswith("> ")][0]
    # Clipped tail, not the full text.
    assert quote_line.startswith("> …")
    # The quote is bounded by max_chars (plus the ellipsis prefix).
    assert len(quote_line) <= 120 + len("> …")
    # Ends with the reasoning's tail — the informative end of the thinking.
    assert quote_line.endswith("the answer is 42.")


def test_reasoning_tail_excerpt_folds_newlines_in_quote() -> None:
    """Newlines inside the quoted tail collapse to ``" | "`` so the quote
    stays a single readable line."""
    completion = _StubCompletion(
        content="",
        reasoning_content="alpha\nbeta\ngamma",
        finish_reason="stop",
    )
    _, caveat = reasoning_tail_excerpt(completion)
    assert "alpha | beta | gamma" in caveat


def test_reasoning_tail_excerpt_ollama_reasoning_attr_fallback() -> None:
    """Ollama's OpenAI-compatible endpoint names the field ``reasoning`` —
    the helper falls back to it when ``reasoning_content`` is absent."""

    class _OllamaCompletion:
        content = ""
        reasoning = "ollama-style thinking"
        finish_reason = "stop"

    _, caveat = reasoning_tail_excerpt(_OllamaCompletion())
    assert "ollama-style thinking" in caveat


def test_reasoning_tail_excerpt_empty_everywhere_returns_empty_pair() -> None:
    """No visible content AND no reasoning is a genuine provider-empty
    response — both return values are empty so callers keep their existing
    empty-completion handling."""
    visible, caveat = reasoning_tail_excerpt(_StubCompletion(content=""))
    assert visible == ""
    assert caveat == ""


def test_reasoning_tail_content_source_marker_value() -> None:
    """The marker playback surfaces switch on is ``"reasoning_tail"``."""
    assert REASONING_TAIL_CONTENT_SOURCE == "reasoning_tail"


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
