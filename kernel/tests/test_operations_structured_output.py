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

import json

import pytest
from kernel.operations.structured_output import (
    REASONING_TAIL_CONTENT_SOURCE,
    DistillationLadder,
    EmptyStructuredOutputError,
    assistant_bootstrap_messages,
    extract_json_object,
    parse_structured_payload,
    reasoning_tail_excerpt,
    run_distillation_ladder,
    salvage_truncated_envelope,
    unwrap_nested_envelope,
    validation_feedback_retry_messages,
)
from pydantic import BaseModel, ValidationError


class _SamplePayload(BaseModel):
    """Minimal schema used by every test in this file."""

    query: str
    results: list[str] = []


class _ArrayPayload(BaseModel):
    """Schema with a top-level array property, for bootstrap-prefix tests."""

    query: str = "q"
    facts: list[str] = []


class _ScalarPayload(BaseModel):
    """Schema with only scalar properties."""

    answer: str = ""


class _FactPayload(BaseModel):
    """Schema whose array property holds objects — the distiller shape."""

    facts: list[_SamplePayload] = []


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
    quote_line = next(line for line in caveat.splitlines() if line.startswith("> "))
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


def test_parse_structured_payload_fenced_schema_error_reports_schema_problem() -> None:
    """When fenced JSON parses but fails the schema, the raised error names
    the schema problem, not "Invalid JSON" for the fence."""
    with pytest.raises(ValidationError) as excinfo:
        parse_structured_payload(_SamplePayload, '```json\n{"results": ["a"]}\n```')
    assert "query" in str(excinfo.value)
    assert "json_invalid" not in str(excinfo.value)


# ── truncated-envelope and re-nested-envelope recovery ──


def test_salvage_recovers_truncated_array_keeping_complete_elements() -> None:
    """A token-budget truncation mid-array keeps complete elements and
    closes the envelope (the 2026-11-08 synthesize incidents)."""
    truncated = '{"facts": [{"action": "create", "fact": "one"}, {"action": "up'
    recovered = salvage_truncated_envelope(truncated)
    assert json.loads(recovered) == {"facts": [{"action": "create", "fact": "one"}]}


def test_salvage_ignores_string_only_arrays() -> None:
    """Salvage is object-scoped: a truncated string-element array has no
    complete object element, so the input passes through unchanged."""
    text = '{"query": "test", "results": ["alpha", "beta"'
    assert salvage_truncated_envelope(text) == text


def test_salvage_returns_input_unchanged_when_nothing_completes() -> None:
    """No balanced element at all returns the input unchanged."""
    text = '{"facts": [{"action": "up'
    assert salvage_truncated_envelope(text) == text


def test_unwrap_flattens_single_renested_envelope() -> None:
    """The 2026-11-10..11 shape — the example envelope echoed as the first
    array element — flattens to the inner facts array."""
    nested = '{"facts": [{"facts": [{"action": "create", "fact": "one"}]}]}'
    assert json.loads(unwrap_nested_envelope(nested)) == {
        "facts": [{"action": "create", "fact": "one"}]
    }


def test_unwrap_requires_matching_inner_key() -> None:
    """A different inner key is left alone — no unrelated flattening."""
    other_key = '{"facts": [{"items": [{"a": 1}]}]}'
    assert unwrap_nested_envelope(other_key) == other_key


def test_parse_structured_payload_salvages_truncated_object_array() -> None:
    """End-to-end: a truncated fact-object array parses via the recovery pass."""
    truncated = '{"facts": [{"query": "a"}, {"query": "b"'
    parsed = parse_structured_payload(_FactPayload, truncated)
    assert [f.query for f in parsed.facts] == ["a"]


def test_parse_structured_payload_unwraps_renested_payload() -> None:
    """End-to-end: the re-nested envelope parses via the recovery pass."""
    nested = '{"facts": [{"facts": ["a", "b"]}]}'
    parsed = parse_structured_payload(_ArrayPayload, nested)
    assert parsed.facts == ["a", "b"]


# ── extract_json_object ────────────────────────────────────────────────


def test_extract_json_object_no_braces_returns_stripped_original() -> None:
    """When there is no ``{...}`` block, the original stripped string is
    returned so callers can distinguish "no JSON found" from "empty"."""
    assert extract_json_object("just prose, no json here") == "just prose, no json here"


def test_extract_json_object_strips_fences() -> None:
    """Markdown fences are stripped before extraction."""
    assert extract_json_object('```json\n{"a": 1}\n```') == '{"a": 1}'


# ── assistant_bootstrap_messages ────────────────────────────────────────────


def test_assistant_bootstrap_prefills_requested_top_key() -> None:
    """The bootstrap retry rehydrates the conversation and prefills the assistant
    turn with the requested top-key array opening."""
    messages = [{"role": "user", "content": "produce facts"}]
    bootstrapped = assistant_bootstrap_messages(messages, _SamplePayload, top_key="results")
    assert bootstrapped[:-1] == messages
    assert bootstrapped[-1]["role"] == "assistant"
    assert bootstrapped[-1]["content"] == '{"results": ['


def test_assistant_bootstrap_default_is_bare_object_opening() -> None:
    """Without a top key the prefill is the bare JSON object opening, which
    keeps the helper generic over any schema."""
    messages = [{"role": "user", "content": "produce payload"}]
    bootstrapped = assistant_bootstrap_messages(messages, _SamplePayload)
    assert bootstrapped[-1]["role"] == "assistant"
    assert bootstrapped[-1]["content"] == '{"'


def test_assistant_bootstrap_does_not_mutate_input() -> None:
    """The input message list is left untouched — retries must be idempotent."""
    messages: list[dict[str, str]] = [{"role": "user", "content": "produce facts"}]
    assistant_bootstrap_messages(messages, _SamplePayload, top_key="results")
    assert messages == [{"role": "user", "content": "produce facts"}]


def test_bootstrap_prefill_concatenation_parses() -> None:
    """The model continues the prefilled envelope; prefix + continuation
    parses while the continuation alone does not — the caller must parse the
    concatenation."""
    continuation = '], "query": "test"}'
    combined = '{"results": [' + continuation
    assert parse_structured_payload(_SamplePayload, combined) == _SamplePayload(
        query="test", results=[]
    )


# ── validation_feedback_retry_messages ──────────────────────────────────────


def test_validation_feedback_names_the_failing_fields_and_note() -> None:
    """The feedback re-ask carries the validation summary and the constraint
    note — a bare re-ask reproduces the failure (the 2026-11-09 synthesize
    incidents: three consecutive over-length facts because the re-ask never
    said the cap)."""
    messages = [{"role": "user", "content": "produce payload"}]
    retried = validation_feedback_retry_messages(
        messages,
        _SamplePayload,
        "query: string_too_long",
        note="Each 'query' must be at most 3 characters.",
    )
    assert retried[:-1] == messages
    assert retried[-1]["role"] == "user"
    text = retried[-1]["content"]
    assert "failed validation: query: string_too_long" in text
    assert "at most 3 characters" in text
    # The schema echo stays so the shape contract holds.
    assert "Return one JSON object only" in text


def test_validation_feedback_without_note_still_names_fields() -> None:
    messages = [{"role": "user", "content": "produce payload"}]
    retried = validation_feedback_retry_messages(
        messages, _SamplePayload, "results: json_invalid", note=""
    )
    text = retried[-1]["content"]
    assert "failed validation: results: json_invalid" in text
    assert "failed validation: results: json_invalid." in text


def test_validation_feedback_does_not_mutate_input() -> None:
    messages: list[dict[str, str]] = [{"role": "user", "content": "produce payload"}]
    validation_feedback_retry_messages(messages, _SamplePayload, "query: string_too_long")
    assert messages == [{"role": "user", "content": "produce payload"}]


# ── run_distillation_ladder ─────────────────────────────────────────────────


class _Completion:
    """Minimal completion stub returning fixed content."""

    def __init__(self, content: str) -> None:
        self.content = content


def _fail_closed(text: str) -> _ArrayPayload:
    return parse_structured_payload(_ArrayPayload, text)


def _bootstrap_prefix() -> str:
    return '{"facts": ['


@pytest.mark.asyncio
async def test_ladder_first_rung_native_structured_output_parses() -> None:
    """A compliant first attempt answers without any retry."""
    calls: list[dict] = []

    async def complete_fn(**kwargs):
        calls.append(kwargs)
        return _Completion('{"facts": ["f1"], "query": "q"}')

    payload = await run_distillation_ladder(
        ladder=DistillationLadder(response_format_model=_ArrayPayload, parse_content=_fail_closed),
        complete_fn=complete_fn,
        messages=[{"role": "user", "content": "distill"}],
        model_def={"model_id": "m"},
        max_tokens=1000,
        temperature=0.1,
        operation_label="test op",
    )

    assert payload.facts == ["f1"]
    assert len(calls) == 1
    assert calls[0]["response_format"] is _ArrayPayload
    assert calls[0]["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_ladder_validation_error_fires_feedback_rung() -> None:
    """A JSON-shaped response failing a schema bound triggers exactly one
    validation-feedback re-ask, which names the failing fields and the note."""
    calls: list[list[dict]] = []

    async def complete_fn(messages=None, **kwargs):
        calls.append(list(messages or []))
        if len(calls) == 1:
            # Over-length array item: fails the schema only via custom parse.
            return _Completion('{"facts": ["' + "x" * 50 + '"], "query": "q"}')
        return _Completion('{"facts": ["ok"], "query": "q"}')

    def parse(text: str) -> _ArrayPayload:
        payload = parse_structured_payload(_ArrayPayload, text)
        for fact in payload.facts:
            if len(fact) > 20:
                raise ValidationError.from_exception_data(
                    title="array",
                    line_errors=[
                        {
                            "type": "string_too_long",
                            "loc": ("facts", 0, "fact"),
                            "input": fact,
                            "ctx": {"max_length": 20},
                        }
                    ],
                )
        return payload

    result = await run_distillation_ladder(
        ladder=DistillationLadder(
            response_format_model=_ArrayPayload,
            parse_content=parse,
            validation_note="Each 'fact' must be at most 20 characters.",
        ),
        complete_fn=complete_fn,
        messages=[{"role": "user", "content": "distill"}],
        model_def={"model_id": "m"},
        max_tokens=1000,
        operation_label="test op",
    )

    assert result.facts == ["ok"]
    # Rung 1 failed validation → rung 2 is the feedback re-ask (it skips the
    # bare JSON-only rung, which cannot fix a bound the model was never told).
    assert len(calls) == 2
    feedback = calls[-1][-1]["content"]
    assert "failed validation" in feedback
    assert "string_too_long" in feedback
    assert "at most 20 characters" in feedback
    assert calls[-1][0]["role"] == "assistant" or calls[-1][-1]["role"] == "user"


@pytest.mark.asyncio
async def test_ladder_prose_failure_reaches_bootstrap_and_parses_prefix() -> None:
    """Reasoning prose on the first rung climbs to the bootstrap prefill, and
    the bootstrap rung parses prefix + continuation."""
    calls: list[list[dict]] = []

    async def complete_fn(messages=None, **kwargs):
        calls.append(list(messages or []))
        if calls[-1] and calls[-1][-1].get("role") == "assistant":
            return _Completion('{"fact": "bootstrapped"}]}')
        return _Completion("Let me analyze this conversation segment.")

    payload = await run_distillation_ladder(
        ladder=DistillationLadder(
            response_format_model=_ArrayPayload,
            parse_content=parses_facts_array,
            bootstrap_prefix=_bootstrap_prefix(),
        ),
        complete_fn=complete_fn,
        messages=[{"role": "user", "content": "distill"}],
        model_def={"model_id": "m"},
        max_tokens=1000,
        operation_label="test op",
    )

    assert payload.facts == ["bootstrapped"]
    assert len(calls) == 3
    assert calls[-1][-1]["role"] == "assistant"
    assert calls[-1][-1]["content"] == '{"facts": ['


def parses_facts_array(text: str) -> _ArrayPayload:
    """Parse only the facts array of a continuation-shaped envelope."""
    import json as _json

    payload = _json.loads(text)
    return _ArrayPayload(facts=[item["fact"] for item in payload["facts"]])


@pytest.mark.asyncio
async def test_ladder_all_rungs_failing_raises_typed_error() -> None:
    """Every rung failing still raises the typed distillation failure."""

    async def complete_fn(messages=None, **kwargs):
        return _Completion("Let me analyze this conversation segment.")

    with pytest.raises(RuntimeError, match="test op distillation failed"):
        await run_distillation_ladder(
            ladder=DistillationLadder(
                response_format_model=_ArrayPayload,
                parse_content=_fail_closed,
                bootstrap_prefix=_bootstrap_prefix(),
            ),
            complete_fn=complete_fn,
            messages=[{"role": "user", "content": "distill"}],
            model_def={"model_id": "m"},
            max_tokens=1000,
            operation_label="test op",
        )


def test_bootstrap_prefix_inferred_from_array_schema() -> None:
    """Without an override the bootstrap prefix opens the schema's first
    array property; a scalar-only schema opens the first scalar."""
    import kernel.operations.structured_output as so

    assert so._bootstrap_prefix_for(_ArrayPayload, "") == '{"facts": ['
    assert so._bootstrap_prefix_for(_ArrayPayload, '{"facts": [') == '{"facts": ['
    # Scalar-only schema opens the first scalar — still tighter than the
    # bare object opening.
    assert so._bootstrap_prefix_for(_ScalarPayload, "") == '{"answer": '
    # The bare opening remains the last-resort default for object-typed opens.
    assert so._ASSISTANT_BOOTSTRAP_PREFIX == '{"'
