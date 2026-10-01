"""Shared structured-output helpers for prompt-authored and Python-authored operations.

Operations that ask a model for a JSON object matching a Pydantic schema share
two failure modes:

1. The provider returns prose, fenced JSON, or extra commentary instead of a
   bare JSON object. ``parse_structured_payload`` tolerates that by extracting
   the first ``{...}`` block before validating.
2. The provider (or the host's ``complete_fn``) does not honor
   ``response_format=<PydanticModel>`` at all and emits free-form text. In that
   case the caller should re-issue the call with an explicit JSON-only
   instruction. ``json_only_retry_messages`` builds that retry payload, and
   ``run_structured_completion_with_retry`` wraps the whole parse-retry loop.

These helpers are runtime-neutral and depend only on Pydantic and the kernel
operation execution context. They are intentionally host-agnostic: the host's
``complete_fn`` is the only provider touchpoint.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "completion_content",
    "extract_json_object",
    "is_structured_output_unsupported",
    "json_only_instruction",
    "json_only_retry_messages",
    "parse_structured_payload",
    "reasoning_tail_excerpt",
    "run_structured_completion_with_retry",
]

#: Default cap on the quoted reasoning tail (characters). Reasoning trails
#: end closest to the intended answer, so the tail — not the head — is the
#: most informative excerpt; 800 chars keeps the quote readable without
#: flooding the user with a wall of thought.
REASONING_TAIL_MAX_CHARS = 800

#: Marker a caller stamps on the turn result / thread entry when the
#: surfaced content is a reasoning-tail quote, not a model answer. Playback
#: surfaces read this to apply redaction and bounds independently of the
#: exhaustive-at-rest thread log.
REASONING_TAIL_CONTENT_SOURCE = "reasoning_tail"


def completion_content(completion: Any) -> str:
    """Extract the text payload from a completion object, safely.

    This is the canonical content extractor for operations that call
    ``ctx.host.complete`` and need the model's text answer. It handles the
    three shapes a host's ``complete_fn`` can return:

    1. An object with a ``content`` attribute (the normal case —
       ``StubCompletion``, ``CompletionResult``, etc.).
    2. A reasoning model whose visible ``content`` is empty while the
       substantive answer lives in ``reasoning_content`` (gpt-5.x,
       glm-5.2, kimi-k2.6). The reasoning text is folded into the return
       so structured-output parsers see the real answer instead of an
       empty string.
    3. A bare string (some test stubs and thin wrappers).

    **Never** falls back to ``str(completion)``. That was the root cause
    of the 2026-07-30 plan_workflow incident: when ``content`` was empty,
    callers wrote ``getattr(c, "content", "") or str(c)``, which produced
    the Python repr of the completion object (e.g.
    ``"StubCompletion(content='', reasoning_content='')"``) and handed
    that to ``parse_structured_payload``. The repr is never valid JSON, so
    every model that returned empty content hit a spurious parse failure
    and a misleading "be more clear" recovery prompt — the request was
    fine, the parser was eating Python object reprs.

    Returns an empty string only when both ``content`` and
    ``reasoning_content`` are absent or blank; callers should treat that as
    a genuine "provider returned nothing" condition (handled by
    ``EmptyStructuredOutputError`` in ``parse_structured_payload``).
    """
    # Bare string (test stubs, thin wrappers).
    if isinstance(completion, str):
        return completion
    content = getattr(completion, "content", "")
    if not isinstance(content, str):
        content = str(content or "")
    if content.strip():
        return content
    # Visible answer empty — fold reasoning_content for reasoning models.
    reasoning = getattr(completion, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


def reasoning_tail_excerpt(
    completion: Any,
    max_chars: int = REASONING_TAIL_MAX_CHARS,
) -> tuple[str, str]:
    """Return ``(visible_content, caveat_block)`` for one completion.

    The turn loop's *user-facing* display sites use this instead of
    ``completion_content``: when the model produced visible text, it is
    returned unchanged and the caveat is empty. When the visible answer is
    empty but reasoning exists (a reasoning model that burned its
    completion budget on thinking — the 2026-09-02 glm/Ollama incident
    and its 2026-09-18 rambling relapse), the caveat block quotes the
    TAIL of the reasoning with an explicit "this is thinking, not an
    answer" label instead of presenting raw thought as the response.

    The fold inside ``completion_content`` stays untouched: parsers rely on
    it (gpt-5.x/glm/kimi legitimately put the real answer in
    ``reasoning_content``), and the in-turn message history keeps the full
    reasoning. Only what is surfaced to the user is bounded here; the
    thread log stays exhaustive by recording the full reasoning under a
    separate ``reasoning_text`` field.

    ``finish_reason == "length"`` adds a truncation clause — it means
    reasoning consumed the completion budget before any visible text.

    Returns ``("", "")`` only when the completion has neither visible
    content nor reasoning (a genuine provider-empty response; callers
    keep their existing empty-completion handling for that case).
    """
    if isinstance(completion, str):  # bare-string stubs
        return (completion, "")
    content = getattr(completion, "content", "")
    if not isinstance(content, str):
        content = str(content or "")
    if content.strip():
        return (content.strip(), "")
    reasoning = getattr(completion, "reasoning_content", None)
    if not isinstance(reasoning, str) or not reasoning.strip():
        # Ollama's OpenAI-compatible endpoint names the field ``reasoning``.
        reasoning = getattr(completion, "reasoning", None)
        if not isinstance(reasoning, str) or not reasoning.strip():
            return ("", "")
    finish_reason = getattr(completion, "finish_reason", None)
    truncation_clause = (
        " — reasoning consumed the completion budget before any visible text"
        if finish_reason == "length"
        else ""
    )
    tail = _clip_tail(reasoning.strip(), max_chars)
    caveat = (
        f"*The model produced no visible answer{truncation_clause} — only "
        f"internal reasoning. Tail of its thinking, not a response:*\n\n"
        f"> {tail}"
    )
    return ("", caveat)


def _clip_tail(text: str, max_chars: int) -> str:
    """Return the last ``max_chars`` of ``text`` at a clean boundary.

    Backs off to the nearest sentence/word boundary so the quote does not
    end mid-word, and prefixes an ellipsis when text was clipped. Newlines
    inside the quoted span collapse to ``" | "`` so the quote stays a
    single readable line.
    """
    if len(text) <= max_chars:
        clipped = text
        prefix = ""
    else:
        clipped = text[-max_chars:]
        prefix = "… "
        for sep in (". ", ".\n", "\n", " "):
            idx = clipped.find(sep)
            if idx >= 0 and len(clipped) - (idx + len(sep)) > max_chars // 4:
                clipped = clipped[idx + len(sep) :]
                break
    return prefix + clipped.strip().replace("\n", " | ")


def extract_json_object(text: str) -> str:
    """Extract the first likely JSON object from a model response.

    Strips Markdown fences and surrounding prose. Returns the original
    string (stripped) when no ``{...}`` block is found so callers can
    distinguish "no JSON found" from "empty response".
    """

    normalized = str(text or "").strip()
    if normalized.startswith("```json"):
        normalized = normalized[7:]
    elif normalized.startswith("```"):
        normalized = normalized[3:]
    if normalized.endswith("```"):
        normalized = normalized[:-3]
    normalized = normalized.strip()

    if normalized.startswith("{") and normalized.endswith("}"):
        return normalized

    start = normalized.find("{")
    end = normalized.rfind("}")
    if start >= 0 and end > start:
        return normalized[start : end + 1]
    return normalized


class EmptyStructuredOutputError(ValueError):
    """Raised when a structured-output completion returned no text at all.

    Distinct from ``ValidationError``: the provider emitted nothing parseable
    (empty string or whitespace only), so there is no JSON to validate.
    Callers should treat this as a hard upstream failure (the model returned
    no content on a JSON-only retry) rather than a malformed-JSON retry case.
    """


def parse_structured_payload(
    response_format_model: type[BaseModel],
    content: str,
) -> BaseModel:
    """Parse structured output, tolerating fenced JSON and wrapper prose.

    Tries a direct ``model_validate_json`` first, then falls back to
    extracting the first ``{...}`` block. Raises ``EmptyStructuredOutputError``
    (a ``ValueError`` subclass) when ``content`` is empty/whitespace, so
    callers can distinguish "provider returned nothing" from "provider
    returned malformed JSON". Raises ``ValidationError`` when the content
    is non-empty but neither attempt yields a valid payload.
    """

    if not str(content or "").strip():
        raise EmptyStructuredOutputError(
            "Empty structured-output content; provider returned no text"
        )

    try:
        return response_format_model.model_validate_json(content)
    except ValidationError as direct_exc:
        extracted_json = extract_json_object(content)
        if extracted_json == content:
            raise
        try:
            return response_format_model.model_validate_json(extracted_json)
        except ValidationError as extracted_exc:
            # The extracted object's schema errors are the actionable ones; the
            # direct error only says the fenced text was not bare JSON.
            raise extracted_exc from direct_exc


def json_only_instruction(response_format_model: type[BaseModel]) -> str:
    """Return a compact JSON-only instruction for response-format fallbacks.

    The instruction embeds the schema's JSON shape, not just its key names:
    reasoning-heavy models (the 2026-09-23 finances-profile distill failures)
    answered the bare key-list variant with analysis prose or wrapped each
    array element in an object (``{"facts": [{"fact": ...}]}``), both of which
    fail validation. An inline example derived from the schema pins the
    expected element types.
    """

    schema = response_format_model.model_json_schema()
    properties = schema.get("properties") or {}
    keys = [str(key) for key in properties]
    key_list = ", ".join(keys) if keys else "the declared schema"
    return (
        "Return one JSON object only. Do not wrap it in markdown fences or prose. "
        f"Populate these top-level keys: {key_list}. "
        f"Match this JSON shape: {json.dumps(schema, separators=(', ', ': '))}"
    )


def json_only_retry_messages(
    messages: list[dict[str, Any]],
    response_format_model: type[BaseModel],
) -> list[dict[str, Any]]:
    """Append a JSON-only retry instruction to a prior message list."""

    return [
        *messages,
        {
            "role": "user",
            "content": json_only_instruction(response_format_model),
        },
    ]


def is_structured_output_unsupported(exc: Exception) -> bool:
    """Return whether the provider rejected native structured output for this request."""

    text = f"{type(exc).__name__}: {exc}".lower()
    if "unsupported" not in text:
        return False
    return any(
        marker in text
        for marker in (
            "requested operation",
            "response_format",
            "structured output",
            "chat.completions.parse",
        )
    )


async def run_structured_completion_with_retry(
    *,
    complete_fn: Callable[..., Awaitable[Any]],
    messages: list[dict[str, Any]],
    model_def: Any,
    response_format_model: type[BaseModel],
    max_tokens: int,
    temperature: float = 0.2,
    tools: list[dict[str, Any]] | None = None,
    completion_content_fn: Callable[[Any], str] | None = None,
) -> BaseModel:
    """Run one structured completion, retrying once with a JSON-only instruction.

    Use this for Python-authored operations that call ``ctx.host.complete`` with
    ``response_format=<PydanticModel>``. It mirrors the retry behavior the
    declarative operation runtime and the router planner already use:

    1. Issue the completion with ``response_format`` set.
    2. If the provider rejects structured output, or the response does not
       parse, re-issue the call with an explicit JSON-only instruction appended
       to the message list (and ``response_format`` cleared so the host does
       not re-reject it).
    3. Parse and return the validated payload.

    Raises ``ValidationError`` if both attempts fail to produce valid JSON.
    Raises ``RuntimeError`` if the provider rejected structured output and the
    retry also failed.

    ``completion_content_fn`` extracts the text payload from the completion
    object. It defaults to ``getattr(result, "content", "")`` so callers can
    pass a custom extractor for non-standard completion shapes.
    """

    def _content(result: Any) -> str:
        if completion_content_fn is not None:
            return completion_content_fn(result) or ""
        return completion_content(result)

    try:
        completion = await complete_fn(
            messages=messages,
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_format=response_format_model,
        )
    except Exception as exc:  # pylint: disable=broad-except
        if not is_structured_output_unsupported(exc):
            raise
        completion = await complete_fn(
            messages=json_only_retry_messages(messages, response_format_model),
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_format=None,
        )

    content = _content(completion)
    try:
        return parse_structured_payload(response_format_model, content)
    except ValidationError:
        logger.debug(
            "structured output first attempt did not parse; retrying with json-only instruction"
        )
        retry_completion = await complete_fn(
            messages=json_only_retry_messages(messages, response_format_model),
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_format=None,
        )
        return parse_structured_payload(response_format_model, _content(retry_completion))
