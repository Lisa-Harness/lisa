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

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "extract_json_object",
    "is_structured_output_unsupported",
    "json_only_instruction",
    "json_only_retry_messages",
    "parse_structured_payload",
    "run_structured_completion_with_retry",
]


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


def parse_structured_payload(
    response_format_model: type[BaseModel],
    content: str,
) -> BaseModel:
    """Parse structured output, tolerating fenced JSON and wrapper prose.

    Tries a direct ``model_validate_json`` first, then falls back to
    extracting the first ``{...}`` block. Raises ``ValidationError`` if
    neither attempt yields a valid payload.
    """

    try:
        return response_format_model.model_validate_json(content)
    except ValidationError as direct_exc:
        extracted_json = extract_json_object(content)
        if extracted_json == content:
            raise
        try:
            return response_format_model.model_validate_json(extracted_json)
        except ValidationError as extracted_exc:
            raise direct_exc from extracted_exc


def json_only_instruction(response_format_model: type[BaseModel]) -> str:
    """Return a compact JSON-only instruction for response-format fallbacks."""

    schema = response_format_model.model_json_schema()
    properties = schema.get("properties") or {}
    keys = [str(key) for key in properties.keys()]
    key_list = ", ".join(keys) if keys else "the declared schema"
    return (
        "Return one JSON object only. Do not wrap it in markdown fences or prose. "
        f"Populate these top-level keys: {key_list}."
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

    Use this for Python-authored operations that call ``ctx.complete_fn`` with
    ``response_format=<PydanticModel>``. It mirrors the retry behavior the
    declarative operation runtime and the frontdoor planner already use:

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
        return str(getattr(result, "content", "") or "")

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
