"""Shared structured-output helpers for prompt-authored and Python-authored operations.

Operations that ask a model for a JSON object matching a Pydantic schema share
a small set of failure modes, and fixing them twice (once per operation) is
how the 2026-11-08..09 distill incidents turned into whack-a-mole:

1. The provider returns prose, fenced JSON, or extra commentary instead of a
   bare JSON object. ``parse_structured_payload`` tolerates that by extracting
   the first ``{...}`` block before validating.
2. The provider (or the host's ``complete_fn``) does not honor
   ``response_format=<PydanticModel>`` at all and emits free-form text. In that
   case the caller should re-issue the call with an explicit JSON-only
   instruction. ``json_only_retry_messages`` builds that retry payload, and
   ``run_structured_completion_with_retry`` wraps the whole parse-retry loop.
3. Reasoning-heavy models answer even the explicit JSON-only instruction
   with analysis prose before any JSON. ``assistant_bootstrap_messages``
   builds the final retry: it replays the conversation and prefills the
   assistant turn with the schema's opening envelope text so the model
   continues *inside* the JSON object instead of reasoning first (the
   2026-11-08..09 mining crash-loop's last unaddressed rung).
4. A JSON-shaped response that fails a schema *bound* (e.g. a per-fact
   character cap) is a failure the re-ask can fix, but only if the re-ask
   names the bound. ``validation_feedback_retry_messages`` carries the
   failing-field summary plus the constraint note.

``run_distillation_ladder`` composes all of these into one escalation
sequence (structured → JSON-only → validation-feedback → bootstrap) behind a
``DistillationLadder`` policy object, so an operation keeps only what
genuinely differs: its parse hook and failure label. The two memory
distillers are its first consumers.

Prompt escalation alone cannot fix payloads whose malformed shape every
rung deterministically reproduces. ``parse_structured_payload`` therefore
runs one normalization pass (``salvage_truncated_envelope`` →
``unwrap_nested_envelope``) before raising, so the ladder climbs against
payloads that differ rung to rung.

These helpers are runtime-neutral and depend only on Pydantic and the kernel
operation execution context. They are intentionally host-agnostic: the host's
``complete_fn`` is the only provider touchpoint.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

PayloadT = TypeVar("PayloadT", bound=BaseModel)

logger = logging.getLogger(__name__)

__all__ = [
    "FACT_CAP_CHARS",
    "DistillationLadder",
    "assistant_bootstrap_messages",
    "completion_content",
    "extract_json_object",
    "is_structured_output_unsupported",
    "json_only_instruction",
    "json_only_retry_messages",
    "parse_structured_payload",
    "reasoning_tail_excerpt",
    "run_distillation_ladder",
    "run_structured_completion_with_retry",
    "salvage_truncated_envelope",
    "unwrap_nested_envelope",
    "validation_feedback_instruction",
    "validation_feedback_retry_messages",
    "validation_summary",
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
       empty string. This is THE canonical fold point — the provider
       gateways deliberately keep ``content`` and ``reasoning_content``
       separate so user-facing display sites can distinguish a real
       answer from leaked chain-of-thought (2026-10-04 incident).
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
    normalized = normalized.removesuffix("```")
    normalized = normalized.strip()

    if normalized.startswith("{") and normalized.endswith("}"):
        return normalized

    start = normalized.find("{")
    end = normalized.rfind("}")
    if start >= 0 and end > start:
        return normalized[start : end + 1]
    return normalized


#: Brace depth of an element inside the top-level array: object + array.
_ELEMENT_DEPTH = 2


def _mask_json_strings(text: str) -> str:
    """Replace JSON string contents with ``x`` so structure scans are simple.

    Delimiting quotes survive; content characters (including escapes) do
    not, so a brace or quote inside a fact string cannot desynchronize a
    structural pass over the masked copy.
    """

    chars = list(text)
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                chars[i] = "x"
                esc = False
            elif ch == "\\":
                chars[i] = "x"
                esc = True
            elif ch == '"':
                in_str = False
            else:
                chars[i] = "x"
            continue
        if ch == '"':
            in_str = True
    return "".join(chars)


def _scan_element_spans(masked: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans of complete object elements.

    Runs over a masked copy, so quotes never straddle a fact object's
    braces. An object element opens at element depth and collects on the
    matching ``}``; truncated tails never collect. String-only arrays
    pass through salvage unchanged (the distiller shapes are object
    arrays).
    """

    depth = 0
    elem_start = -1
    spans: list[tuple[int, int]] = []
    for i, ch in enumerate(masked):
        if ch == "{":
            if elem_start < 0 and depth == _ELEMENT_DEPTH:
                elem_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == _ELEMENT_DEPTH and elem_start >= 0:
                spans.append((elem_start, i))
                elem_start = -1
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
    return spans


def salvage_truncated_envelope(text: str) -> str:
    """Rebuild a closed JSON envelope from a truncated one, element-wise.

    Model output truncated by a token budget (Ollama mid-array, the
    2026-11-08 synthesize incidents) dies inside an open array. The scan
    masks string contents, keeps every object element that closed balanced
    and complete, skips the partial tail, and closes what was left open.
    Returns the input unchanged when it does not begin with ``{``, when
    no element completed, or when the array never opened.
    """

    normalized = str(text or "").strip()
    if not normalized or normalized[0] != "{":
        return normalized
    spans = _scan_element_spans(_mask_json_strings(normalized))
    if not spans:
        return normalized
    arr_open = normalized.find("[")
    if arr_open < 0:
        return normalized
    prefix = normalized[: arr_open + 1]
    return prefix + ",".join(normalized[start : end + 1] for start, end in spans) + "]}"


def unwrap_nested_envelope(text: str) -> str:
    """Flatten one re-nested envelope level (``{"facts": [{"facts": [...]}]}``).

    Reasoning models echo the prompt's example object as the first array
    element (the 2026-11-10..11 synthesize incidents), yielding a valid
    JSON object whose array holds an envelope instead of an element.
    Applies only when the outer envelope holds exactly one key whose array
    holds exactly one single-key envelope whose value is a non-empty list —
    and the inner key must equal the outer key (both guards stop the
    flattening from re-interpreting legitimately different payloads).
    """

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text
    if not isinstance(payload, dict) or len(payload) != 1:
        return text
    ((array_key, array),) = payload.items()
    if (
        not isinstance(array, list)
        or len(array) != 1
        or not isinstance(array[0], dict)
        or len(array[0]) != 1
    ):
        return text
    ((inner_key, inner),) = array[0].items()
    if not isinstance(inner, list) or not inner or inner_key != array_key:
        return text
    payload[array_key] = inner
    logger.warning(
        "unwrap_nested_envelope: flattened re-nested %s envelope",
        array_key,
    )
    return json.dumps(payload, ensure_ascii=False)


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

    Before raising, one deterministic normalization pass runs —
    ``salvage_truncated_envelope`` then ``unwrap_nested_envelope`` — and a
    recovered payload gets one more ``model_validate_json`` attempt. Only a
    recovered-and-valid payload succeeds here; when recovery changes
    nothing or still fails, the extracted object's ``ValidationError``
    propagates so the ladder's validation-feedback rung keys on the typed
    error exactly as before (the 2026-11-08..11 truncated and re-nested
    envelope incidents).
    """

    if not str(content or "").strip():
        raise EmptyStructuredOutputError(
            "Empty structured-output content; provider returned no text"
        )

    try:
        return response_format_model.model_validate_json(content)
    except ValidationError as direct_exc:
        extracted_json = extract_json_object(content)
        if extracted_json != content:
            try:
                return response_format_model.model_validate_json(extracted_json)
            except ValidationError:
                pass
        recovered = unwrap_nested_envelope(salvage_truncated_envelope(str(content or "")))
        if recovered != str(content or "").strip():
            try:
                return response_format_model.model_validate_json(recovered)
            except ValidationError:
                pass
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


def validation_feedback_instruction(
    response_format_model: type[BaseModel],
    error_summary: str,
    *,
    note: str = "",
) -> str:
    """Build a re-ask that carries what the last attempt got wrong.

    A bare re-ask reproduces the failure: the 2026-11-09 synthesize incidents
    returned over-length ``fact`` strings three times in a row because the
    JSON-only instruction never said the per-fact cap. The re-ask therefore
    names the failing fields (``error_summary`` — the one-line
    ``loc: type`` list from the validation error) and, when ``note`` is
    given, the constraint behind it (e.g. a character cap). The schema echo
    stays so the shape contract holds.
    """

    instruction = json_only_instruction(response_format_model)
    feedback = f"Your previous response failed validation: {error_summary}."
    if note:
        feedback += f" {note}"
    return f"{feedback} {instruction}"


def validation_feedback_retry_messages(
    messages: list[dict[str, Any]],
    response_format_model: type[BaseModel],
    error_summary: str,
    *,
    note: str = "",
) -> list[dict[str, Any]]:
    """Append a validation-feedback re-ask to a prior message list."""

    return [
        *messages,
        {
            "role": "user",
            "content": validation_feedback_instruction(
                response_format_model, error_summary, note=note
            ),
        },
    ]


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


#: Bare object opening stamped into the assistant bootstrap turn when no
#: ``top_key`` is given — generic over any schema.
_ASSISTANT_BOOTSTRAP_PREFIX = '{"'


def assistant_bootstrap_messages(
    messages: list[dict[str, Any]],
    response_format_model: type[BaseModel],
    *,
    top_key: str | None = None,
    bootstrap_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Build a final retry whose assistant turn is prefilled with JSON.

    Reasoning-heavy models can answer every re-ask with analysis prose before
    any JSON (the 2026-11-08..09 mining crash-loop: ``Let me analyze this
    conv…`` twice per segment, 21 segments, zero mined). Prefilling the
    assistant turn with the schema's opening envelope forces the completion
    to continue *inside* the JSON object instead of reasoning first.

    ``top_key`` opens the named top-level property's array (``'{"facts": ['``)
    and an explicit ``bootstrap_prefix`` wins over it; without either the
    bare ``{"`` object opening is used. The completion is a continuation of
    the prefix: the caller must parse the concatenation of prefix and
    completion, never the continuation alone.
    """

    if bootstrap_prefix is not None:
        prefix = bootstrap_prefix
    elif top_key:
        prefix = f'{{"{top_key}": ['
    else:
        prefix = _ASSISTANT_BOOTSTRAP_PREFIX
    return [
        *messages,
        {"role": "assistant", "content": prefix},
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


#: Ladder rung budgets as multipliers of the base attempt's budget. The
#: escalation mirrors the kernel's truncation-rescue pattern
#: (``finish_reason=length`` → one retry at 2×): on reasoning models the
#: visible JSON shares the completion budget with reasoning tokens, so
#: re-issuing the SAME budget with a JSON-only nudge reproduces the
#: truncation (the 2026-09-23 finances-profile incidents, and the
#: 2026-11-08..09 mining crash-loop where a 640-token budget starved every
#: attempt identically).
_LADDER_RUNG_BUDGETS = (1, 2, 2)


#: Cap note appended by the ladder's validation-feedback rung when a schema
#: bounds a string field and the re-ask must name the bound. The 2026-11-09
#: incidents: over-length ``facts.N.fact`` values returned repeatedly
#: because every re-ask was a bare JSON-only instruction that never said the
#: cap — the model reproduced the failure it was not told about.
FACT_CAP_CHARS = 400


@dataclass(frozen=True, slots=True)
class DistillationLadder(Generic[PayloadT]):
    """Policy object for the shared model-distillation retry ladder.

    Every Python-authored operation that distills content into a strict
    Pydantic payload via ``ctx.host.complete`` runs the same escalation:
    structured output → JSON-only re-ask → a validation-feedback re-ask
    naming what failed plus the string-cap constraint → an
    assistant-bootstrap prefill that continues inside the JSON envelope.
    The 2026-11-08..09 incidents (the mining crash-loop, then three
    consecutive over-length synthesize captures) were all this one ladder
    fixed rung by rung — in two hand-rolled copies that drifted. This
    policy object makes the sequence one place; the operation keeps only
    what genuinely differs: the parse hook (fail-closed mining vs
    prose-fallback synthesize) and the failure message.
    """

    response_format_model: type[PayloadT]
    #: Extracts the model's text from one completion. Raising here marks the
    #: rung failed and advances the ladder: a ``ValidationError`` naming a
    #: schema bound triggers the validation-feedback rung, anything else
    #: (prose refusal, empty output) advances without it.
    parse_content: Callable[[str], PayloadT]
    #: Constraint note the validation-feedback rung states (e.g. the per-fact
    #: character cap) — the re-ask that names what failed must also name what
    #: to fix.
    validation_note: str = ""
    #: Multiplier applied to the base budget on the retry rungs (2 escalates).
    retry_budget_multiplier: int = 2
    #: Prefix stamped into the bootstrap prefill (empty: the schema's first
    #: non-required top-level array key is inferred).
    bootstrap_prefix: str = ""


def _bootstrap_prefix_for(response_format_model: type[BaseModel], override: str) -> str:
    """Return the JSON prefix the bootstrap rung prefills the assistant with.

    An explicit ``override`` wins. Otherwise the first top-level array-or-
    object property in the schema is opened (``'{"facts": ['``); a schema
    with only scalar properties opens its first scalar (``'{"answer": '``),
    still tighter than the bare object opening. The completion continues the
    prefix, so the caller parses prefix + continuation.
    """

    if override:
        return override
    schema = response_format_model.model_json_schema()
    properties = schema.get("properties") or {}
    # Prefer an array property (the payload's list content is where schema
    # bounds bite); a scalar property is the fallback before the bare open.
    for key, spec in properties.items():
        if isinstance(spec, dict) and spec.get("type") == "array":
            return f'{{"{key}": ['
    for key, spec in properties.items():
        if (
            isinstance(spec, dict)
            and isinstance(spec.get("type"), str)
            and spec["type"] != "object"
        ):
            return f'{{"{key}": '
    return _ASSISTANT_BOOTSTRAP_PREFIX


async def run_distillation_ladder(
    *,
    ladder: DistillationLadder[PayloadT],
    complete_fn: Callable[..., Awaitable[Any]],
    messages: list[dict[str, Any]],
    model_def: Any,
    max_tokens: int,
    temperature: float = 0.1,
    operation_label: str,
) -> PayloadT:
    """Run the shared distillation retry ladder and return the parsed payload.

    Rungs, in order (each re-issued at ``retry_budget_multiplier``× the base
    budget):

    1. **Structured output** — native ``response_format``; strict hosts that
       reject it surface here.
    2. **JSON-only re-ask** — ``response_format`` cleared, the schema echoed.
       A re-ask alone reproduces whatever made attempt 1 fail (the
       2026-11-08..09 mining crash-loop: reasoning prose before any JSON,
       21 segments, zero mined).
    3. **Validation-feedback re-ask** — fired only when a rung raised a
       ``ValidationError``: the re-ask names the failing fields (the
       ``loc: type`` summary) plus ``ladder.validation_note``. The 2026-11-09
       synthesize incidents returned over-length facts three times because
       the re-ask never said the per-fact cap. Fires once per ladder, after
       whichever rung raised the typed error.
    4. **Assistant-bootstrap prefill** — the assistant turn opens the JSON
       envelope so the completion continues inside it instead of reasoning
       first. Parsed as prefix + continuation.
    """

    schema_model = ladder.response_format_model
    bootstrap_prefix = _bootstrap_prefix_for(schema_model, ladder.bootstrap_prefix)
    bootstrapped = assistant_bootstrap_messages(
        messages, schema_model, bootstrap_prefix=bootstrap_prefix
    )
    multiplier = max(1, ladder.retry_budget_multiplier)
    feedback_fired = False
    last_exc: Exception | None = None

    rungs = list(enumerate(_LADDER_RUNG_BUDGETS))
    final_rung_index = len(rungs) - 1
    for rung_index, rung_budget in rungs:
        rung_messages: list[dict[str, Any]]
        native_format: type[PayloadT] | None
        is_bootstrap = rung_index == final_rung_index
        if rung_index == 0:
            rung_messages = messages
            native_format = schema_model
        else:
            rung_messages = (
                bootstrapped if is_bootstrap else json_only_retry_messages(messages, schema_model)
            )
            native_format = None
        try:
            completion = await complete_fn(
                messages=rung_messages,
                model_def=model_def,
                max_tokens=max_tokens * rung_budget,
                temperature=temperature,
                response_format=native_format,
            )
            raw_text = completion_content(completion)
            if is_bootstrap:
                # The completion continues the prefilled envelope; parse the
                # concatenation, never the continuation alone.
                raw_text = bootstrap_prefix + raw_text
            return ladder.parse_content(raw_text)
        except ValidationError as exc:
            # JSON-shaped output that failed a schema bound is the shape a
            # better-aimed re-ask can fix: name the failing fields and the
            # constraint, then keep climbing.
            last_exc = exc
            if feedback_fired:
                continue
            feedback_fired = True
            try:
                feedback_completion = await complete_fn(
                    messages=validation_feedback_retry_messages(
                        messages,
                        schema_model,
                        validation_summary(exc),
                        note=ladder.validation_note,
                    ),
                    model_def=model_def,
                    max_tokens=max_tokens * multiplier,
                    temperature=temperature,
                    response_format=None,
                )
                return ladder.parse_content(completion_content(feedback_completion))
            except Exception as feedback_exc:  # pylint: disable=broad-exception-caught
                last_exc = feedback_exc
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc

    raise RuntimeError(f"{operation_label} distillation failed: {last_exc}") from last_exc


def validation_summary(exc: ValidationError) -> str:
    """One-line ``loc: type`` list, e.g. ``facts.0.fact: string_too_long``."""
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['type']}" for err in exc.errors()
    )
