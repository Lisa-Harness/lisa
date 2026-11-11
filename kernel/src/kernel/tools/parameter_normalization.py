"""Schema-driven normalization of model-written tool and operation payloads.

Model-written JSON arrives at ``Parameters.model_validate`` shaped by the
provider's habits rather than the schema's contract: ``"null"``/``""`` for
optional ids, string-encoded arrays, in-type-but-out-of-bounds numbers.
Every one of those is an *encoding* of the right intent, and per-field
coercion validators absorbed them one sentinel at a time across packs
(~252 rejected dispatches in nine days). This module is the one normalization pass both registries
run before pydantic sees the payload.

Rules (schema-derived from ``model_fields``; top-level fields only —
nested-model rules are a named follow-up):

* R1 null-sentinel — a field whose annotation accepts ``None`` and whose
  default is ``None`` (genuinely optional) coerces the strings
  ``""``/``"null"``/``"None"`` to ``None``. Required fields are never
  coerced.
* R2 string-encoded arrays — a ``list``-annotated field holding a string
  is decoded JSON-first, then ``ast.literal_eval``, then two safe
  convention fallbacks (a single JSON object lifts to a one-element list;
  a single bare line wraps as a one-element list). A value that survives
  no decode is left untouched so pydantic rejects with its own message.
* R3 bounded numerics — an int/float value on a field carrying
  ``Field(ge=..., le=...)`` bounds clamps into range. Wrong-typed values
  are left for pydantic.
* Over-cap strings are never normalized: truncation would silently drop
  content (a memory record's claim, a digest), so the rejection path
  names the field, the cap, and the remedy instead — see
  :func:`describe_validation_failure`.

Opt a schema out with ``model_config = {"strict_parameters": True}``.
Every coercion logs one warning naming the target and the coerced fields,
and the registries stamp the same fields onto ``tool.telemetry`` rows via
``ToolExecutionTelemetry.normalized_fields`` — the warning contract is how
the incident family was diagnosed, and the telemetry makes provider
argument quality measurable per shape without reading logs.
"""

from __future__ import annotations

import ast
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, get_args, get_origin

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

logger = logging.getLogger(__name__)

__all__ = [
    "NormalizationResult",
    "describe_validation_failure",
    "normalize_payload",
]

#: String encodings of "absent" observed in the incident log.
NULL_SENTINELS = ("", "null", "None")

_STRICT_PARAMETERS_KEY = "strict_parameters"

_NoneType = type(None)


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """The rewritten payload and the fields the normalizer touched."""

    payload: dict[str, Any]
    normalized_fields: tuple[str, ...] = ()


def normalize_payload(
    schema: type[BaseModel],
    raw: Mapping[str, Any],
    *,
    target_id: str = "",
) -> NormalizationResult:
    """Return ``raw`` with schema-safe encoding fixes applied for ``schema``.

    The input mapping is never mutated; the returned payload is a fresh
    dict. ``target_id`` only labels the coercion warning. Schemas that set
    ``model_config = {"strict_parameters": True}`` pass through untouched.
    """

    payload = dict(raw)
    if not payload or schema.model_config.get(_STRICT_PARAMETERS_KEY) is True:
        return NormalizationResult(payload=payload)
    normalized: list[str] = []
    for name, field_info in schema.model_fields.items():
        if name not in payload:
            continue
        rewritten, changed = _normalize_value(field_info, payload[name])
        if not changed:
            continue
        payload[name] = rewritten
        normalized.append(name)
    if normalized:
        logger.warning(
            "normalized provider arguments for %s: %s",
            target_id or schema.__name__,
            ", ".join(sorted(normalized)),
        )
    return NormalizationResult(payload=payload, normalized_fields=tuple(normalized))


def describe_validation_failure(exc: ValidationError) -> str:
    """Rewrite a pydantic rejection as intent-naming sentences.

    The incident log's 19-repeats finding: a shape-only error message does
    not state the intent contract, so the model resends the identical
    payload. Each sentence names the field, the declared bound, and the
    remedy instead of pydantic's positional ``string_too_long``.
    """

    parts: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        if error.get("type") == "string_too_long":
            cap = (error.get("ctx") or {}).get("max_length")
            parts.append(
                f"'{loc}' is over its {cap}-character cap: "
                "condense the content and resend (do not repeat the same payload)"
            )
        else:
            parts.append(f"'{loc}': {error.get('msg', 'invalid value')}")
    return "; ".join(parts) if parts else str(exc)


def _normalize_value(field_info: Any, value: Any) -> tuple[Any, bool]:
    """Apply the schema-derived rules to one field value."""

    annotation = _unwrapped_annotation(field_info.annotation)
    if (
        _accepts_none(annotation)
        and field_info.default is None
        and field_info.default is not PydanticUndefined
        and isinstance(value, str)
        and value.strip() in NULL_SENTINELS
    ):
        return None, True
    base = _none_stripped(annotation)
    if get_origin(base) is list and isinstance(value, str):
        decoded = _decode_string_encoded_list(value)
        if decoded is not None:
            return decoded, True
    if base in (int, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return _clamp_to_bounds(field_info, value)
    return value, False


def _unwrapped_annotation(annotation: Any) -> Any:
    """Strip ``Annotated[...]`` wrappers, keeping the innermost type."""

    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _accepts_none(annotation: Any) -> bool:
    return _NoneType in get_args(annotation)


def _none_stripped(annotation: Any) -> Any:
    """Return the annotation with its ``None`` arm removed, if any."""

    args = [arg for arg in get_args(annotation) if arg is not _NoneType]
    if len(args) == 1:
        return args[0]
    return annotation


def _clamp_to_bounds(field_info: Any, value: Any) -> tuple[Any, bool]:
    ge = None
    le = None
    for meta in field_info.metadata:
        if ge is None:
            ge = getattr(meta, "ge", None)
        if le is None:
            le = getattr(meta, "le", None)
    if ge is None and le is None:
        return value, False
    clamped = value
    if ge is not None and clamped < ge:
        clamped = ge
    if le is not None and clamped > le:
        clamped = le
    return clamped, clamped != value


def _decode_string_encoded_list(value: str) -> list[Any] | None:  # noqa: PLR0911  # decode ladder branches per encoding: JSON, literal, object, bare line
    """Decode a string-encoded array, or ``None`` to leave it for pydantic.

    The ladder mirrors the pack's per-field decoder, generalized: strict
    JSON first, then Python literals (tuples included), then the two safe
    convention fallbacks — one JSON object lifts to a one-element list,
    and a single bare line wraps as a one-element list. Multi-line prose
    and undecodable text return ``None``: wrapping those would repair a
    genuinely ambiguous payload silently.
    """

    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    try:
        evaluated = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        evaluated = None
    if isinstance(evaluated, (list, tuple, set)):
        return list(evaluated)
    if isinstance(evaluated, dict):
        return [evaluated]
    if "\n" not in text and not text.startswith(("[", "{", "(", '"', "'")):
        return [text]
    return None
