"""Clarity-owned payload sanitizers built on simple rule-based redaction."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3})?[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}")
_CC_RE = re.compile(r"(?:\b\d[ -]*?){13,16}\b")


def redact_emails(text: str) -> str:
    return _EMAIL_RE.sub("[REDACTED_EMAIL]", text)


def redact_phones(text: str) -> str:
    return _PHONE_RE.sub("[REDACTED_PHONE]", text)


def redact_credit_cards(text: str) -> str:
    return _CC_RE.sub("[REDACTED_CC]", text)


def apply_redaction_pipeline(text: str, pipeline: list[Callable[[str], str]] | None = None) -> str:
    if text is None:
        return text
    active_pipeline = pipeline or [redact_emails, redact_phones, redact_credit_cards]
    out = text
    for fn in active_pipeline:
        out = fn(out)
    return out


def sanitize_clarity_record(record: dict[str, object]) -> dict[str, object]:
    """Shallow-sanitize one Clarity record by redacting common text fields."""

    out = dict(record)
    for key in ("message", "stack", "innerText", "text", "title", "url"):
        value = out.get(key)
        if isinstance(value, str):
            out[key] = apply_redaction_pipeline(value)

    node = out.get("node")
    if isinstance(node, dict):
        node_text = node.get("text")
        if isinstance(node_text, str):
            node["text"] = apply_redaction_pipeline(node_text)
            out["node"] = node

    return out


def sanitize_payload(obj: Any) -> Any:
    """Recursively sanitize strings inside a Clarity payload."""

    if obj is None:
        return obj
    if isinstance(obj, str):
        return apply_redaction_pipeline(obj)
    if isinstance(obj, dict):
        return {key: sanitize_payload(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_payload(value) for value in obj]
    return obj


__all__ = [
    "apply_redaction_pipeline",
    "sanitize_clarity_record",
    "sanitize_payload",
]
