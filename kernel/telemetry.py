"""Runtime-neutral OpenTelemetry span helpers.

This module intentionally stays free of app-layer config and Azure exporter
bootstrap so kernel code can annotate spans without depending on app.*.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer


def get_tracer(name: str) -> Tracer:
    """Return one tracer from the global tracer provider."""
    return trace.get_tracer(name)


def set_span_attributes(span: Span, attributes: dict[str, object]) -> None:
    """Attach supported span attributes, coercing complex values to strings."""
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            span.set_attribute(key, value)
            continue
        span.set_attribute(key, str(value))


def mark_span_failed(span: Span, exc: BaseException) -> None:
    """Record one exception on the active span and mark it as failed."""
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR))
