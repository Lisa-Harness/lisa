"""Runtime-neutral OpenTelemetry span helpers."""

from __future__ import annotations

import asyncio
import logging

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

logger = logging.getLogger(__name__)

__all__ = [
    "PENDING_TELEMETRY_PUBLISH_TASKS",
    "get_tracer",
    "mark_span_failed",
    "schedule_telemetry_publish",
    "set_span_attributes",
]


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


#: Strong references to in-flight fire-and-forget telemetry publishes
#: (RUF006). Pruned on completion so the set cannot grow unboundedly.
PENDING_TELEMETRY_PUBLISH_TASKS: set[asyncio.Task[None]] = set()


def schedule_telemetry_publish(
    bus: object,
    topic: str,
    payload: object,
    *,
    source: str,
) -> None:
    """Schedule one fire-and-forget bus publish that never drops silently.

    The durable ledgers are evidence: a publish that fails — synchronously
    while creating the task, or asynchronously inside it — means a record
    the ledger will never receive, so both failure shapes log a WARNING
    naming the source. The previous per-site pattern (``create_task`` in a
    ``try`` with a DEBUG log) only saw the synchronous shape; an exception
    raised inside the task was never retrieved and vanished.

    Never raises: a scheduling failure on the telemetry path must not
    replace the caller's own result (the 2026-09-08 fetch_page incident).
    """
    publish = getattr(bus, "publish", None)
    if not callable(publish):
        logger.warning("telemetry publish skipped for %s: bus has no publish method", source)
        return
    try:
        task = asyncio.create_task(publish(topic, payload))
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "telemetry publish scheduling failed for %s on topic %s", source, topic, exc_info=True
        )
        return
    PENDING_TELEMETRY_PUBLISH_TASKS.add(task)

    def _on_done(done: asyncio.Task[None]) -> None:
        PENDING_TELEMETRY_PUBLISH_TASKS.discard(done)
        if done.cancelled():
            logger.warning("telemetry publish cancelled for %s on topic %s", source, topic)
            return
        exc = done.exception()
        if exc is not None:
            logger.warning(
                "telemetry publish failed for %s on topic %s: %s: %s",
                source,
                topic,
                type(exc).__name__,
                exc,
            )

    task.add_done_callback(_on_done)
