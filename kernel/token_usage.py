"""Durable token-usage telemetry handler.

This module provides ``TokenUsageTelemetryHandler``, the canonical consumer
for ``TokenUsageEvent`` payloads. It appends every event as one JSONL line
to a host-supplied ``token-usage.jsonl`` file so token accounting survives
process restarts.

Design notes:
    - Runtime-neutral: the handler takes a ``Path`` and an optional event
      filter. It does not import app-layer config or Azure exporters.
    - Idempotency: the ``TokenUsageEvent.idempotency_key`` is set to a
      timestamp + nanosecond suffix when empty, so retries and stream-final
      events always append without collapsing (per the event docstring).
    - Atomic appends: each line is written with ``"a"`` mode under a
      ``mkdir(parents=True, exist_ok=True)`` so the file appears lazily.
    - Failures are swallowed: telemetry must never break the turn loop.
      Errors are logged through ``logging`` and dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kernel.models import TokenUsageEvent

logger = logging.getLogger(__name__)


class TokenUsageTelemetryHandler:
    """Append every ``TokenUsageEvent`` to a durable JSONL file.

    Hosts construct one instance with the resolved ``token-usage.jsonl``
    path and register it as an event-bus consumer for the ``token_used``
    event type. The handler is async-publishable (``async def publish``)
    so it slots into the existing ``EventBus`` protocol, but the file
    write itself is synchronous and fast (one ``open(..., "a")`` per
    event).

    Idempotency key:
        When the incoming event has an empty ``idempotency_key``, the
        handler stamps it with ``emitted_at`` plus a nanosecond suffix
        derived from ``time.time_ns()``. This guarantees uniqueness
        across retries and stream-final events without collapsing
        legitimate duplicate calls.

    Args:
        path: Absolute path to the ``token-usage.jsonl`` file. Parent
            directories are created lazily on first write.
        filter_fn: Optional predicate that receives the event and
            returns ``True`` to persist, ``False`` to skip. Hosts use
            this to drop events below a severity threshold or to
            redact sensitive fields before persistence.
    """

    def __init__(
        self,
        path: Path,
        *,
        filter_fn: Any | None = None,
    ) -> None:
        self._path = path
        self._filter_fn = filter_fn

    @property
    def path(self) -> Path:
        """Return the resolved JSONL path."""
        return self._path

    async def publish(self, event: Any) -> None:
        """Persist one ``TokenUsageEvent`` as a single JSONL line.

        Silently returns for non-``TokenUsageEvent`` payloads so the
        handler can be wired into a bus that fans out to many event
        types without per-type filtering at the call site.
        """
        if not isinstance(event, TokenUsageEvent):
            return
        if self._filter_fn is not None:
            try:
                if not self._filter_fn(event):
                    return
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("token-usage filter raised; persisting event: %s", exc)
        self._append(event)

    def _append(self, event: TokenUsageEvent) -> None:
        if not event.idempotency_key:
            event.idempotency_key = f"{event.emitted_at}:{_time_ns_suffix()}"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
        except OSError as exc:
            logger.warning("token-usage append failed for %s: %s", self._path, exc)

    def read_recent(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return the last ``limit`` persisted records as parsed dicts.

        Reads the JSONL file from the tail, parses each line, and returns a
        list of dicts (oldest first within the tail window). Decode errors
        are skipped silently. Returns an empty list when the file does not
        exist or is unreadable.
        """
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        import json

        tail = lines[-limit:] if limit > 0 else lines
        results: list[dict[str, Any]] = []
        for raw_line in tail:
            try:
                decoded = json.loads(raw_line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(decoded, dict):
                results.append(decoded)
        return results


def _time_ns_suffix() -> str:
    """Return a short nanosecond suffix for idempotency keys."""
    import time

    return str(time.time_ns() % 1_000_000_000)


def build_token_usage_event(
    *,
    provider: str,
    model_id: str,
    kind: str = "chat_completion",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    duration_ms: float = 0.0,
    latency_ms: int = 0,
    cost_usd: float = 0.0,
    time_cost_usd: float = 0.0,
    request_id: str = "",
    mode_id: str = "",
    stage_id: str = "",
    call_site: str = "",
    conversation_id: str = "",
    user_slug: str = "default",
    surface_id: str = "",
    extra: dict[str, Any] | None = None,
) -> TokenUsageEvent:
    """Build one ``TokenUsageEvent`` with normalized defaults.

    Hosts call this from each LLM call site (e.g. ``_complete_live``)
    so every provider call produces exactly one durable record. The
    ``total_tokens`` field falls out as ``prompt + completion`` when
    the caller does not supply it directly.

    The ``kind`` literal discriminates billing shape:
        - ``chat_completion`` / ``embedding`` / ``image_generation`` /
          ``stream_chunk``: token-based billing.
        - ``time_billed``: duration-based billing (e.g. Ollama local
          GPU time). ``duration_ms`` and ``time_cost_usd`` carry the
          billable signal; token counters may still be populated for
          observability even when cost is time-derived.
    """
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return TokenUsageEvent(
        provider=provider,
        model_id=model_id,
        kind=kind,  # type: ignore[arg-type]
        prompt_tokens=int(prompt_tokens or 0),
        completion_tokens=int(completion_tokens or 0),
        total_tokens=int(total_tokens or 0),
        cached_tokens=int(cached_tokens or 0),
        reasoning_tokens=int(reasoning_tokens or 0),
        duration_ms=float(duration_ms or 0.0),
        latency_ms=int(latency_ms or 0),
        cost_usd=float(cost_usd or 0.0),
        time_cost_usd=float(time_cost_usd or 0.0),
        request_id=str(request_id or ""),
        mode_id=str(mode_id or ""),
        stage_id=str(stage_id or ""),
        call_site=str(call_site or ""),
        conversation_id=str(conversation_id or ""),
        user_slug=str(user_slug or "default"),
        surface_id=str(surface_id or ""),
        extra=dict(extra) if extra else {},
    )
