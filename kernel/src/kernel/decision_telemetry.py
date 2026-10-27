"""Durable decision/dispatch telemetry appender.

The turn loop already publishes two decision-telemetry topics on the host's
event bus:

- ``turn.decision`` — one payload per iteration carrying the lifecycle action,
  its reason, the in-stage counter, and the write/read receipt counts. This is
  the record that answers "why did the turn do that?".
- ``tool.telemetry`` — one payload per dispatch carrying target id, kind,
  status, duration, and the error message.

Both were published and dropped: only ``lisa.token_usage`` had a durable
subscriber, so the bus recreated on every hook build carried them nowhere.
This module supplies the missing consumer, appending each payload as one JSONL
line so the ledger survives process restarts and can be joined to the thread
and the cost log by ``turn_index``.

Design notes
------------
- **Not stdlib logging.** The root file handler installs ``RateLimitFilter``,
  which suppresses repeated identical records keyed on
  ``(name, level, message)``. A mode that resolves the same lifecycle action on
  consecutive iterations would have those decisions silently dropped. A direct
  appender is the only way to keep every record.
- **Separate file.** ``decisions.jsonl`` rotates independently of
  ``runtime.jsonl``, so verbose runtime logging cannot evict the decision
  ledger, and the ledger cannot consume the log's rotation budget.
- **Failures are swallowed.** Telemetry must never break the turn loop; errors
  are logged and dropped, matching ``TokenUsageTelemetryHandler``.
- **Untyped payloads are fine.** The publishers send plain dicts
  (``turn.decision``) or serialized events (``tool.telemetry``), so the handler
  records whatever mapping it receives and stamps a provenance marker naming
  the topic. Only non-mapping payloads are ignored.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Event-bus topic carrying per-iteration lifecycle decisions.
TURN_DECISION_TOPIC = "turn.decision"

#: Event-bus topic carrying per-dispatch tool/operation telemetry.
TOOL_TELEMETRY_TOPIC = "tool.telemetry"

#: Field stamped onto every persisted record so a reader can tell which topic
#: produced a line without inferring it from its key set.
TOPIC_FIELD = "topic"

__all__ = [
    "TOOL_TELEMETRY_TOPIC",
    "TOPIC_FIELD",
    "TURN_DECISION_TOPIC",
    "DecisionTelemetryHandler",
]


class DecisionTelemetryHandler:
    """Append ``turn.decision`` and ``tool.telemetry`` payloads to one JSONL file.

    Hosts construct one instance with the resolved ``decisions.jsonl`` path and
    register :meth:`publish_turn_decision` / :meth:`publish_tool_telemetry` as
    event-bus subscribers. Both are ``async`` so they slot into the existing
    ``EventBus`` protocol, but the write itself is a synchronous single-line
    append.

    Args:
        path: Absolute path to the ``decisions.jsonl`` file. Parent
            directories are created lazily on first write.
        filter_fn: Optional predicate receiving ``(topic, record)`` and
            returning ``True`` to persist. Hosts use it to drop topics or
            redact fields before persistence.
    """

    def __init__(
        self,
        path: Path,
        *,
        filter_fn: Any | None = None,
    ) -> None:
        self._path = path
        self._filter_fn = filter_fn
        self._sequence = 0

    @property
    def path(self) -> Path:
        """Return the resolved JSONL path."""
        return self._path

    async def publish_turn_decision(self, payload: Any) -> None:
        """Persist one ``turn.decision`` payload."""
        self._persist(TURN_DECISION_TOPIC, payload)

    async def publish_tool_telemetry(self, payload: Any) -> None:
        """Persist one ``tool.telemetry`` payload."""
        self._persist(TOOL_TELEMETRY_TOPIC, payload)

    # ── Internals ──────────────────────────────────────────────────────

    def _persist(self, topic: str, payload: Any) -> None:
        record = _coerce_record(payload)
        if record is None:
            logger.debug(
                "decision telemetry dropped non-mapping payload on topic %s: %s",
                topic,
                type(payload).__name__,
            )
            return
        record[TOPIC_FIELD] = topic
        if self._filter_fn is not None:
            try:
                if not self._filter_fn(topic, record):
                    return
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("decision telemetry filter raised; persisting record: %s", exc)
        self._append(record)

    def _append(self, record: dict[str, Any]) -> None:
        # A per-process monotonic sequence keeps two same-timestamp records
        # distinguishable on a coarse Windows clock, mirroring the token
        # handler's nanosecond suffix. It is bookkeeping, not an idempotency
        # key: both topics are append-only observations, so duplicates are
        # evidence rather than a side effect to suppress.
        self._sequence += 1
        record.setdefault("logged_at", time.time())
        record.setdefault("sequence", self._sequence)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning("decision telemetry append failed for %s: %s", self._path, exc)

    def read_recent(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return the last ``limit`` persisted records as parsed dicts.

        Oldest first within the tail window. Decode errors are skipped. Returns
        an empty list when the file does not exist or is unreadable.
        """
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

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


def _coerce_record(payload: Any) -> dict[str, Any] | None:
    """Normalize one published payload into a JSON-serializable mapping.

    Publishers send either a plain dict (``turn.decision``) or a Pydantic
    event serialized with ``model_dump(mode="json")`` (``tool.telemetry``).
    Events arriving unserialized are dumped here so a direct in-process
    subscriber sees the same shape as the bus path. Anything that is not a
    mapping is rejected rather than wrapped, so a malformed payload cannot
    masquerade as evidence.
    """
    if isinstance(payload, dict):
        return dict(payload)
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return None
