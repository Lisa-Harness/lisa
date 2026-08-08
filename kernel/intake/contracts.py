"""Minimal bottom-half processing contracts.

The bottom half is a one-shot process that collects signals from installed
``SignalCollector`` extensions, runs installed ``ProcessingLoop`` extensions
on the results, and writes the merged ``UnifiedState`` to disk. The top half
(turn loop) reads that state via the host's ``_state_provider()``.

Three contracts only:

- ``SignalCollector`` (in ``connectors.py``): one passive feed. Stateless,
  idempotent, cheap to run.
- ``ProcessingLoop`` (here): one background processing chain. Triggered
  after collection, not by a clock.
- ``StateStore`` (host-owned, file or SQLite): where processed results land.
  The top half reads from here.

No long-running threads. The bottom-half process is triggered by a systemd
timer, cron, or ``lisa intake`` and exits when done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from kernel.models import AnomalyAlert, ContextEvidence, UnifiedState


@dataclass(slots=True)
class CollectionResult:
    """One collector's result from a single tick.

    ``snapshot`` is the typed Pydantic model returned by ``collect()`` (or
    a fallback snapshot on error/unconfigured). ``run`` is the execution
    metadata.
    """

    collector_id: str
    status: str = "ok"
    snapshot: Any = None
    run: Any = None


@dataclass(slots=True)
class ProcessContext:
    """Context provided to one ``ProcessingLoop.process()`` call.

    ``collections`` holds the results from all collectors that ran in this
    tick. ``state`` is the current ``UnifiedState`` (may be empty on the
    first tick). ``http_client`` and ``complete_fn`` are optional host
    services.
    """

    collections: list[CollectionResult] = field(default_factory=list)
    state: UnifiedState | None = None
    http_client: Any = None
    complete_fn: Any = None


@dataclass(slots=True)
class ProcessedState:
    """Output of one ``ProcessingLoop.process()`` call.

    The bottom-half tick merges all ``ProcessedState`` outputs into a single
    ``UnifiedState`` before writing to disk. ``context_evidence`` is the
    preferred contract: processing loops emit :class:`ContextEvidence`
    records directly. The legacy ``active_alerts``, ``signal_snapshots``,
    and ``intel_collections`` fields are still accepted and converted to
    ``ContextEvidence`` by the intake merge — they are a backward-compat
    bridge for loops that have not yet migrated.
    """

    loop_id: str = ""
    signal_snapshots: dict[str, dict[str, Any]] | None = None
    intel_collections: dict[str, Any] | None = None
    active_alerts: list[AnomalyAlert] = field(default_factory=list)
    context_evidence: list[ContextEvidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ProcessingLoop(Protocol):
    """One background processing chain triggered after signal collection.

    A processing loop takes the collection results from one tick and produces
    a ``ProcessedState`` that the bottom-half tick merges into ``UnifiedState``.

    Processing loops are discovered via the ``lisa.extensions.processing_loops``
    entry-point group. Each loop runs once per tick, sequentially, after all
    collectors have finished.

    Common uses:
    - Anomaly detection (examine snapshots, emit ``active_alerts``)
    - Trend analysis (examine snapshots, emit ``intel_collections``)
    - State derivation (examine snapshots, emit ``signal_snapshots``)
    """

    loop_id: str

    async def process(
        self,
        ctx: ProcessContext,
    ) -> ProcessedState: ...


__all__ = [
    "CollectionResult",
    "ProcessContext",
    "ProcessedState",
    "ProcessingLoop",
]
