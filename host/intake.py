"""Intake tick — one-shot signal collection, processing, and state write.

This is the host-side intake process. It collects signals from installed
``SignalCollector`` extensions, runs installed ``ProcessingLoop`` extensions
on the results, and writes the merged ``UnifiedState`` to disk. The next
turn reads that state via the host's state provider so the model has fresh
ambient data (GitHub PR status, Stripe billing health, Cloudflare zone
state, etc.).

Triggered by ``lisa intake`` or a systemd timer / cron. The process runs
once and exits — no long-running threads.

Flow:

    1. Discover installed signal collectors via entry points.
    2. Run ``collect()`` on each (parallel, bounded by ``asyncio.gather``).
    3. Discover installed processing loops via entry points.
    4. Run each ``ProcessingLoop.process()`` on the collection results.
    5. Merge results into ``UnifiedState``.
    6. Write to ``<data_dir>/state.json``.
    7. Exit.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from kernel.intake.connector_discovery import discover_installed_signal_collectors
from kernel.intake.contracts import (
    CollectionResult,
    ProcessContext,
    ProcessedState,
    ProcessingLoop,
)
from kernel.models import ContextEvidence, UnifiedState
from kernel.pack_activation import load_disabled_pack_names, normalize_pack_name

from host.content_roots import seed_bundled_packs_lock

logger = logging.getLogger(__name__)

_PROCESSING_LOOP_GROUP = "lisa.extensions.processing_loops"
_DEFAULT_STATE_FILE = "state.json"
_DEFAULT_TIMEOUT_SECONDS = 30

_ALERT_SEVERITY_SALIENCE = {"critical": 1.0, "warn": 0.85, "info": 0.65}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _evidence_from_alert(
    *,
    alert: Any,
    collected_at: str,
) -> ContextEvidence | None:
    """Convert one :class:`AnomalyAlert` (or dict) into a :class:`ContextEvidence`."""
    if isinstance(alert, dict):
        severity = str(alert.get("severity", "info")).strip().lower() or "info"
        source = str(alert.get("source", "")).strip() or "system"
        message = str(alert.get("message", "")).strip()
        detected_at = str(alert.get("detected_at", "")) or collected_at
    else:
        severity = str(getattr(alert, "severity", "info")).strip().lower() or "info"
        source = str(getattr(alert, "source", "")).strip() or "system"
        message = str(getattr(alert, "message", "")).strip()
        detected_at = str(getattr(alert, "detected_at", "")) or collected_at
    if not message:
        return None
    return ContextEvidence(
        evidence_id=f"state:alert:{source}:{detected_at}:{hash(message) & 0xFFFFFF:x}",
        namespace="alerts",
        source=source,
        source_type="intake",
        evidence_type="alert",
        title=f"{severity.title()} alert from {source}",
        body=message,
        collected_at=detected_at,
        salience_score=_ALERT_SEVERITY_SALIENCE.get(severity, 0.5),
        confidence=0.95,
        tags=["alert", severity, source],
        metadata={"provenance": "system_observed", "severity": severity},
    )


def _evidence_from_intel(
    *,
    intel_key: str,
    intel: Any,
    collected_at: str,
) -> ContextEvidence | None:
    """Convert one intel collection (or dict) into a :class:`ContextEvidence`."""
    if isinstance(intel, dict):
        generated_at = str(intel.get("generated_at", "")) or collected_at
        summary_bullets = list(intel.get("summary_bullets", []))
        source = str(intel.get("source", intel_key)).strip() or intel_key
    else:
        generated_at = str(getattr(intel, "generated_at", "")) or collected_at
        summary_bullets = list(getattr(intel, "summary_bullets", []))
        source = intel_key
    body = "\n".join(f"- {b.strip()}" for b in summary_bullets if b.strip())
    if not body:
        return None
    return ContextEvidence(
        evidence_id=f"state:intel:{intel_key}",
        namespace="intel",
        source=source,
        source_type="intake",
        evidence_type="intel_summary",
        title=f"Intel: {intel_key}",
        body=body,
        collected_at=generated_at,
        salience_score=0.55,
        confidence=0.8,
        tags=["intel", intel_key],
        metadata={"provenance": "authoritative"},
    )


def _evidence_from_signal(
    *,
    collector_id: str,
    status: str,
    snapshot: Any,
    collected_at: str,
) -> ContextEvidence | None:
    """Convert one signal snapshot (dict) into a :class:`ContextEvidence`."""
    if not isinstance(snapshot, dict):
        body = str(snapshot)
    else:
        # Render the snapshot as ``key=value`` pairs, skipping the reserved
        # metadata keys the collector framework writes.
        skip = {"status", "collected_at", "generated_at", "updated_at"}
        parts = [f"status={snapshot.get('status', status)}"]
        for key, value in sorted(snapshot.items()):
            if key in skip:
                continue
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}={value}")
        body = "\n".join(parts)
    if not body:
        return None
    return ContextEvidence(
        evidence_id=f"state:signal:{collector_id}",
        namespace="signals",
        source=collector_id,
        source_type="intake",
        evidence_type="signal_snapshot",
        title=f"Signal: {collector_id} ({status})",
        body=body,
        collected_at=collected_at,
        salience_score=0.5,
        confidence=0.7,
        tags=["signal", collector_id, status],
        metadata={"provenance": "system_observed"},
    )


def _entry_point_pack_name(ep: importlib.metadata.EntryPoint) -> str:
    """Return the distribution package name for an entry point, or empty string."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    name = dist.metadata.get("Name", "") if dist.metadata else ""
    return normalize_pack_name(str(name or ""))


def _discover_processing_loops(
    excluded_packs: set[str] | None = None,
) -> list[ProcessingLoop]:
    """Discover and instantiate installed processing loops via entry points.

    Args:
        excluded_packs: Normalized pack names to exclude from discovery.
            When ``None`` (default), no filtering is applied.
    """
    loops: list[ProcessingLoop] = []
    try:
        eps = importlib.metadata.entry_points(group=_PROCESSING_LOOP_GROUP)
    except Exception:  # pylint: disable=broad-except
        logger.debug("No processing-loops entry-point group found", exc_info=True)
        return loops

    for ep in eps:
        if excluded_packs and _entry_point_pack_name(ep) in excluded_packs:
            continue
        try:
            loop_cls = ep.load()
            loop = loop_cls()
            if hasattr(loop, "process") and hasattr(loop, "loop_id"):
                loops.append(loop)
            else:
                logger.warning(
                    "Processing loop '%s' does not satisfy ProcessingLoop protocol",
                    ep.name,
                )
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to load processing loop %r", ep.name, exc_info=True)

    return loops


async def _collect_one(
    collector_id: str,
    entry_point: Any,
    client: httpx.AsyncClient,
) -> CollectionResult:
    """Run one signal collector and return its result."""
    try:
        collector = entry_point.load()()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to instantiate collector %r: %s", collector_id, exc)
        return CollectionResult(
            collector_id=collector_id,
            status="error",
            run={"error": str(exc)},
        )

    missing = collector.missing_configuration()
    if missing:
        logger.info("Collector %r is unconfigured (missing: %s)", collector_id, missing)
        snapshot = collector.build_unconfigured_snapshot(_now_iso())
        return CollectionResult(
            collector_id=collector_id,
            status="unconfigured",
            snapshot=snapshot.model_dump(mode="json") if hasattr(snapshot, "model_dump") else None,
            run={"missing": missing},
        )

    try:
        snapshot = await asyncio.wait_for(
            collector.collect(client, _now_iso()),
            timeout=_DEFAULT_TIMEOUT_SECONDS,
        )
        status = collector.extract_status(snapshot)
        return CollectionResult(
            collector_id=collector_id,
            status=status,
            snapshot=snapshot.model_dump(mode="json") if hasattr(snapshot, "model_dump") else None,
            run={"status": status},
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Collector %r failed: %s", collector_id, exc)
        snapshot = collector.build_error_snapshot(_now_iso())
        return CollectionResult(
            collector_id=collector_id,
            status="error",
            snapshot=snapshot.model_dump(mode="json") if hasattr(snapshot, "model_dump") else None,
            run={"error": str(exc)},
        )


async def _collect_all(
    excluded_packs: set[str] | None = None,
) -> list[CollectionResult]:
    """Run all installed signal collectors in parallel.

    Args:
        excluded_packs: Normalized pack names to exclude from collection.
    """
    collectors_map = discover_installed_signal_collectors(excluded_packs=excluded_packs)
    if not collectors_map:
        logger.info("No signal collectors installed")
        return []

    entry_points = importlib.metadata.entry_points(group="lisa.extensions.signal_connectors")
    ep_map = {ep.name: ep for ep in entry_points}

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS) as client:
        tasks = [
            _collect_one(collector_id, ep_map[collector_id], client)
            for collector_id in collectors_map
            if collector_id in ep_map
        ]
        return await asyncio.gather(*tasks) if tasks else []


def _merge_results(
    collections: list[CollectionResult],
    processed: list[ProcessedState],
) -> UnifiedState:
    """Merge collection results and processed states into one UnifiedState.

    All ambient context — signals, intel, alerts, and any processed-state
    evidence — is converted to :class:`ContextEvidence` records and merged
    into ``context_evidence_catalog``. The kernel never sees typed
    alert/intel models; the single catalog is the only feed the context
    mapper walks.
    """
    compiled_at = _now_iso()
    state = UnifiedState(compiled_at=compiled_at)
    evidence: list[Any] = []

    # Convert raw signal snapshots from collectors into context evidence.
    for result in collections:
        if result.snapshot is not None:
            evidence.append(
                _evidence_from_signal(
                    collector_id=result.collector_id,
                    status=result.status,
                    snapshot=result.snapshot,
                    collected_at=compiled_at,
                )
            )

    # Merge processed states: convert their alerts/intel/signals to evidence,
    # and pass through any context_evidence emitted directly by the loop.
    for ps in processed:
        evidence.extend(_evidence_from_processed_state(ps, compiled_at))

    state.context_evidence_catalog = [e for e in evidence if e is not None]
    return state


def _evidence_from_processed_state(
    ps: ProcessedState,
    compiled_at: str,
) -> list[ContextEvidence | None]:
    """Convert one ProcessedState's fields into ContextEvidence records."""
    out: list[ContextEvidence | None] = []
    if ps.context_evidence:
        out.extend(ps.context_evidence)
    if ps.signal_snapshots:
        for connector_id, snapshot in ps.signal_snapshots.items():
            status = (
                str(snapshot.get("status", "ok"))
                if isinstance(snapshot, dict)
                else "ok"
            )
            out.append(
                _evidence_from_signal(
                    collector_id=connector_id,
                    status=status,
                    snapshot=snapshot,
                    collected_at=compiled_at,
                )
            )
    if ps.intel_collections:
        for intel_key, intel in ps.intel_collections.items():
            out.append(
                _evidence_from_intel(
                    intel_key=intel_key,
                    intel=intel,
                    collected_at=compiled_at,
                )
            )
    if ps.active_alerts:
        for alert in ps.active_alerts:
            out.append(
                _evidence_from_alert(
                    alert=alert,
                    collected_at=compiled_at,
                )
            )
    return out


async def run_intake(*, data_dir: Path) -> UnifiedState:
    """Run one intake tick: collect → process → merge → write.

    Args:
        data_dir: Directory where ``state.json`` is written. Also used to
            locate ``installed-packs.json`` for disabled-pack exclusion.

    Returns:
        The merged ``UnifiedState``.
    """
    logger.info("Intake tick starting")

    # Seed bundled packs into the lock so disabled-pack exclusion sees the
    # complete inventory. Intake runs as a one-shot process outside the
    # host's setup_framework, so it must seed independently.
    seed_bundled_packs_lock(data_dir)
    excluded_packs = load_disabled_pack_names(data_dir)

    # 1. Collect signals
    collections = await _collect_all(excluded_packs=excluded_packs)
    logger.info("Collected %d signal(s)", len(collections))

    # 2. Run processing loops
    loops = _discover_processing_loops(excluded_packs=excluded_packs)
    processed: list[ProcessedState] = []
    for loop in loops:
        ctx = ProcessContext(collections=collections)
        try:
            result = await asyncio.wait_for(
                loop.process(ctx),
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
            processed.append(result)
            logger.info("Processing loop %r completed", loop.loop_id)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Processing loop %r failed: %s", getattr(loop, "loop_id", "?"), exc)

    # 3. Merge into UnifiedState
    state = _merge_results(collections, processed)

    # 4. Write to disk
    state_path = data_dir / _DEFAULT_STATE_FILE
    data_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    logger.info("State written to %s", state_path)

    return state


__all__ = ["run_intake"]
