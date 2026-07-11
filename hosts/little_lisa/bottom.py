"""Bottom-half one-shot process: collect signals, run processing loops, write state.

Triggered by ``little-lisa bottom --tick`` or a systemd timer / cron. The
process runs once and exits — no long-running threads.

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
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from kernel.intake.connector_discovery import discover_installed_signal_collectors
from kernel.intake.contracts import (
    CollectionResult,
    ProcessContext,
    ProcessingLoop,
    ProcessedState,
)
from kernel.models import UnifiedState

logger = logging.getLogger(__name__)

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

_PROCESSING_LOOP_GROUP = "lisa.extensions.processing_loops"
_DEFAULT_STATE_FILE = "state.json"
_DEFAULT_TIMEOUT_SECONDS = 30


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _discover_processing_loops() -> list[ProcessingLoop]:
    """Discover and instantiate installed processing loops via entry points."""
    loops: list[ProcessingLoop] = []
    try:
        eps = importlib.metadata.entry_points(group=_PROCESSING_LOOP_GROUP)
    except Exception:  # pylint: disable=broad-except
        logger.debug("No processing-loops entry-point group found", exc_info=True)
        return loops

    for ep in eps:
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


async def _collect_all() -> list[CollectionResult]:
    """Run all installed signal collectors in parallel."""
    collectors_map = discover_installed_signal_collectors()
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
    """Merge collection results and processed states into one UnifiedState."""
    compiled_at = _now_iso()
    state = UnifiedState(compiled_at=compiled_at)

    # Merge signal snapshots from collectors
    for result in collections:
        if result.snapshot is not None:
            state.signal_snapshots[result.collector_id] = result.snapshot

    # Merge processed states
    for ps in processed:
        if ps.signal_snapshots:
            state.signal_snapshots.update(ps.signal_snapshots)
        if ps.intel_collections:
            state.intel_collections.update(ps.intel_collections)
        if ps.active_alerts:
            state.active_alerts.extend(ps.active_alerts)

    return state


async def run_tick(*, data_dir: Path) -> UnifiedState:
    """Run one bottom-half tick: collect → process → merge → write.

    Args:
        data_dir: Directory where ``state.json`` is written.

    Returns:
        The merged ``UnifiedState``.
    """
    logger.info("Bottom-half tick starting")

    # 1. Collect signals
    collections = await _collect_all()
    logger.info("Collected %d signal(s)", len(collections))

    # 2. Run processing loops
    loops = _discover_processing_loops()
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


__all__ = ["run_tick"]