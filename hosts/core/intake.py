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
from kernel.pack_activation import load_disabled_pack_names, normalize_pack_name

logger = logging.getLogger(__name__)

_PROCESSING_LOOP_GROUP = "lisa.extensions.processing_loops"
_DEFAULT_STATE_FILE = "state.json"
_DEFAULT_TIMEOUT_SECONDS = 30


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


async def run_intake(*, data_dir: Path) -> UnifiedState:
    """Run one intake tick: collect → process → merge → write.

    Args:
        data_dir: Directory where ``state.json`` is written. Also used to
            locate ``installed-packs.json`` for disabled-pack exclusion.

    Returns:
        The merged ``UnifiedState``.
    """
    logger.info("Intake tick starting")

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