"""Host-neutral helpers for extension discovery and independent execution."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import EntryPoint, entry_points
from time import perf_counter
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from kernel.intake.connectors import SignalCollector, SignalRunMode
from kernel.models import (
    SignalCollectorError,
    SignalCollectorRefreshResult,
    SignalCollectorRun,
)

_SIGNAL_COLLECTOR_GROUP = "lisa.extensions.signal_connectors"


class ExtensionEntry(BaseModel):
    """Discoverable extension entry-point metadata."""

    entry_id: str
    kind: Literal["collector"] = "collector"
    import_path: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _entry_points_by_group(group: str) -> dict[str, EntryPoint]:
    return {ep.name: ep for ep in entry_points(group=group)}


def _entry_import_path(ep: EntryPoint) -> str:
    if ep.value:
        return ep.value
    if ep.attr:
        return f"{ep.module}:{ep.attr}"
    return ep.module


def discover_signal_collector_entry_points() -> dict[str, EntryPoint]:
    """Return discovered signal collector entry points keyed by entry ID."""
    return _entry_points_by_group(_SIGNAL_COLLECTOR_GROUP)


def list_extension_entries() -> list[ExtensionEntry]:
    """Return all extension entry points with normalized metadata."""
    entries: list[ExtensionEntry] = []
    for entry_id, ep in discover_signal_collector_entry_points().items():
        entries.append(
            ExtensionEntry(
                entry_id=entry_id,
                kind="collector",
                import_path=_entry_import_path(ep),
            )
        )
    return sorted(entries, key=lambda item: (item.kind, item.entry_id))


def load_signal_collector(entry_id: str) -> SignalCollector[Any]:
    """Instantiate one signal collector from extension entry points."""
    ep = discover_signal_collector_entry_points().get(entry_id)
    if ep is None:
        raise KeyError(f"Unknown collector entry '{entry_id}'.")
    collector_cls = ep.load()
    collector = collector_cls()
    if not isinstance(collector, SignalCollector):
        raise TypeError(f"Entry '{entry_id}' did not load a SignalCollector instance.")
    return collector


def build_collector_contract_preview(
    collector: SignalCollector[Any],
    *,
    collected_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic contract preview without making network calls."""
    snapshot_time = collected_at or _now_iso()
    unconfigured_snapshot = collector.build_unconfigured_snapshot(snapshot_time)
    error_snapshot = collector.build_error_snapshot(snapshot_time)
    return {
        "entry_kind": "collector",
        "collector_id": collector.collector_id,
        "display_name": collector.display_name,
        "missing_configuration": collector.missing_configuration(),
        "unconfigured_snapshot": unconfigured_snapshot.model_dump(mode="json"),
        "error_snapshot": error_snapshot.model_dump(mode="json"),
    }


def _classify_error(exc: Exception) -> SignalCollectorError:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return SignalCollectorError(
            kind="http_status_error",
            message=str(exc),
            status_code=status_code,
            retryable=status_code == 429 or status_code >= 500,
        )
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return SignalCollectorError(
            kind=type(exc).__name__,
            message=str(exc),
            retryable=True,
        )
    return SignalCollectorError(
        kind=type(exc).__name__,
        message=str(exc),
        retryable=False,
    )


async def run_signal_collector_live(
    collector: SignalCollector[Any],
    *,
    run_mode: SignalRunMode = "live_test",
    client: httpx.AsyncClient | None = None,
) -> SignalCollectorRefreshResult:
    """Run one collector once and return a normalized refresh result."""
    started_at = _now_iso()
    timer_start = perf_counter()
    missing_configuration = collector.missing_configuration()
    if missing_configuration:
        finished_at = _now_iso()
        return SignalCollectorRefreshResult(
            run=SignalCollectorRun(
                collector_id=collector.collector_id,
                run_mode=run_mode,
                status="unconfigured",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=int((perf_counter() - timer_start) * 1000),
                attempts=0,
                error=SignalCollectorError(
                    kind="missing_configuration",
                    message=", ".join(missing_configuration),
                    retryable=False,
                ),
            ),
            snapshot=collector.build_unconfigured_snapshot(finished_at),
        )

    if client is None:
        async with httpx.AsyncClient() as managed_client:
            return await run_signal_collector_live(
                collector,
                run_mode=run_mode,
                client=managed_client,
            )

    attempt_timestamp = _now_iso()
    try:
        snapshot = await collector.collect(client, attempt_timestamp)
    except Exception as exc:  # pylint: disable=broad-except
        finished_at = _now_iso()
        error = _classify_error(exc)
        return SignalCollectorRefreshResult(
            run=SignalCollectorRun(
                collector_id=collector.collector_id,
                run_mode=run_mode,
                status="error",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=int((perf_counter() - timer_start) * 1000),
                attempts=1,
                error=error,
            ),
            snapshot=collector.build_error_snapshot(finished_at),
        )

    finished_at = _now_iso()
    return SignalCollectorRefreshResult(
        run=SignalCollectorRun(
            collector_id=collector.collector_id,
            run_mode=run_mode,
            status=collector.extract_status(snapshot),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=int((perf_counter() - timer_start) * 1000),
            attempts=1,
        ),
        snapshot=snapshot,
    )


__all__ = [
    "ExtensionEntry",
    "build_collector_contract_preview",
    "discover_signal_collector_entry_points",
    "list_extension_entries",
    "load_signal_collector",
    "run_signal_collector_live",
]
