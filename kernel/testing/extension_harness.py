"""Host-neutral helpers for extension discovery and independent execution."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import EntryPoint, entry_points
from time import perf_counter
from typing import Any, Literal, cast

import httpx
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field, SerializeAsAny

from kernel.intake.sources import DataSource
from kernel.seams.protocols import DataSourceHooks
from kernel.models import SignalConnectorError, SignalConnectorRefreshResult, SignalConnectorRun
from kernel.intake.connectors import SignalConnector, SignalRunMode
from kernel.testing.stubs import make_data_source_hooks

_SIGNAL_CONNECTOR_GROUP = "lisa.extensions.signal_connectors"
_DATA_SOURCE_GROUP = "lisa.extensions.data_sources"

DataSourceRunMode = Literal["scheduled", "on_demand", "live_test"]


class ExtensionEntry(BaseModel):
    """Discoverable extension entry-point metadata."""

    entry_id: str
    kind: Literal["connector", "data_source"]
    import_path: str


class DataSourceRunResult(BaseModel):
    """Result envelope for one independent data-source execution."""

    source_id: str
    run_mode: str = "live_test"
    status: str = "ok"
    started_at: str
    finished_at: str
    duration_ms: int = 0
    missing_configuration: list[str] = Field(default_factory=list)
    error: SignalConnectorError | None = None
    snapshot: SerializeAsAny[BaseModel] | None = None


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


def discover_signal_connector_entry_points() -> dict[str, EntryPoint]:
    """Return discovered signal connector entry points keyed by entry ID."""
    return _entry_points_by_group(_SIGNAL_CONNECTOR_GROUP)


def discover_data_source_entry_points() -> dict[str, EntryPoint]:
    """Return discovered data-source entry points keyed by entry ID."""
    return _entry_points_by_group(_DATA_SOURCE_GROUP)


def list_extension_entries() -> list[ExtensionEntry]:
    """Return all extension entry points with normalized metadata."""
    entries: list[ExtensionEntry] = []
    for entry_id, ep in discover_signal_connector_entry_points().items():
        entries.append(
            ExtensionEntry(
                entry_id=entry_id,
                kind="connector",
                import_path=_entry_import_path(ep),
            )
        )
    for entry_id, ep in discover_data_source_entry_points().items():
        entries.append(
            ExtensionEntry(
                entry_id=entry_id,
                kind="data_source",
                import_path=_entry_import_path(ep),
            )
        )
    return sorted(entries, key=lambda item: (item.kind, item.entry_id))


def load_signal_connector(entry_id: str) -> SignalConnector[Any]:
    """Instantiate one signal connector from extension entry points."""
    ep = discover_signal_connector_entry_points().get(entry_id)
    if ep is None:
        raise KeyError(f"Unknown connector entry '{entry_id}'.")
    connector_cls = ep.load()
    connector = connector_cls()
    if not isinstance(connector, SignalConnector):
        raise TypeError(f"Entry '{entry_id}' did not load a SignalConnector instance.")
    return connector


def load_data_source(entry_id: str) -> DataSource[Any]:
    """Instantiate one data source from extension entry points."""
    ep = discover_data_source_entry_points().get(entry_id)
    if ep is None:
        raise KeyError(f"Unknown data source entry '{entry_id}'.")
    source_cls = ep.load()
    source = source_cls()
    if not isinstance(source, DataSource):
        raise TypeError(f"Entry '{entry_id}' did not load a DataSource instance.")
    return source


def build_connector_contract_preview(
    connector: SignalConnector[Any],
    *,
    collected_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic contract preview without making network calls."""
    snapshot_time = collected_at or _now_iso()
    unconfigured_snapshot = connector.build_unconfigured_snapshot(snapshot_time)
    error_snapshot = connector.build_error_snapshot(snapshot_time)
    return {
        "entry_kind": "connector",
        "connector_id": connector.connector_id,
        "display_name": connector.display_name,
        "missing_configuration": connector.missing_configuration(),
        "unconfigured_snapshot": unconfigured_snapshot.model_dump(mode="json"),
        "error_snapshot": error_snapshot.model_dump(mode="json"),
    }


def build_data_source_contract_preview(source: DataSource[Any]) -> dict[str, Any]:
    """Build a deterministic data-source contract preview without live calls."""
    preview_error = RuntimeError("contract_preview_error")
    unconfigured_snapshot = source.build_unconfigured_snapshot()
    error_snapshot = source.build_error_snapshot(preview_error)
    return {
        "entry_kind": "data_source",
        "source_id": source.source_id,
        "display_name": source.display_name,
        "missing_configuration": source.missing_configuration(),
        "unconfigured_snapshot": unconfigured_snapshot.model_dump(mode="json"),
        "error_snapshot": error_snapshot.model_dump(mode="json"),
    }


def _classify_error(exc: Exception) -> SignalConnectorError:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return SignalConnectorError(
            kind="http_status_error",
            message=str(exc),
            status_code=status_code,
            retryable=status_code == 429 or status_code >= 500,
        )
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return SignalConnectorError(
            kind=type(exc).__name__,
            message=str(exc),
            retryable=True,
        )
    return SignalConnectorError(
        kind=type(exc).__name__,
        message=str(exc),
        retryable=False,
    )


async def run_signal_connector_live(
    connector: SignalConnector[Any],
    *,
    run_mode: SignalRunMode = "live_test",
    client: httpx.AsyncClient | None = None,
) -> SignalConnectorRefreshResult:
    """Run one connector once and return a normalized refresh result."""
    started_at = _now_iso()
    timer_start = perf_counter()
    missing_configuration = connector.missing_configuration()
    if missing_configuration:
        finished_at = _now_iso()
        return SignalConnectorRefreshResult(
            run=SignalConnectorRun(
                connector_id=connector.connector_id,
                run_mode=run_mode,
                status="unconfigured",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=int((perf_counter() - timer_start) * 1000),
                attempts=0,
                error=SignalConnectorError(
                    kind="missing_configuration",
                    message=", ".join(missing_configuration),
                    retryable=False,
                ),
            ),
            snapshot=connector.build_unconfigured_snapshot(finished_at),
        )

    if client is None:
        async with httpx.AsyncClient() as managed_client:
            return await run_signal_connector_live(
                connector,
                run_mode=run_mode,
                client=managed_client,
            )

    attempt_timestamp = _now_iso()
    try:
        snapshot = await connector.collect(client, attempt_timestamp)
    except Exception as exc:  # pylint: disable=broad-except
        finished_at = _now_iso()
        error = _classify_error(exc)
        return SignalConnectorRefreshResult(
            run=SignalConnectorRun(
                connector_id=connector.connector_id,
                run_mode=run_mode,
                status="error",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=int((perf_counter() - timer_start) * 1000),
                attempts=1,
                error=error,
            ),
            snapshot=connector.build_error_snapshot(finished_at),
        )

    finished_at = _now_iso()
    return SignalConnectorRefreshResult(
        run=SignalConnectorRun(
            connector_id=connector.connector_id,
            run_mode=run_mode,
            status=connector.extract_status(snapshot),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=int((perf_counter() - timer_start) * 1000),
            attempts=1,
        ),
        snapshot=snapshot,
    )


async def run_data_source_live(
    source: DataSource[Any],
    *,
    run_mode: DataSourceRunMode = "live_test",
    hooks: DataSourceHooks | None = None,
    openai_client: AsyncAzureOpenAI | None = None,
    client: httpx.AsyncClient | None = None,
) -> DataSourceRunResult:
    """Run one data source once and return a normalized run result."""
    started_at = _now_iso()
    timer_start = perf_counter()
    missing_configuration = source.missing_configuration()
    if missing_configuration:
        finished_at = _now_iso()
        return DataSourceRunResult(
            source_id=source.source_id,
            run_mode=run_mode,
            status="unconfigured",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=int((perf_counter() - timer_start) * 1000),
            missing_configuration=missing_configuration,
            error=SignalConnectorError(
                kind="missing_configuration",
                message=", ".join(missing_configuration),
                retryable=False,
            ),
            snapshot=source.build_unconfigured_snapshot(),
        )

    resolved_hooks = hooks or cast(DataSourceHooks, make_data_source_hooks())

    if client is None:
        async with httpx.AsyncClient() as managed_client:
            return await run_data_source_live(
                source,
                run_mode=run_mode,
                hooks=resolved_hooks,
                openai_client=openai_client,
                client=managed_client,
            )

    try:
        snapshot = await source.execute_cycle(client, openai_client, hooks=resolved_hooks)
    except Exception as exc:  # pylint: disable=broad-except
        finished_at = _now_iso()
        error = _classify_error(exc)
        return DataSourceRunResult(
            source_id=source.source_id,
            run_mode=run_mode,
            status="error",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=int((perf_counter() - timer_start) * 1000),
            error=error,
            snapshot=source.build_error_snapshot(exc),
        )

    finished_at = _now_iso()
    return DataSourceRunResult(
        source_id=source.source_id,
        run_mode=run_mode,
        status=source.extract_status(snapshot),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=int((perf_counter() - timer_start) * 1000),
        snapshot=snapshot,
    )
