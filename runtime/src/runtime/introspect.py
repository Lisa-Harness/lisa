"""Live Python-level introspection for Lisa's self-model.

This module gathers runtime telemetry from the host process: process memory,
asyncio task count, thread count, MCP pool health, token usage trends, and
recent log errors. It requires no LLM calls and runs entirely within the slice
budget.

The host owns the concrete resources (MCP pool, log paths, token usage file)
and injects them via the ``introspect_fn`` callback on ``TurnHooks``. This
module provides the pure ``gather_introspection`` function that the host's
callback delegates to.

This is a host-runtime implementation backed by host-owned resources, so it
lives in ``runtime`` rather than the kernel. The kernel only owns the
``IntrospectSnapshot`` contracts and the ``introspect_fn`` seam on
``OperationExecutionContext`` / ``TurnRunnerTurnConfig``.

Design — soft dependencies:
    ``psutil`` is optional. When unavailable, process-level fields (RSS, CPU,
    uptime) return zero and the snapshot is still valid — just less rich.
    This keeps ``psutil`` out of the hard dependency set.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernel.models import (
    IntrospectSnapshot,
    LogErrorEntry,
    McpConnectionStatus,
    McpServerHealth,
    OperationalHealth,
    ProxyHealthStatus,
    TokenUsageSummary,
)
from sdk.forensics import (
    filter_window,
    in_window,
    iter_jsonl_tail,
    parse_ts,
)

if TYPE_CHECKING:
    from kernel.proxy_launch.pool import ProxyPool
    from kernel.tools.mcp_client import MCPToolPool

logger = logging.getLogger(__name__)

_LOG_ERROR_LEVELS = frozenset({"error", "warning", "critical"})
_LOG_ERROR_LIMIT = 50
#: Both telemetry summaries are trailing windows, not "the last N lines".
#: The rotation-aware, byte-bounded read itself lives in :mod:`sdk.forensics`
#: so the pack layer can share it without importing ``runtime``.
_WINDOW_SECONDS = 24 * 60 * 60
#: Health classification uses a much shorter window than the evidence list.
#: ``_ERROR_WARN_THRESHOLD`` / ``_ERROR_CRITICAL_THRESHOLD`` were calibrated
#: against a 100-line tail (roughly minutes of activity). Feeding them a 24h
#: count reclassifies almost every session as ``critical``, and since the
#: self-model only injects its vibe when health is degraded or struggling,
#: that turns a per-turn ambient signal into permanent prompt overhead. The
#: list stays wide for evidence; the classifier stays narrow for "right now".
_HEALTH_ERROR_WINDOW_SECONDS = 15 * 60
_MCP_UNKNOWN = "unknown"
_TASKS_WARN_THRESHOLD = 120
_TASKS_CRITICAL_THRESHOLD = 250
_RSS_WARN_MB = 1500.0
_RSS_CRITICAL_MB = 3000.0
_ERROR_WARN_THRESHOLD = 3
_ERROR_CRITICAL_THRESHOLD = 8
#: Two or more failed MCP servers flips health straight to critical (a
#: single failure is "degraded" — any more and the session is unusable).
_CRITICAL_FAILED_MCP_SERVERS = 2


async def gather_introspection(
    *,
    mcp_pool: MCPToolPool | None = None,
    proxy_pool: ProxyPool | None = None,
    log_path: Path | None = None,
    token_usage_path: Path | None = None,
    process_start_time: datetime | None = None,
) -> IntrospectSnapshot:
    """Gather a live introspection snapshot from the running host process.

    All parameters are optional — each contributes what it can. When a
    resource is unavailable, the corresponding fields are left at their
    defaults (zero / empty). This function never raises; failures in
    individual collectors are logged and skipped.
    """
    captured_at = datetime.now(UTC).isoformat()

    uptime_seconds = _compute_uptime(process_start_time)
    asyncio_tasks = _count_asyncio_tasks()
    python_threads = threading.active_count()
    python_modules = len(sys.modules)
    rss_mb, cpu_percent = _gather_process_info()
    mcp_health = gather_mcp_pool_health(mcp_pool)
    proxy_health = await gather_proxy_pool_health(proxy_pool)
    token_summary = _summarize_token_usage(token_usage_path)
    log_errors, log_tail_lines = _tail_log_errors(log_path)
    failed_mcp_servers = sum(1 for health in mcp_health.values() if health.status == "failed")
    # Health is a "right now" judgement, so it uses the short window even
    # though the evidence list above stays wide. See _HEALTH_ERROR_WINDOW_SECONDS.
    recent_error_count = _count_recent_errors(
        log_errors, cutoff=datetime.now(UTC) - timedelta(seconds=_HEALTH_ERROR_WINDOW_SECONDS)
    )
    operational_health, operational_health_reasons = _classify_operational_health(
        asyncio_tasks=asyncio_tasks,
        process_rss_mb=rss_mb,
        failed_mcp_servers=failed_mcp_servers,
        recent_error_count=recent_error_count,
    )
    failed_proxies = sum(entry.status == "failed" for entry in proxy_health)
    if failed_proxies:
        operational_health_reasons.append(f"proxy failures: {failed_proxies}")
        if operational_health != "critical":
            operational_health = "degraded"

    return IntrospectSnapshot(
        captured_at=captured_at,
        uptime_seconds=uptime_seconds,
        asyncio_tasks=asyncio_tasks,
        python_threads=python_threads,
        python_modules=python_modules,
        process_rss_mb=rss_mb,
        process_cpu_percent=cpu_percent,
        mcp_pool_health=mcp_health,
        proxy_pool_health=proxy_health,
        token_usage_24h=token_summary,
        recent_log_errors=log_errors,
        logfile_tail_lines=log_tail_lines,
        failed_mcp_servers=failed_mcp_servers,
        recent_error_count=recent_error_count,
        operational_health=_normalize_operational_health(operational_health),
        operational_health_reasons=operational_health_reasons,
    )


# ── Private collectors ────────────────────────────────────────────────────


def _compute_uptime(process_start_time: datetime | None) -> float:
    """Return seconds since the process started, or 0 when unknown."""
    if process_start_time is None:
        return 0.0
    try:
        delta = datetime.now(UTC) - process_start_time
        return max(0.0, delta.total_seconds())
    except (TypeError, OverflowError):
        return 0.0


def _count_asyncio_tasks() -> int:
    """Return the number of active asyncio tasks in the running loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return 0
    try:
        return len(asyncio.all_tasks(loop))
    except RuntimeError:
        return 0


def _gather_process_info() -> tuple[float, float]:
    """Return (rss_mb, cpu_percent) using psutil, or (0.0, 0.0) when unavailable."""
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return 0.0, 0.0
    except (OSError, AttributeError):
        return 0.0, 0.0
    try:
        proc = psutil.Process()
        mem_info = proc.memory_info()
        rss_mb = mem_info.rss / (1024.0 * 1024.0) if mem_info else 0.0
        cpu_percent = proc.cpu_percent(interval=0.0)
        return round(rss_mb, 1), round(cpu_percent, 1)
    except (OSError, AttributeError, PermissionError):
        return 0.0, 0.0


def gather_mcp_pool_health(mcp_pool: MCPToolPool | None) -> dict[str, McpServerHealth]:
    """Build a per-server health dict from the MCP tool pool's introspection surface."""
    if mcp_pool is None:
        return {}
    health: dict[str, McpServerHealth] = {}
    try:
        warm_ids = set(mcp_pool.warm_server_ids())
        warm_failed_ids = set(mcp_pool.warm_failed_ids())
        cold_failed_ids = set(_safe_call(mcp_pool, "cold_failed_ids", set))
        permanently_failed_ids = set(_safe_call(mcp_pool, "permanently_failed_ids", set))
        restart_counts = _safe_call(mcp_pool, "restart_counts", dict)
        last_errors = _safe_call(mcp_pool, "last_errors", dict)
        tools_by_server = _safe_call(mcp_pool, "tools_by_server", dict)
    except (AttributeError, RuntimeError):
        return health

    for config in mcp_pool.configs:
        server_id = config.server_id
        if server_id in permanently_failed_ids or server_id in warm_failed_ids:
            status: McpConnectionStatus = "failed"
        elif server_id in cold_failed_ids:
            # A cold server that tried and failed to start is ``failed``,
            # not ``cold`` — ``cold`` means "not yet attempted."
            status = "failed"
        elif server_id in warm_ids:
            status = "warm"
        elif config.warm:
            # Warm but not connected and not in the failed set: treat as
            # failed (a warm server that is not up is down, not cold).
            status = "failed"
        else:
            status = "cold"
        server_tools = tools_by_server.get(server_id, [])
        tool_names = [spec.name for spec in server_tools]
        health[server_id] = McpServerHealth(
            server_id=server_id,
            status=status,
            warm=config.warm,
            transport=getattr(config, "runtime_kind", None) or _derive_transport(config),
            url=config.url or "",
            command=config.command or "",
            tool_count=len(tool_names),
            tool_names=tool_names,
            permanently_failed=server_id in permanently_failed_ids,
            restart_count=int(restart_counts.get(server_id, 0)),
            last_error=last_errors.get(server_id, ""),
        )
    return health


async def gather_proxy_pool_health(proxy_pool: ProxyPool | None) -> list[ProxyHealthStatus]:
    """Build a readiness and telemetry view of compiled provider proxies.

    Reads the pool's sync :meth:`ProxyPool.health` view, then best-effort
    fetches each warm proxy's pack-declared stats so the snapshot carries
    live counters (tokens saved, hit rate, ...) when the pack declares
    them. A pool that is None (no proxies configured) yields an empty list.
    HTTP/JSON failures appear in each entry's ``stats_error``.
    """
    if proxy_pool is None:
        return []
    views: list[ProxyHealthStatus] = []
    pool_views = proxy_pool.health()
    for view in pool_views:
        stats: dict[str, Any] = {}
        if view.status == "warm" and view.status_url_path:
            fetched = await proxy_pool.fetch_stats(view.proxy_id)
            if fetched is not None:
                stats = fetched
        views.append(
            ProxyHealthStatus(
                proxy_id=view.proxy_id,
                status=view.status,
                protocol=view.protocol,
                providers=list(view.providers),
                base_url=view.base_url,
                warm=view.warm,
                required=view.required,
                declared_stats_path=view.status_url_path or "",
                stats={k: v for k, v in stats.items() if isinstance(v, (str, int, float, bool))},
                last_error=view.last_error,
                stats_error=proxy_pool.stats_error(view.proxy_id),
            )
        )
    return views


def _safe_call(pool: MCPToolPool, attr: str, default_factory: Any) -> Any:
    """Call a pool accessor defensively, returning a default when absent.

    Keeps ``gather_mcp_pool_health`` working against a ``MagicMock`` or an
    older pool that does not yet expose the new accessors (``cold_failed_ids``,
    ``permanently_failed_ids``, ``restart_counts``, ``last_errors``,
    ``tools_by_server``). Real pools always have them after Phase 1.
    """
    method = getattr(pool, attr, None)
    if method is None or not callable(method):
        return default_factory()
    try:
        return method()
    except (AttributeError, RuntimeError):
        return default_factory()


def _derive_transport(config: Any) -> str:
    """Derive a transport label when ``runtime_kind`` was not carried through.

    Falls back to ``\"http\"`` when a URL is set, else ``\"stdio\"``. Only
    used for configs constructed before the compiler passed ``runtime_kind``
    through; the introspection surface prefers the truthful
    ``config.runtime_kind`` when present.
    """
    if getattr(config, "url", None):
        return "http"
    return "stdio"


def _summarize_token_usage(token_usage_path: Path | None) -> TokenUsageSummary | None:
    """Summarize ``token-usage.jsonl`` over the trailing 24 hours.

    Both the file selection (rotation set, byte budget) and the time filter
    are real: the summary previously described "the last 200 records" while
    claiming a 24h window, which silently reported yesterday's spend as
    today's on a quiet day and a partial hour as a full day on a busy one.
    Returns ``None`` only when there is no readable file at all; a file whose
    records are all older than the window returns a zeroed summary so the
    caller can tell "no spend in 24h" from "no telemetry".
    """
    records = list(iter_jsonl_tail(token_usage_path))
    if not records:
        return None

    windowed = filter_window(records, now=datetime.now(UTC), window_seconds=_WINDOW_SECONDS)
    total_cost = 0.0
    total_tokens = 0
    billed_tokens = 0
    cached_tokens = 0
    cache_creation_tokens = 0
    model_totals: dict[str, int] = {}
    spike_cost = 0.0
    spike_at = ""
    event_count = 0

    for record in windowed:
        event_count += 1
        cost = float(record.get("cost_usd") or 0.0)
        tokens = int(record.get("total_tokens") or 0)
        cached = int(record.get("cached_tokens") or 0)
        cache_created = int(record.get("cache_creation_tokens") or 0)
        total_cost += cost
        total_tokens += tokens
        # Billed view = provider-reported total minus cache-hit tokens,
        # which the provider meter bills at a much lower rate (or excludes).
        billed_tokens += max(0, tokens - cached)
        cached_tokens += cached
        cache_creation_tokens += cache_created
        model_id = str(record.get("model_id") or "")
        if model_id:
            model_totals[model_id] = model_totals.get(model_id, 0) + tokens
        if cost > spike_cost:
            spike_cost = cost
            spike_at = str(record.get("emitted_at") or "")

    top_model = ""
    top_model_tokens = 0
    if model_totals:
        top_model, top_model_tokens = max(model_totals.items(), key=lambda kv: kv[1])

    return TokenUsageSummary(
        total_cost_estimate=round(total_cost, 4),
        total_tokens=total_tokens,
        billed_tokens=billed_tokens,
        top_model=top_model,
        top_model_tokens=top_model_tokens,
        spike_at=spike_at,
        event_count=event_count,
        cached_tokens=cached_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )


def _tail_log_errors(log_path: Path | None) -> tuple[list[LogErrorEntry], int]:
    """Extract error/warning entries from the log's trailing 24h window.

    Returns ``(errors, records_read)``.

    Reads the rotation set, not just the active file: ``RotatingFileHandler``
    moves records into ``<name>.1``..``<name>.3``, so an active-file-only read
    silently loses older evidence in the window.

    The message comes from ``msg`` — the field :class:`sdk.logging_setup.
    JSONFormatter` actually writes. Reading only ``message`` (the key the
    retired bespoke writer used) produced an empty message for every error,
    which is how the self-model reported "recent errors" without saying what
    any of them were.
    """
    if log_path is None:
        return [], 0
    records = list(iter_jsonl_tail(log_path))
    if not records:
        return [], 0

    windowed = filter_window(records, now=datetime.now(UTC), window_seconds=_WINDOW_SECONDS)
    errors: list[LogErrorEntry] = []
    for record in windowed:
        level = str(record.get("level") or "").lower()
        if level not in _LOG_ERROR_LEVELS:
            continue
        payload = record.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        errors.append(
            LogErrorEntry(
                ts=str(
                    record.get("ts")
                    or record.get("timestamp_utc")
                    or record.get("emitted_at")
                    or ""
                ),
                level=level,
                event=str(record.get("event") or ""),
                message=str(
                    record.get("msg") or record.get("message") or payload.get("error_message") or ""
                ),
                server=str(record.get("server") or payload.get("server_id") or ""),
            )
        )
    # Keep chronological order (oldest first), matching the long-standing read
    # contract; when the window holds more than the cap, keep the most recent.
    return errors[-_LOG_ERROR_LIMIT:], len(records)


def _count_recent_errors(errors: list[LogErrorEntry], *, cutoff: datetime) -> int:
    """Count errors inside the health window.

    Separate from the evidence list on purpose: the list is a wide forensic
    window, the classifier is a narrow "is the process healthy right now"
    question. An entry without a parseable timestamp is counted — an error the
    log could not date is still an error, and excluding it would let a broken
    writer hide a live failure.
    """
    count = 0
    for entry in errors:
        if in_window(parse_ts(entry.ts), cutoff):
            count += 1
    return count


def _classify_operational_health(
    *,
    asyncio_tasks: int,
    process_rss_mb: float,
    failed_mcp_servers: int,
    recent_error_count: int,
) -> tuple[OperationalHealth, list[str]]:
    """Classify runtime health from a bounded set of deterministic telemetry signals."""
    reasons: list[str] = []
    status: OperationalHealth = "healthy"

    if failed_mcp_servers >= _CRITICAL_FAILED_MCP_SERVERS:
        status = "critical"
        reasons.append(f"mcp failures: {failed_mcp_servers}")
    elif failed_mcp_servers > 0:
        status = "degraded"
        reasons.append(f"mcp failures: {failed_mcp_servers}")

    if recent_error_count >= _ERROR_CRITICAL_THRESHOLD:
        status = "critical"
        reasons.append(f"recent errors: {recent_error_count}")
    elif recent_error_count >= _ERROR_WARN_THRESHOLD and status != "critical":
        status = "degraded"
        reasons.append(f"recent errors: {recent_error_count}")

    if asyncio_tasks >= _TASKS_CRITICAL_THRESHOLD:
        status = "critical"
        reasons.append(f"task pressure: {asyncio_tasks}")
    elif asyncio_tasks >= _TASKS_WARN_THRESHOLD and status != "critical":
        status = "degraded"
        reasons.append(f"task pressure: {asyncio_tasks}")

    if process_rss_mb >= _RSS_CRITICAL_MB:
        status = "critical"
        reasons.append(f"rss_mb: {process_rss_mb:.1f}")
    elif process_rss_mb >= _RSS_WARN_MB and status != "critical":
        status = "degraded"
        reasons.append(f"rss_mb: {process_rss_mb:.1f}")

    if not reasons and asyncio_tasks == 0 and process_rss_mb == 0.0 and failed_mcp_servers == 0:
        return "unknown", []

    return status, reasons


def _normalize_operational_health(value: str) -> OperationalHealth:
    if value == "healthy":
        return "healthy"
    if value == "degraded":
        return "degraded"
    if value == "critical":
        return "critical"
    return "unknown"
