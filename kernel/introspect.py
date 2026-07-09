"""Live Python-level introspection for Lisa's self-model.

This module gathers runtime telemetry from the host process: process memory,
asyncio task count, thread count, MCP pool health, token usage trends, and
recent log errors. It requires no LLM calls and runs entirely within the slice
budget.

The host owns the concrete resources (MCP pool, log paths, token usage file)
and injects them via the ``introspect_fn`` callback on ``TurnHooks``. This
module provides the pure ``gather_introspection`` function that the host's
callback delegates to.

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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from kernel.models import (
    IntrospectSnapshot,
    LogErrorEntry,
    McpConnectionStatus,
    McpServerHealth,
    TokenUsageSummary,
)

if TYPE_CHECKING:
    from kernel.tools.mcp_client import MCPToolPool

logger = logging.getLogger(__name__)

_TOKEN_USAGE_TAIL_LIMIT = 200
_LOG_TAIL_LIMIT = 100
_LOG_ERROR_LEVELS = frozenset({"error", "warning", "critical"})
_MCP_UNKNOWN = "unknown"


async def gather_introspection(
    *,
    mcp_pool: MCPToolPool | None = None,
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
    mcp_health = _gather_mcp_pool_health(mcp_pool)
    token_summary = _summarize_token_usage(token_usage_path)
    log_errors, log_tail_lines = _tail_log_errors(log_path)

    return IntrospectSnapshot(
        captured_at=captured_at,
        uptime_seconds=uptime_seconds,
        asyncio_tasks=asyncio_tasks,
        python_threads=python_threads,
        python_modules=python_modules,
        process_rss_mb=rss_mb,
        process_cpu_percent=cpu_percent,
        mcp_pool_health=mcp_health,
        token_usage_24h=token_summary,
        recent_log_errors=log_errors,
        logfile_tail_lines=log_tail_lines,
    )


# ── Private collectors ────────────────────────────────────────────────────


def _compute_uptime(process_start_time: datetime | None) -> float:
    """Return seconds since the process started, or 0 when unknown."""
    if process_start_time is None:
        return 0.0
    try:
        delta = datetime.now(UTC) - process_start_time
        return max(0.0, delta.total_seconds())
    except Exception:
        return 0.0


def _count_asyncio_tasks() -> int:
    """Return the number of active asyncio tasks in the running loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return 0
    try:
        return len(asyncio.all_tasks(loop))
    except Exception:
        return 0


def _gather_process_info() -> tuple[float, float]:
    """Return (rss_mb, cpu_percent) using psutil, or (0.0, 0.0) when unavailable."""
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return 0.0, 0.0
    except Exception:
        return 0.0, 0.0
    try:
        proc = psutil.Process()
        mem_info = proc.memory_info()
        rss_mb = mem_info.rss / (1024.0 * 1024.0) if mem_info else 0.0
        cpu_percent = proc.cpu_percent(interval=0.0)
        return round(rss_mb, 1), round(cpu_percent, 1)
    except Exception:
        return 0.0, 0.0


def _gather_mcp_pool_health(mcp_pool: MCPToolPool | None) -> dict[str, McpServerHealth]:
    """Build a per-server health dict from the MCP tool pool's introspection surface."""
    if mcp_pool is None:
        return {}
    health: dict[str, McpServerHealth] = {}
    try:
        warm_ids = set(mcp_pool.warm_server_ids())
        failed_ids = set(mcp_pool.warm_failed_ids())
    except Exception:
        return health

    for config in mcp_pool.configs:
        server_id = config.server_id
        if server_id in failed_ids:
            status: McpConnectionStatus = "failed"
        elif server_id in warm_ids:
            status = "warm"
        elif config.warm:
            status = "failed"
        else:
            status = "cold"
        health[server_id] = McpServerHealth(
            server_id=server_id,
            status=status,
            warm=config.warm,
            last_error="",
        )
    return health


def _summarize_token_usage(token_usage_path: Path | None) -> TokenUsageSummary | None:
    """Read the tail of token-usage.jsonl and produce a 24h summary."""
    if token_usage_path is None or not token_usage_path.exists():
        return None
    try:
        lines = token_usage_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    if not lines:
        return None

    import json

    tail = lines[-_TOKEN_USAGE_TAIL_LIMIT:]
    total_cost = 0.0
    total_tokens = 0
    model_totals: dict[str, int] = {}
    spike_cost = 0.0
    spike_at = ""
    event_count = 0

    for raw_line in tail:
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        event_count += 1
        cost = float(record.get("cost_usd") or 0.0)
        tokens = int(record.get("total_tokens") or 0)
        total_cost += cost
        total_tokens += tokens
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
        top_model=top_model,
        top_model_tokens=top_model_tokens,
        spike_at=spike_at,
        event_count=event_count,
    )


def _tail_log_errors(log_path: Path | None) -> tuple[list[LogErrorEntry], int]:
    """Read the tail of the turn log and extract error/warning entries.

    Returns ``(errors, total_lines_in_tail)``.
    """
    if log_path is None or not log_path.exists():
        return [], 0
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return [], 0
    if not lines:
        return [], 0

    import json

    tail = lines[-_LOG_TAIL_LIMIT:]
    errors: list[LogErrorEntry] = []
    for raw_line in tail:
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        level = str(record.get("level") or "").lower()
        if level not in _LOG_ERROR_LEVELS:
            continue
        errors.append(
            LogErrorEntry(
                ts=str(record.get("ts") or record.get("timestamp_utc") or ""),
                level=level,
                event=str(record.get("event") or ""),
                message=str(record.get("message") or record.get("payload", {}).get("error_message", "")),
                server=str(record.get("server") or record.get("payload", {}).get("server_id", "")),
            )
        )
    return errors, len(tail)