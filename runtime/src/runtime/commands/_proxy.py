"""``/proxy`` command: provider-proxy introspection readout.

Renders a per-proxy health table (status, protocol, providers, endpoint,
pack-declared live stats) and the provider routing map (which provider
gateways route through which proxy, and which go direct). Reads the live
:class:`~kernel.proxy_launch.pool.ProxyPool` on
``runtime.resources``; the dispatcher already awaits
``runtime.ensure_pool_ready()`` before invoking this handler.

Transport-neutral Markdown output. Mirrors :mod:`runtime.commands._mcp`
in structure: a pure summary renderer plus an async stats fetch for the
pack-declared counters. Dependency direction: ``_proxy`` → ``_simple``,
``kernel.*``, ``runtime.*``. No imports from other command submodules.
"""

from __future__ import annotations

import logging

from runtime.commands._lists import escape_markdown_cell
from runtime.commands._simple import handle_simple_command
from runtime.introspect import gather_proxy_pool_health
from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)

_STATS_VALUE_TRUNC = 40


async def handle_proxy_command(runtime: HostRuntime, stripped: str, message: str) -> str:
    """Handle ``/proxy`` (summary)."""
    response_text = await render_proxy_summary(runtime)
    return handle_simple_command(runtime, message, "proxy", response_text)


async def render_proxy_summary(runtime: HostRuntime) -> str:
    """Render per-proxy health plus the provider routing map as Markdown."""
    pool = runtime.resources.proxy_pool
    health = await gather_proxy_pool_health(pool)
    routes = dict(runtime.resources.proxy_routes) if runtime.resources else {}

    if not health and not routes:
        return "No provider proxies are configured."

    lines = ["## Provider Proxies", ""]
    if health:
        lines += [
            "| Proxy | Status | Protocol | Providers | Endpoint | Stats |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for entry in health:
            stats_cell = _render_stats_cell(entry.stats) if entry.status == "warm" else "—"
            providers = ", ".join(entry.providers) if entry.providers else "—"
            lines.append(
                f"| {escape_markdown_cell(entry.proxy_id)} "
                f"| {entry.status} | {entry.protocol} "
                f"| {escape_markdown_cell(providers)} "
                f"| {escape_markdown_cell(entry.base_url)} "
                f"| {escape_markdown_cell(stats_cell)} |"
            )
    else:
        lines.append("No proxy processes were compiled for this session.")

    lines += ["", "## Provider Routing", ""]
    if routes:
        lines += ["| Provider | Via |", "| --- | --- |"]
        for provider in sorted(routes):
            route = routes[provider]
            lines.append(
                f"| {escape_markdown_cell(provider)} "
                f"| {escape_markdown_cell(route.proxy_id)} @ "
                f"{escape_markdown_cell(route.base_url)} |"
            )
    else:
        lines.append(
            "_No provider is routed through a proxy — every provider client "
            "uses its direct endpoint._"
        )
    return "\n".join(lines)


def _render_stats_cell(stats: dict[str, object]) -> str:
    """Render the fetched stats fields as a compact ``k=v`` list."""
    if not stats:
        return "—"
    parts = [
        f"{key}={_format_stat_value(value)}"
        for key, value in stats.items()
    ]
    return ", ".join(parts)


def _format_stat_value(value: object) -> str:
    """Compact string for one stats value (floats trim to 2 decimals)."""
    text = str(value)
    if isinstance(value, float):
        text = f"{value:.2f}"
    if len(text) > _STATS_VALUE_TRUNC:
        text = text[: _STATS_VALUE_TRUNC - 1] + "…"
    return text