"""Project the runtime's compiled MCP servers into an external client config.

The pure projection lives in :mod:`sdk.mcp_host.export`; this module supplies
what only the host knows:

- the spawnable command for an **in-process** config (in-process launch has no
  subprocess command, but the same server installs a console script, so an
  external client can spawn that);
- the resolved data directory to record on the result.

Nothing here starts, stops, or connects to a server: the export describes the
configuration the host already compiled, so it works before the pool starts
and after it closes.

Dependency direction: ``runtime`` -> ``sdk`` -> ``kernel``. The export lives in
the SDK so an external host embedding Lisa can produce the same document
without importing the runtime.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from kernel.tools.mcp_client import MCPClientConfig
from sdk.mcp_host.export import EXPORT_FORMAT_MCPSERVERS, MCPExport, export_mcp_servers

if TYPE_CHECKING:
    from runtime.runtime import HostRuntime

__all__ = ["export_runtime_mcp_servers", "resolve_in_process_command"]


def resolve_in_process_command(server_id: str, *, venv_root: Path | None = None) -> str | None:
    """Return the console-script path for a first-party server, when present.

    An in-process config runs inside the host's event loop and carries no
    command. The same distribution still installs a console script named after
    the server id (``lisa-memory``, ``lisa-storage``, …), which is exactly what
    an external client would spawn. Returns ``None`` when that script is not
    installed, so the caller reports a skip instead of exporting a command that
    does not exist.
    """
    from runtime.paths import venv_root as _venv_root

    root = venv_root if venv_root is not None else _venv_root()
    bin_dir = root / ("Scripts" if sys.platform == "win32" else "bin")
    candidate = bin_dir / server_id
    if sys.platform == "win32":
        candidate = candidate.with_suffix(".exe")
    return str(candidate) if candidate.is_file() else None


def _command_for(config: MCPClientConfig) -> str | None:
    """Resolve a spawnable command for one compiled config."""
    if config.command:
        return config.command
    if (config.runtime_kind or "") == "in_process":
        return resolve_in_process_command(config.server_id)
    return None


def export_runtime_mcp_servers(
    runtime: HostRuntime,
    *,
    server_ids: Sequence[str] | None = None,
    include_http: bool = False,
    include_internal: bool = False,
    format: str = EXPORT_FORMAT_MCPSERVERS,
) -> MCPExport:
    """Export the runtime's compiled MCP servers as a client config.

    Reads the compiled warm/cold configs from the runtime's resources — the
    same lists the pool would start — and projects them. Does not require the
    pool to be started, so an operator can get a paste-ready config without
    paying for server startup.

    Args:
        runtime: The host runtime whose configs to export.
        server_ids: Restrict to these servers. ``None`` exports all.
        include_http: Include HTTP endpoints (URL only, no credentials).
        include_internal: Include servers the host hides from its own model.
        format: Export format; only ``mcpServers`` is implemented.

    Returns:
        The :class:`~sdk.mcp_host.export.MCPExport` for the runtime.
    """
    configs = [
        *runtime.resources.mcp_warm_configs,
        *runtime.resources.mcp_cold_configs,
    ]
    if not configs:
        # No compiled configs yet (bootstrap did not run or produced none).
        # Falling back to the live pool would report only *started* servers and
        # silently omit warm-but-not-yet-connected ones; return an empty export
        # instead so the caller renders "nothing configured".
        return MCPExport(format=format, data_dir=str(runtime.paths.data_dir))

    return export_mcp_servers(
        configs,
        format=format,
        server_ids=server_ids,
        include_http=include_http,
        include_internal=include_internal,
        command_resolver=_command_for,
        data_dir=str(runtime.paths.data_dir),
    )
