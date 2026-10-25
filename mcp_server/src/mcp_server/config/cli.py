"""Shared CLI argument parser for all first-party MCP servers.

Every first-party server calls :func:`build_parser` in ``main()`` and adds
server-specific args as needed. The contract is:

- ``--data-dir`` is **required** and is the canonical data root. Servers
  derive all storage paths relative to it. No env-var fallback, no
  hardcoded default.
- ``--transport`` defaults to ``stdio`` (only stdio is supported in v1).
- ``--log-level`` defaults to the ``LOG_LEVEL`` env var or ``INFO``.
- ``--workspace-root``, ``--config``, and ``--server-id`` are optional.
- ``--readonly`` is added only when the server supports it.

Env vars are for behavioral configuration only (API keys, feature flags,
endpoints, log formatting). They must not be used for path resolution.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

__all__ = ["build_parser", "configure_logging", "resolve_data_dir", "run"]


def build_parser(server_name: str, *, supports_readonly: bool = False) -> argparse.ArgumentParser:
    """Build the standard argument parser for one MCP server.

    Every first-party server calls this in ``main()`` and adds server-specific
    args as needed.
    """
    parser = argparse.ArgumentParser(
        prog=server_name,
        description=f"{server_name} — Lisa Harness MCP server",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help="Canonical data root. All storage paths are resolved relative to this.",
    )
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=None,
        help="Logical workspace root for relative path resolution.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON/YAML server config file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Defaults to LOG_LEVEL env var or INFO.",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio"],
        help="MCP transport (only stdio is supported in v1).",
    )
    parser.add_argument(
        "--server-id",
        type=str,
        default=None,
        help="Explicit server identity override for diagnostics.",
    )
    if supports_readonly:
        parser.add_argument(
            "--readonly",
            action="store_true",
            default=False,
            help="Disable write-side operations.",
        )
    return parser


def resolve_data_dir(cli_value: str) -> Path:
    """Resolve the data directory from a CLI arg.

    No env-var fallback, no hardcoded default. The host is responsible
    for providing a sensible default; the server just uses what it gets.
    """
    path = Path(cli_value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(level: str, *, server_name: str) -> logging.Logger:
    """Configure standard logging for one MCP server.

    Routes through the shared :func:`sdk.logging_setup.configure_logging`
    so MCP servers emit the same JSON format as the hosts. MCP servers run
    as separate stdio subprocesses; their log file lives under the
    server's ``--data-dir`` (via ``LISA_MCP_LOG_PATH``, injected by the
    host at launch) and stderr mirroring follows the
    ``LISA_LOG_STDERR`` env var (default off for subprocesses).

    When ``LISA_MCP_LOG_PATH`` is unset (e.g. a server launched outside the
    host pool, or a third-party server that cleared its env), the
    server falls back to a per-server JSONL file at
    ``<cwd>/<server_name>.jsonl`` — never to bare stderr. The MCP Python
    SDK's ``StdioServerParameters`` has no ``stderr=`` field in this
    version, so a ``basicConfig``-on-stderr default would inherit into
    the host terminal and corrupt the Rich TUI on Linux. Set
    ``LISA_LOG_STDERR=1`` to force the stderr mirror on for debugging.
    """
    from sdk.logging_setup import (
        LISA_LOG_DEBUG_ENV,
        _parse_debug_modules,
    )
    from sdk.logging_setup import (
        configure_logging as _configure_shared,
    )

    log_path = os.getenv("LISA_MCP_LOG_PATH", "")
    if not log_path:
        # Fall back to a per-server JSONL file in the current working
        # directory (the host sets cwd to the data dir for first-party
        # servers). Avoids the inherited-stderr leak. This path is only
        # reached when the server is launched outside the host's compiler
        # (which always sets ``LISA_MCP_LOG_PATH`` under the resolved
        # data dir, treating an empty value as unset since 2026-08-25).
        log_path = str(Path.cwd() / f"{server_name}.jsonl")
    _configure_shared(
        log_path=Path(log_path),
        level=level,
        debug_modules=_parse_debug_modules(os.getenv(LISA_LOG_DEBUG_ENV, "")),
        host_kind=None,
    )
    return logging.getLogger(server_name)


def run(
    server_name: str,
    mcp_server: MCPServer[Any],
    init: Callable[[argparse.Namespace], None],
    *,
    supports_readonly: bool = False,
    add_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """Run one first-party MCP server with the standard CLI contract.

    Replaces the copy-pasted ``main()`` body shared by every server. Builds
    the standard parser (via :func:`build_parser`), optionally extends it
    with server-specific args, parses argv, configures logging, calls
    ``init(args)`` to let the server populate its lifespan config, then
    runs the stdio server. ``init`` is the only server-specific seam; it
    receives the parsed namespace and must stash any resolved config onto
    the server's lifespan config holder before the lifespan closure reads
    it.
    """
    parser = build_parser(server_name, supports_readonly=supports_readonly)
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name=server_name)
    data_dir = resolve_data_dir(args.data_dir)
    log.info("%s starting (data_dir=%s)", server_name, data_dir)

    init(args)
    mcp_server.run(transport=args.transport)
