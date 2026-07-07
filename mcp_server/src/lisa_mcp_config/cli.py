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
from pathlib import Path

__all__ = ["build_parser", "resolve_data_dir", "configure_logging"]


def build_parser(server_name: str, *, supports_readonly: bool = False) -> argparse.ArgumentParser:
    """Build the standard argument parser for one MCP server.

    Every first-party server calls this in ``main()`` and adds server-specific
    args as needed.
    """
    parser = argparse.ArgumentParser(
        prog=server_name,
        description=f"{server_name} — Lisa Framework MCP server",
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
    """Configure standard logging for one MCP server."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    return logging.getLogger(server_name)