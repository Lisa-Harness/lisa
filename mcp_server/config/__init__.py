"""Shared configuration helpers for all first-party Lisa MCP servers.

This package owns the standard CLI argument contract. Every first-party
server calls :func:`lisa_mcp_config.cli.build_parser` in ``main()`` and
resolves its data directory via :func:`lisa_mcp_config.cli.resolve_data_dir`.

Env vars are for behavioral configuration only (API keys, feature flags,
endpoints). Path-like configuration comes exclusively from CLI args.
"""

from mcp_server.config.cli import build_parser, configure_logging, resolve_data_dir

__all__ = ["build_parser", "configure_logging", "resolve_data_dir"]

