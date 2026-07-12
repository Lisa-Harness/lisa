"""Lisa Content SDK — content catalog, authored-text reading, and diagnostics.

Public API:
- :func:`sdk.content._catalog.build_mode_catalog_json` — mode catalog JSON
- :func:`sdk.content._catalog.build_operation_catalog_json` — operation catalog JSON
- :func:`sdk.content._catalog.build_tool_catalog_json` — tool catalog JSON
- :func:`sdk.content._catalog.resolve_mode_roots` — mode subdirectory roots
- :func:`sdk.content._catalog.resolve_operation_roots` — operation subdirectory roots
- :func:`sdk.content._catalog.resolve_tool_roots` — tool subdirectory roots
- :func:`sdk.content._catalog.invalidate_catalog_cache` — clear catalog caches
- :class:`sdk.content._reader.ContentReader` — authored-text file reader
- :func:`sdk.content._reader.is_safe_relative_path` — path-safety check
- :func:`sdk.content._roots.resolve_content_roots` — merged root discovery
- :func:`sdk.content._roots.roots_fingerprint` — cache key for root sets
- :func:`sdk.content._diagnostics.build_catalog_diagnostics_json` — diagnostics report

The MCP servers (``lisa-content`` for file reading, ``lisa-catalog`` for
catalogs and diagnostics) are thin wrappers over these functions and classes.
Hosts or tools that want in-process content access can call them directly
without spawning an MCP server.
"""

from __future__ import annotations

__all__: list[str] = []
