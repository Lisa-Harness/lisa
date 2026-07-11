"""Lisa Content MCP server — the sole content mediator for the kernel.

Tools:
- ``read_authored_text`` — read an authored text asset from merged roots
- ``list_content_roots`` — list all discovered content root paths
- ``list_modes`` — return the full workflow mode catalog as JSON
- ``list_operations`` — return the full operation catalog as JSON
- ``list_tools`` — return the full internal tool catalog as JSON
- ``invalidate_cache`` — clear the file-read cache
- ``invalidate_catalog`` — clear the mode/operation/tool catalog caches
"""

from __future__ import annotations

__all__ = ["init_content", "main", "mcp"]
