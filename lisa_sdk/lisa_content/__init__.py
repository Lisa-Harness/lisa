"""Lisa Content MCP server — content registry over discovered content roots.

Tools:
- ``read_authored_text`` — read an authored text asset from merged roots
- ``list_content_roots`` — list all discovered content root paths
- ``invalidate_cache`` — clear the read cache
"""

from __future__ import annotations

__all__ = ["mcp", "main", "init_content"]