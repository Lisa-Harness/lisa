"""Lisa Storage MCP server — blob I/O over a filesystem backend.

Tools (22, one per ``BlobIO`` method). Bytes use base64 encoding over the
wire. Structured types serialize as JSON strings.
"""

from __future__ import annotations

__all__ = ["mcp", "main", "init_storage"]