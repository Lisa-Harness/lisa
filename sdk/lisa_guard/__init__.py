"""Lisa Guard MCP server — content safety validation.

Tools:
- ``validate_input`` — validate user input before it reaches the model
- ``validate_output`` — validate model output before it reaches the user
- ``guard_configured`` — report backend configuration

The default backend is ``passthrough`` (allows everything, sanitizes nothing).
Backend selection via ``--backend`` CLI arg (default ``passthrough``). When a
guard blocks, the server logs a structured audit record to stderr.
"""

from __future__ import annotations

__all__ = ["main", "mcp"]
