"""Lisa Guard SDK — content-safety validation backends.

Public API (import from :mod:`sdk.guard`, not the ``_backend`` module):

- :class:`GuardBackend` — protocol for a content-safety backend
- :class:`PassthroughGuardBackend` — the default no-op backend (allows
  everything, sanitizes nothing)
- :func:`format_guard_result` — render a :class:`GuardResult` as the
  JSON string the lisa-guard MCP tools return
- :func:`log_blocked_guard` — emit a structured audit record for a
  blocked guard check

The SDK owns backend selection and the result/audit shape. The MCP
server owns transport, ``MCPServer`` wiring, ``--backend`` arg parsing,
and the process entrypoint. Hosts or tools that want in-process guard
validation can use a backend directly without spawning the MCP server.
"""

from __future__ import annotations

from sdk.guard._backend import (
    GuardBackend,
    PassthroughGuardBackend,
    format_guard_result,
    get_backend,
    log_blocked_guard,
)

__all__ = [
    "GuardBackend",
    "PassthroughGuardBackend",
    "format_guard_result",
    "get_backend",
    "log_blocked_guard",
]
