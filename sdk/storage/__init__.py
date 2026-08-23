"""Lisa Storage SDK — sandboxed file I/O.

Public API (import from :mod:`sdk.storage`, not the ``_backend`` module):

- :class:`StorageBackend` — owns the files sandbox, deny-glob policy,
  and model-manifest read/write. The ``lisa-storage`` MCP server wraps
  this; hosts or tools that want in-process storage access can use it
  directly without spawning the MCP server.
- :class:`PathRejected` — raised when a path matches a deny glob (distinct
  from the ``None`` returned for unsafe / escaping paths).

The SDK owns the policy (sandbox resolution, denylist, manifest paths).
The MCP server owns transport, ``MCPServer`` wiring, ``--readonly`` /
``--deny-glob`` arg parsing, and the process entrypoint.
"""

from __future__ import annotations

from sdk.storage._backend import PathRejected, StorageBackend

__all__ = ["PathRejected", "StorageBackend"]
