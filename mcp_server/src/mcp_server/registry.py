"""In-process MCP server registry.

Maps first-party ``server_id`` strings to their :class:`MCPServer` instance
and ``init`` function so the kernel's :class:`MCPToolPool` can connect to
them in-process via the MCP SDK's in-memory transport — no subprocesses.

Each first-party server module exposes:

- ``mcp`` — the :class:`~mcp.server.mcpserver.MCPServer` instance.
- ``_init(args)`` — populates the server's lifespan config holder from a
  parsed argparse namespace. Called once before the in-process session
  starts.

The registry is a **lazy** mapping: server modules are imported on first
access to keep ``--version`` / ``verify`` fast paths cheap (importing a
server module pulls in SDK backends, catalog scanners, etc.). Once
imported, the :class:`MCPServer` instance is cached for the process
lifetime.

Architecture note: this module lives in ``mcp_server`` (not ``kernel``)
because it imports server modules, and the kernel must not depend on the
MCP server package. The kernel's :class:`MCPToolPool` resolves the
registry via a lazy import of ``mcp_server.registry`` at connection time,
preserving the ``kernel → mcp_server`` dependency direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

__all__ = ["InProcessServerEntry", "get_in_process_server", "in_process_server_ids"]

# Type alias for the init function signature shared by every server module.
# ``args`` is a SimpleNamespace-like object with attributes matching the
# standard CLI contract (``data_dir``, ``workspace_root``, ``readonly``,
# etc.) plus server-specific extras.
InitFn = Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class InProcessServerEntry:
    """One first-party server ready for in-process connection.

    Attributes:
        server_id: The stable identifier used in ``mcp_servers.yaml`` and
            by the pool's routing layer (e.g. ``"lisa-content"``).
        mcp: The :class:`MCPServer` instance — its ``_lowlevel_server``
            drives the in-memory transport.
        init: The server's ``_init(args)`` function. Called once before
            the in-process session starts to populate lifespan config.
    """

    server_id: str
    mcp: MCPServer[Any]
    init: InitFn


# Lazy-import table: server_id → module path. Imported on first access.
_LAZY_IMPORTS: dict[str, str] = {
    "lisa-catalog": "mcp_server.catalog.server",
    "lisa-content": "mcp_server.content.server",
    "lisa-context": "mcp_server.context.server",
    "lisa-guard": "mcp_server.guard.server",
    "lisa-memory": "mcp_server.memory.server",
    "lisa-storage": "mcp_server.storage.server",
    "lisa-thread": "mcp_server.thread.server",
}

# Cache of resolved entries (populated on first access per server_id).
_cache: dict[str, InProcessServerEntry] = {}


def in_process_server_ids() -> tuple[str, ...]:
    """Return the tuple of first-party server_ids available in-process."""
    return tuple(_LAZY_IMPORTS.keys())


def get_in_process_server(server_id: str) -> InProcessServerEntry | None:
    """Resolve one server_id to its :class:`InProcessServerEntry`.

    Imports the server module lazily on first access and caches the
    ``MCPServer`` instance. Returns ``None`` when ``server_id`` is not a
    known first-party server (e.g. a third-party or pack-contributed
    server that doesn't support in-process mode).
    """
    if server_id in _cache:
        return _cache[server_id]

    module_path = _LAZY_IMPORTS.get(server_id)
    if module_path is None:
        return None

    import importlib

    mod = importlib.import_module(module_path)
    entry = InProcessServerEntry(
        server_id=server_id,
        mcp=mod.mcp,
        init=mod._init,
    )
    _cache[server_id] = entry
    return entry