"""Lisa Memory SDK — SQLite-backed memory store library.

Public API (import from :mod:`sdk.memory`, or the ``store`` module for the
implementation):

- :class:`SqliteMemoryStore` — scoped memory records, recall ranking, and
  access tracking over a local SQLite file.
- :class:`ReadOnlyMemoryStoreError` — raised by mutating operations when the
  store was opened with ``readonly=True``.
- :class:`EmbeddingProvider` — the optional semantic-recall protocol.
- ``SCHEMA_VERSION`` / ``DEFAULT_TTL_DAYS`` — schema and retention constants.

The ``lisa-memory`` MCP server (``mcp_server.memory.server``) wraps this;
hosts or tools that want in-process memory access can use the store
directly without spawning the server.
"""

from __future__ import annotations

from sdk.memory.store import (
    DEFAULT_TTL_DAYS,
    SCHEMA_VERSION,
    EmbeddingProvider,
    ReadOnlyMemoryStoreError,
    SqliteMemoryStore,
)

__all__ = [
    "DEFAULT_TTL_DAYS",
    "SCHEMA_VERSION",
    "EmbeddingProvider",
    "ReadOnlyMemoryStoreError",
    "SqliteMemoryStore",
]
