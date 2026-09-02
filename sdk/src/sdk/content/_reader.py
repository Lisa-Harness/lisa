"""Authored-text reading for the ``lisa-content`` MCP server.

Encapsulates the path-safety check, root traversal, and read cache that
previously lived inline in ``mcp_server/content/server.py``. The MCP
server is now a thin wrapper: it calls :func:`read_authored_text` here.

This gives the SDK a complete contract for authored-text reading that can
be fulfilled without MCP — a host or tool that wants in-process content
access can call this function directly instead of spawning the MCP server.
"""

from __future__ import annotations

from pathlib import Path

from sdk.content._roots import resolve_content_roots, roots_fingerprint

__all__ = ["ContentReader", "is_safe_relative_path"]


def is_safe_relative_path(normalized: str) -> bool:
    """Reject paths that could escape a content root.

    Blocks absolute paths and any path containing a ``..`` segment so a
    crafted ``relative_path`` cannot read files outside discovered roots.
    """
    if not normalized:
        return False
    if Path(normalized).is_absolute():
        return False
    if ".." in Path(normalized).parts:
        return False
    return True


class ContentReader:
    """Filesystem reader for authored text assets across merged content roots.

    Holds an in-memory read cache keyed by ``(roots_fingerprint, relative_path)``
    so repeated reads of the same asset in one server session skip the filesystem.
    Call :meth:`invalidate` after content roots change.

    Args:
        extra_roots: Additional content roots beyond discovered ones.
            The server passes ``--extra-roots`` here; tests pass tmp dirs.
    """

    def __init__(self, extra_roots: list[Path] | None = None) -> None:
        self._extra_roots: list[Path] = list(extra_roots) if extra_roots else []
        self._read_cache: dict[tuple[str, str], str] = {}

    @property
    def extra_roots(self) -> list[Path]:
        return list(self._extra_roots)

    def roots(self) -> list[Path]:
        """Return the current merged content root list."""
        return resolve_content_roots(self._extra_roots)

    def read_authored_text(self, relative_path: str) -> str:
        """Return the first matching authored text asset, or empty string.

        Traverses roots in discovery order (installed → configured → extra).
        Caches hits only — misses are NOT cached so that content added
        mid-session (e.g. a pack installed after startup) is found on the
        next lookup without requiring an explicit :meth:`invalidate` call.
        Re-scanning absent paths is cheap relative to a model call.
        """
        normalized = relative_path.strip().strip("/\\")
        if not is_safe_relative_path(normalized):
            return ""
        fp = roots_fingerprint(self.roots())
        cache_key = (fp, normalized)
        cached = self._read_cache.get(cache_key)
        if cached is not None:
            return cached
        for root in self.roots():
            candidate = root / normalized
            if candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8")
                    self._read_cache[cache_key] = content
                    return content
                except OSError:
                    continue
        return ""

    def invalidate(self) -> None:
        """Clear the file-read cache after content roots change."""
        self._read_cache.clear()
