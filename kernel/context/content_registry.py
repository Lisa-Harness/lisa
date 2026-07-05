"""Default ``ContentRegistry`` backed by discovered content roots.

Reads authored text assets from pack and extension content roots discovered
through ``kernel.content_roots``. Replaces ``app.content_registry`` for
public hosts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from kernel.content_roots import (
    discover_configured_content_roots,
    discover_installed_content_roots,
)


class DiskContentRegistry:
    """Filesystem-backed content registry merging all discovered content roots.

    ``read_authored_text`` searches every discovered content root for the
    requested relative path and returns the first match. Results are cached
    per path; call ``invalidate()`` after content roots change.
    """

    def __init__(self, *, extra_roots: list[Path] | None = None) -> None:
        self._extra_roots = list(extra_roots) if extra_roots else []

    def _all_roots(self) -> list[Path]:
        installed = discover_installed_content_roots()
        configured = discover_configured_content_roots()
        seen: set[str] = set()
        merged: list[Path] = []
        for root in (*installed, *configured, *self._extra_roots):
            key = str(root.resolve())
            if key in seen:
                continue
            seen.add(key)
            merged.append(root)
        return merged

    @lru_cache(maxsize=512)
    def read_authored_text(self, relative_path: str) -> str:
        """Return the first matching authored text asset, or empty string."""

        normalized = relative_path.strip().strip("/\\")
        if not normalized:
            return ""
        for root in self._all_roots():
            candidate = root / normalized
            if candidate.is_file():
                try:
                    return candidate.read_text(encoding="utf-8")
                except OSError:
                    continue
        return ""

    def invalidate(self) -> None:
        """Clear the read cache after content roots change."""

        self.read_authored_text.cache_clear()