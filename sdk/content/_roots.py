"""Shared content-root discovery for the ``lisa-content`` and ``lisa-catalog``
MCP servers.

Centralizes the installed + configured + extra root merge so both servers
(and the SDK diagnostics module) use one consistent discovery path. The
servers call :func:`resolve_content_roots` at startup and on each tool
invocation that needs the root set.

The discovery delegates to :mod:`kernel.content_roots` for entry-point and
``SDK_CONTENT_ROOTS`` env-var scanning, and :mod:`kernel.pack_activation`
for disabled-pack filtering. This module is the SDK's single entry point
for "give me the merged content root list" — callers never touch the
kernel discovery helpers directly.
"""

from __future__ import annotations

from pathlib import Path

from kernel.content_roots import (
    discover_configured_content_roots,
    discover_installed_content_roots,
)
from kernel.pack_activation import load_disabled_pack_names_from_env

__all__ = ["resolve_content_roots", "roots_fingerprint"]


def resolve_content_roots(extra_roots: list[Path] | None = None) -> list[Path]:
    """Return the merged, deduplicated content root list.

    Combines:
    - Installed pack roots discovered through the ``lisa.extensions.content_roots``
      entry-point group (minus disabled packs from ``LISA_DISABLED_PACKS``).
    - Configured roots from the ``SDK_CONTENT_ROOTS`` env var.
    - Caller-supplied ``extra_roots`` (e.g. test fixtures).

    The result is deduplicated by resolved path and ordered
    installed → configured → extra so first-wins semantics are stable.
    """
    excluded = load_disabled_pack_names_from_env()
    installed = discover_installed_content_roots(excluded_packs=excluded)
    configured = discover_configured_content_roots()
    seen: set[str] = set()
    merged: list[Path] = []
    for root in (*installed, *configured, *(extra_roots or [])):
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        merged.append(root)
    return merged


def roots_fingerprint(roots: list[Path]) -> str:
    """Return a stable string key for a set of content roots.

    Used as a cache key for file-read and catalog caches.
    """
    return "\n".join(str(r.resolve()) for r in roots)