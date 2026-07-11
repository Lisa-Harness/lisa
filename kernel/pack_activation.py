"""Shared helpers for reading disabled-pack state from the installed-packs lock.

The installed-packs lock (``installed-packs.json`` in the host's data directory)
records the ``enabled`` flag for each pack. Four independent entry-point
discovery mechanisms (content roots, signal collectors, processing loops,
context evidence) use this module to determine which packs to exclude from
discovery.

All functions fail open: a missing or corrupt lock file produces an empty
exclusion set so discovery returns to "all installed packs included" — the
same behavior as before this module existed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_PACK_LOCK_FILE_NAME = "installed-packs.json"
_DISABLED_PACKS_ENV = "LISA_DISABLED_PACKS"


def _normalize_pack_name(raw: str) -> str:
    """Normalize a pack name for comparison (lower-case, hyphen-separated)."""
    value = raw.strip().lower().replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    return value


def load_disabled_pack_names(data_dir: Path) -> set[str]:
    """Return the set of disabled pack names from the installed-packs lock.

    Reads ``<data_dir>/installed-packs.json`` and returns normalized package
    names where ``enabled`` is explicitly ``false``. Returns an empty set if
    the file is missing, unreadable, or contains no disabled packs.

    Args:
        data_dir: The host's data directory containing ``installed-packs.json``.

    Returns:
        A set of normalized pack names (e.g. ``{"lisa-pack-cloudflare"}``).
    """
    lock_path = data_dir / _PACK_LOCK_FILE_NAME
    if not lock_path.exists():
        return set()

    try:
        decoded = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to read pack lock %r", str(lock_path), exc_info=True)
        return set()

    if not isinstance(decoded, dict):
        return set()

    packs = decoded.get("packs")
    if not isinstance(packs, list):
        return set()

    disabled: set[str] = set()
    for item in packs:
        if not isinstance(item, dict):
            continue
        if bool(item.get("enabled", True)):
            continue
        package_name = _normalize_pack_name(str(item.get("package_name") or ""))
        if package_name:
            disabled.add(package_name)

    return disabled


def load_disabled_pack_names_from_env() -> set[str]:
    """Return disabled pack names from the ``LISA_DISABLED_PACKS`` env var.

    Used by child processes (e.g. the ``lisa-content`` MCP server) that cannot
    receive Python arguments directly from the host. The env var is
    ``os.pathsep``-separated (matching ``SDK_CONTENT_ROOTS`` convention).

    Returns:
        A set of normalized pack names. Empty if the env var is unset or empty.
    """
    raw = os.environ.get(_DISABLED_PACKS_ENV, "")
    if not raw:
        return set()

    disabled: set[str] = set()
    for part in raw.split(os.pathsep):
        normalized = _normalize_pack_name(part.strip())
        if normalized:
            disabled.add(normalized)
    return disabled


__all__ = [
    "load_disabled_pack_names",
    "load_disabled_pack_names_from_env",
]