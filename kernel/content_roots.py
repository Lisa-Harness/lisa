"""Public helpers for discovering installed and configured content-pack roots."""

from __future__ import annotations

import importlib.metadata
import logging
import os
from collections.abc import Iterable
from pathlib import Path

from kernel.config import settings
from kernel.pack_activation import _normalize_pack_name

logger = logging.getLogger(__name__)

_CONTENT_ROOT_ENTRY_POINT_GROUP = "lisa.extensions.content_roots"


def _entry_point_pack_name(ep: importlib.metadata.EntryPoint) -> str:
    """Return the distribution package name for an entry point, or empty string."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    name = dist.metadata.get("Name", "") if dist.metadata else ""
    return _normalize_pack_name(str(name or ""))


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def discover_installed_content_roots(
    excluded_packs: set[str] | None = None,
) -> list[Path]:
    """Return installed content roots contributed through entry points.

    Args:
        excluded_packs: Normalized pack names to exclude from discovery.
            When ``None`` (default), no filtering is applied — all installed
            packs are returned. When non-empty, entry points whose distribution
            package name is in the set are skipped.
    """

    roots: list[Path] = []
    for ep in importlib.metadata.entry_points(group=_CONTENT_ROOT_ENTRY_POINT_GROUP):
        if excluded_packs and _entry_point_pack_name(ep) in excluded_packs:
            continue
        try:
            root_loader = ep.load()
            candidate = Path(root_loader())
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to load content root entry-point %r", ep.name, exc_info=True)
            continue
        if candidate.is_dir():
            roots.append(candidate)
    return _dedupe_paths(roots)


def discover_configured_content_roots() -> list[Path]:
    """Return folder-installed content roots from ``SDK_CONTENT_ROOTS``.

    Uses ``os.pathsep`` so Windows can pass drive-letter paths safely.
    """

    try:
        raw = settings.sdk_content_roots
    except Exception:  # pylint: disable=broad-except
        return []

    roots: list[Path] = []
    for part in raw.split(os.pathsep) if raw else []:
        normalized = part.strip().strip('"')
        if not normalized:
            continue
        candidate = Path(normalized)
        if candidate.is_dir():
            roots.append(candidate)
        else:
            logger.warning("SDK content root '%s' does not exist, skipping.", candidate)
    return _dedupe_paths(roots)


def resolve_content_subdir_roots(roots: Iterable[Path], subdir: str) -> list[Path]:
    """Return existing subdirectory roots under discovered content roots."""

    normalized_subdir = subdir.strip().strip("/\\")
    candidates: list[Path] = []
    for root in roots:
        candidate = root / normalized_subdir if normalized_subdir else root
        if candidate.is_dir():
            candidates.append(candidate)
    return _dedupe_paths(candidates)


__all__ = [
    "discover_configured_content_roots",
    "discover_installed_content_roots",
    "resolve_content_subdir_roots",
]
