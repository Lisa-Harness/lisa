"""Public helpers for discovering installed and configured content-pack roots."""

from __future__ import annotations

import importlib.metadata
import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path

from kernel.config import settings
from kernel.pack_activation import normalize_pack_name

logger = logging.getLogger(__name__)

CONTENT_ROOT_ENTRY_POINT_GROUP = "lisa.extensions.content_roots"
_LISA_PACKS_DIR_NAME = "lisa_packs"


def _entry_point_pack_name(ep: importlib.metadata.EntryPoint) -> str:
    """Return the distribution package name for an entry point, or empty string."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    name = dist.metadata.get("Name", "") if dist.metadata else ""
    return normalize_pack_name(str(name or ""))


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
    for ep in importlib.metadata.entry_points(group=CONTENT_ROOT_ENTRY_POINT_GROUP):
        if excluded_packs and _entry_point_pack_name(ep) in excluded_packs:
            continue
        try:
            root_loader = ep.load()
            candidate = Path(root_loader())
        except (ImportError, AttributeError, OSError, ValueError):
            logger.warning("Failed to load content root entry-point %r", ep.name, exc_info=True)
            continue
        if candidate.is_dir():
            roots.append(candidate)
    return _dedupe_paths(roots)


def discover_packs_venv_content_roots(  # noqa: C901,PLR0912
    data_dir: Path | None = None,
    *,
    excluded_packs: set[str] | None = None,
    venv_path_resolver: Callable[[Path | None], Path] | None = None,
    site_packages_resolver: Callable[[Path], Path] | None = None,
) -> list[Path]:
    """Discover content roots from packs installed in the packs venv.

    External packs are installed into a separate ``packs-venv`` whose
    site-packages contain ``lisa_packs/<pack_name>/`` directories. These
    directories hold ``modes/``, ``operations/``, ``tools/``, and
    ``content/`` subdirectories — the same layout bundled packs use under
    ``sdk/packs/<pack_name>/``.

    Entry-point discovery (``discover_installed_content_roots``) cannot
    find these packs because ``importlib.metadata.entry_points()`` does not
    re-scan after the packs venv's site-packages are added to ``sys.path``
    at startup. This function scans the filesystem directly, finding every
    ``lisa_packs/<pack_name>/`` directory that looks like a content pack
    (has a ``pyproject.toml`` with a ``[project] name`` starting with
    ``lisa-pack-``).

    Args:
        data_dir: Host data directory used to resolve the packs venv path.
            Passed to ``venv_path_resolver`` when provided.
        excluded_packs: Normalized pack names to exclude from discovery.
        venv_path_resolver: Callable that resolves ``data_dir`` (or ``None``)
            to the packs venv ``Path``. Injected by the host to avoid a
            kernel→hosts import. When ``None``, falls back to a lazy import
            of ``hosts.core.paths`` (preserving behavior for callers that
            have not been updated).
        site_packages_resolver: Callable that maps a venv ``Path`` to its
            ``site-packages`` directory. Injected by the host. When ``None``,
            falls back to the same lazy import.

    Returns:
        Content root paths (the ``lisa_packs/<pack_name>/`` directory),
        deduped and filtered. Returns an empty list when the packs venv
        does not exist or contains no ``lisa_packs/`` directory.
    """
    if venv_path_resolver is None or site_packages_resolver is None:
        from hosts.core.paths import (  # noqa: PLC0415
            default_packs_venv,
            packs_venv_for,
            packs_venv_site_packages,
        )  # type: ignore[import-not-found]

        if venv_path_resolver is None:
            def venv_path_resolver(data_dir: Path | None) -> Path:
                if data_dir is not None:
                    return packs_venv_for(Path(data_dir))
                return default_packs_venv()
        if site_packages_resolver is None:
            def site_packages_resolver(venv_path: Path) -> Path:
                return packs_venv_site_packages(venv_path)

    try:
        venv_path = venv_path_resolver(data_dir)
    except (OSError, ValueError, RuntimeError):
        return []

    site_packages = site_packages_resolver(venv_path)
    lisa_packs_dir = site_packages / _LISA_PACKS_DIR_NAME
    if not lisa_packs_dir.is_dir():
        return []

    import tomllib  # noqa: PLC0415

    roots: list[Path] = []
    for pack_dir in sorted(lisa_packs_dir.iterdir()):
        if not pack_dir.is_dir():
            continue
        if pack_dir.name.startswith("_") or pack_dir.name.startswith("."):
            continue
        # Infer the package name from pyproject.toml.
        pyproject = pack_dir / "pyproject.toml"
        if pyproject.is_file():
            try:
                payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                project = payload.get("project") if isinstance(payload, dict) else None
                name = project.get("name") if isinstance(project, dict) else None
            except (OSError, ValueError):
                name = None
        else:
            name = None
        if not name:
            # Fallback: infer from the directory name.
            name = f"lisa-pack-{pack_dir.name}"
        normalized = normalize_pack_name(str(name))
        if excluded_packs and normalized in excluded_packs:
            continue
        roots.append(pack_dir)
    return _dedupe_paths(roots)


def discover_configured_content_roots() -> list[Path]:
    """Return folder-installed content roots from ``SDK_CONTENT_ROOTS``.

    Uses ``os.pathsep`` so Windows can pass drive-letter paths safely.
    """

    try:
        raw = settings.sdk_content_roots
    except (AttributeError, OSError):
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
    "CONTENT_ROOT_ENTRY_POINT_GROUP",
    "discover_configured_content_roots",
    "discover_installed_content_roots",
    "discover_packs_venv_content_roots",
    "resolve_content_subdir_roots",
]
