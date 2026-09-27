"""Public helpers for discovering installed and configured content-pack roots."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from kernel.config import settings
from kernel.extension_names import normalize_pack_name

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

    The same pack may be installed in more than one venv visible to
    ``importlib.metadata`` (e.g. the framework venv and the packs venv when
    ``mcp_server._packs_venv_site`` adds the packs venv site-packages to
    ``sys.path`` in the catalog/content subprocess). Two distinct paths for
    one pack produce duplicate catalog entries → ``CatalogCollisionError``
    at boot. Dedupe by normalized pack name (first wins) so each pack
    contributes at most one content root regardless of how many venvs
    expose it.
    """

    roots: list[Path] = []
    seen_packs: set[str] = set()
    for ep in importlib.metadata.entry_points(group=CONTENT_ROOT_ENTRY_POINT_GROUP):
        pack_name = _entry_point_pack_name(ep)
        if excluded_packs and pack_name in excluded_packs:
            continue
        if pack_name and pack_name in seen_packs:
            continue
        try:
            root_loader = ep.load()
            candidate = Path(root_loader())
        except Exception:
            # ep.load() executes foreign entry-point code; a broken pack can
            # raise anything (ImportError, RuntimeError from module init, ...).
            # Skip it rather than aborting discovery for every other pack.
            logger.warning("Failed to load content root entry-point %r", ep.name, exc_info=True)
            continue
        if candidate.is_dir():
            roots.append(candidate)
            if pack_name:
                seen_packs.add(pack_name)
    return _dedupe_paths(roots)


def discover_packs_venv_content_roots(
    data_dir: Path | None = None,
    *,
    excluded_packs: set[str] | None = None,
    venv_path_resolver: Callable[[Path | None], Path],
    site_packages_resolver: Callable[[Path], Path],
) -> list[Path]:
    """Discover content roots from packs installed in the packs venv.

    External packs are installed into a separate ``packs venv`` whose
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
            to the packs venv ``Path``. Injected by the host so the kernel
            never imports ``hosts.*``.
        site_packages_resolver: Callable that maps a venv ``Path`` to its
            ``site-packages`` directory. Injected by the host.

    Returns:
        Content root paths (the ``lisa_packs/<pack_name>/`` directory),
        deduped and filtered. Returns an empty list when the packs venv
        does not exist or contains no ``lisa_packs/`` directory.
    """
    try:
        venv_path = venv_path_resolver(data_dir)
    except (OSError, ValueError, RuntimeError):
        return []

    site_packages = site_packages_resolver(venv_path)
    lisa_packs_dir = site_packages / _LISA_PACKS_DIR_NAME

    roots: list[Path] = []
    # Non-editable installs: lisa_packs/<pack_name>/ directories copied
    # into the packs venv site-packages.
    if lisa_packs_dir.is_dir():
        roots.extend(_scan_lisa_packs_dir(lisa_packs_dir, excluded_packs))
    # Editable installs: uv pip install -e creates __editable___<name>_finder.py
    # files whose MAPPING dict points back to the source lisa_packs/ directory.
    # Entry-point discovery misses these because importlib.metadata caches at
    # startup; the filesystem scan misses them because nothing is copied into
    # site-packages/lisa_packs/. Parse the finder files to recover the source
    # lisa_packs/ dir and scan it the same way as the non-editable path.
    roots.extend(_scan_editable_packs(site_packages, excluded_packs))
    return _dedupe_paths(roots)


def _scan_lisa_packs_dir(
    lisa_packs_dir: Path,
    excluded_packs: set[str] | None,
) -> list[Path]:
    """Scan a ``lisa_packs/`` directory for pack content roots."""
    import tomllib

    roots: list[Path] = []
    for pack_dir in sorted(lisa_packs_dir.iterdir()):
        if not pack_dir.is_dir():
            continue
        if pack_dir.name.startswith("_") or pack_dir.name.startswith("."):
            continue
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
            name = f"lisa-pack-{pack_dir.name}"
        normalized = normalize_pack_name(str(name))
        if excluded_packs and normalized in excluded_packs:
            continue
        roots.append(pack_dir)
    return roots


_EDITABLE_FINDER_PREFIX = "__editable___lisa_pack_"
_EDITABLE_FINDER_SUFFIX = "_finder.py"
# Match: MAPPING: dict[str, str] = {'lisa_packs': '<path>'}
_EDITABLE_MAPPING_RE = re.compile(
    r"MAPPING\s*:\s*dict\[str,\s*str\]\s*=\s*\{\s*'lisa_packs'\s*:\s*'([^']+)'\s*\}"
)


def _scan_editable_packs(
    site_packages: Path,
    excluded_packs: set[str] | None,
) -> list[Path]:
    """Scan editable-install finder files for pack content roots.

    ``uv pip install -e`` writes ``__editable___lisa_pack_<name>_finder.py``
    into site-packages. Each finder's ``MAPPING`` dict maps
    ``'lisa_packs'`` to the source ``lisa_packs/`` directory. This scans
    those source directories for ``<pack_name>/`` content roots the same
    way the non-editable path scans ``site-packages/lisa_packs/``.
    """
    roots: list[Path] = []
    # Guard: packs venv may not exist yet
    if not site_packages.is_dir():
        return roots
    for entry in sorted(site_packages.iterdir()):
        if not entry.is_file() or not entry.name.startswith(_EDITABLE_FINDER_PREFIX):
            continue
        if not entry.name.endswith(_EDITABLE_FINDER_SUFFIX):
            continue
        try:
            text = entry.read_text(encoding="utf-8")
        except OSError:
            continue
        match = _EDITABLE_MAPPING_RE.search(text)
        if not match:
            continue
        source_lisa_packs = Path(match.group(1).encode("utf-8").decode("unicode_escape"))
        if not source_lisa_packs.is_dir():
            continue
        roots.extend(_scan_lisa_packs_dir(source_lisa_packs, excluded_packs))
    return roots


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
