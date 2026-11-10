"""In-process catalog/content sources backed by the shared SDK service.

Implements the kernel's :class:`~kernel.context._catalog_sources.CatalogSource`
and :class:`~kernel.context._catalog_sources.ContentSource` contracts by
calling the same ``sdk.content`` builders the ``lisa-catalog`` and
``lisa-content`` MCP servers call — one implementation of front-matter
parsing, entry-point discovery, provider merging, and read caching shared
by the subprocess and in-process paths (Slice B of
``docs/SPEC/PROMPT_STARTUP.md``).

All configuration arrives as explicit inputs derived once from the host's
immutable typed configuration; no ``os.environ`` reads or mutations happen
on these paths. Reads are synchronous — these sources are only wired into
synchronous kernel consumers.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from sdk.content import (
    ContentReader,
    build_mode_catalog_json,
    build_operation_catalog_json,
    build_tool_catalog_json,
    invalidate_catalog_cache,
    resolve_mode_roots,
    resolve_operation_roots,
    resolve_tool_roots,
)
from sdk.content._roots import resolve_content_roots, roots_fingerprint

logger = logging.getLogger(__name__)

__all__ = ["CatalogSourceConfig", "DirectCatalogSource", "DirectContentSource"]


@dataclass(frozen=True, slots=True)
class CatalogSourceConfig:
    """Immutable in-process source configuration.

    Mirrors, field for field, the environment the stdio servers receive
    (see ``sdk.mcp_host.content_servers.build_content_mcp_config``) so the
    two transports observe identical inputs. Resolve roots, disabled
    packs, session provider selections, overlay root and default MCP tool
    IDs once on the host and pass this typed configuration.
    """

    content_roots: list[Path]
    """Active SDK content roots (pack roots + configured roots)."""
    disabled_packs: frozenset[str]
    """Normalized pack names excluded from entry-point discovery."""
    provider_selections_json: str
    """The exact ``LISA_SESSION_PROVIDER_SELECTIONS`` JSON payload."""
    overlays_dir: Path | None = None
    """User-authored overlays directory (prepended when it exists)."""
    default_mcp_tool_ids: frozenset[str] = frozenset()
    """Namespaced MCP-native ambient tool ids (``server_id.tool``)."""
    packs_venv_site_packages: Path | None = None
    """Packs venv site-packages (kept for wiring parity with stdio)."""

    @classmethod
    def from_payloads(
        cls,
        *,
        content_roots: list[Path],
        disabled_packs: set[str] | frozenset[str],
        provider_selections_json: str,
        overlays_dir: Path | None,
        default_mcp_tool_ids: set[str] | frozenset[str] | None = None,
        packs_venv_site_packages: Path | None = None,
    ) -> CatalogSourceConfig:
        """Build a frozen config; ``None`` overlays normalize to absent."""
        return cls(
            content_roots=list(content_roots),
            disabled_packs=frozenset(disabled_packs),
            provider_selections_json=provider_selections_json,
            overlays_dir=overlays_dir,
            default_mcp_tool_ids=frozenset(default_mcp_tool_ids or ()),
            packs_venv_site_packages=packs_venv_site_packages,
        )

    def effective_roots(self) -> list[Path]:
        """Return the merged roots exactly as the stdio server derives them.

        Mirrors ``build_content_mcp_config``: pack roots plus ``content/``
        subdirectories, overlays prepended when the directory exists.
        """
        subdir_roots = [r / "content" for r in self.content_roots if (r / "content").is_dir()]
        configured_roots = _unique_resolved([*self.content_roots, *subdir_roots])
        if self.overlays_dir is not None and self.overlays_dir.is_dir():
            configured_roots = [self.overlays_dir, *configured_roots]
        return resolve_content_roots(
            disabled_packs=os.pathsep.join(sorted(self.disabled_packs)),
            configured_roots=configured_roots,
        )

    def selections_json(self) -> str:
        """Return the provider-selections payload (empty string when unset)."""
        return self.provider_selections_json


def _unique_resolved(paths: list[Path]) -> list[Path]:
    """De-duplicate resolved paths, preserving first-seen order."""
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


class DirectCatalogSource:
    """Synchronous ``CatalogSource`` calling the shared SDK catalog builders.

    Same functions, same caches, same collision errors as the
    ``lisa-catalog`` MCP tools — minus the subprocess.
    """

    def __init__(self, config: CatalogSourceConfig) -> None:
        self._config = config

    def list_modes(self) -> str:
        """Return the mode catalog JSON (same shape as ``lisa-catalog``)."""
        roots = self._config.effective_roots()
        overlay_root = self._config.overlays_dir
        if overlay_root is not None and not overlay_root.is_dir():
            overlay_root = None
        return build_mode_catalog_json(
            resolve_mode_roots(roots),
            operation_catalog_json=self.list_operations(),
            tool_catalog_json=self.list_tools(),
            default_mcp_tool_ids=set(self._config.default_mcp_tool_ids),
            overlay_root=overlay_root,
            provider_selections_json=self._config.selections_json(),
        )

    def list_operations(self) -> str:
        """Return the operation catalog JSON."""
        return build_operation_catalog_json(
            resolve_operation_roots(self._config.effective_roots()),
            provider_selections_json=self._config.selections_json(),
        )

    def list_tools(self) -> str:
        """Return the internal tool catalog JSON."""
        return build_tool_catalog_json(
            resolve_tool_roots(self._config.effective_roots()),
            provider_selections_json=self._config.selections_json(),
        )

    def invalidate(self) -> None:
        """Clear the shared catalog caches (explicit, idempotent)."""
        invalidate_catalog_cache()

    def list_content_roots(self) -> str:
        """Return the effective root list as a JSON array of path strings."""
        return json.dumps([str(r) for r in self._config.effective_roots()])


class DirectContentSource:
    """Synchronous ``ContentSource`` backed by :class:`ContentReader`.

    The exact class the ``lisa-content`` server wraps — path safety,
    traversal rejection, hit-only read caching and explicit invalidation
    are shared by construction.
    """

    def __init__(self, config: CatalogSourceConfig) -> None:
        self._reader = ContentReader(extra_roots=config.effective_roots())

    def read_authored_text(self, relative_path: str) -> str:
        """Return the first matching authored text asset, or empty string."""
        return self._reader.read_authored_text(relative_path)

    def invalidate(self) -> None:
        """Clear the read cache after content roots change."""
        self._reader.invalidate()

    def roots_fingerprint(self) -> str:
        """Return the effective root fingerprint (parity with stdio)."""
        return roots_fingerprint(self._reader.roots())
