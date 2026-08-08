"""Production ``ExtensionItemBuilder`` that walks installed extensions.

Discovers ``lisa.extensions.context_items`` entry points at build time and
calls each extension's ``build_context_items(runtime_context, state)``
function to collect :class:`~kernel.models.ContextItem` records from
every installed extension (Cloudflare, GitHub, Stripe, etc.).

Hosts call :func:`set_production_extension_item_builder` during
``setup_framework`` so the broker's catalog builder picks up extension
evidence automatically. The default :class:`NoOpExtensionItemBuilder`
returns an empty list when no extensions are installed.
"""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from kernel.pack_activation import normalize_pack_name

logger = logging.getLogger(__name__)

__all__ = [
    "EntryPointExtensionItemBuilder",
    "set_production_extension_item_builder",
]

_ENTRY_POINT_GROUP = "lisa.extensions.context_items"


def _entry_point_pack_name(ep: EntryPoint) -> str:
    """Return the distribution package name for an entry point, or empty string."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    name = dist.metadata.get("Name", "") if dist.metadata else ""
    return normalize_pack_name(str(name or ""))


class EntryPointExtensionItemBuilder:
    """Aggregate context evidence from all installed extensions.

    Walks the ``lisa.extensions.context_items`` entry-point group and
    calls each loaded callable. Each callable must accept
    ``(runtime_context, state)`` and return a ``list[ContextItem]``.
    Failures in one extension are logged and swallowed so a broken
    extension does not block context assembly.

    Each evidence record returned by an extension is tagged with the
    normalized distribution package name (``pack``) so the context broker
    can enforce pack-scoped evidence boundaries: a mode only sees extension
    evidence from its own provider pack plus any packs it explicitly allows
    via ``context_item_packs`` in its ``_mode.md`` front-matter.

    Args:
        excluded_packs: Normalized pack names to exclude from evidence
            collection. When ``None`` (default), all installed extensions
            contribute evidence.
    """

    def __init__(self, excluded_packs: set[str] | None = None) -> None:
        self._excluded_packs = excluded_packs

    def build_extension_context_items(
        self,
        runtime_context: Any,
        state: Any,
    ) -> list[Any]:
        """Return the combined evidence list from all installed extensions."""
        evidence: list[Any] = []
        try:
            eps = entry_points(group=_ENTRY_POINT_GROUP)
        except Exception:  # pylint: disable=broad-except
            logger.debug("No entry-point group %r found", _ENTRY_POINT_GROUP, exc_info=True)
            return evidence

        for ep in eps:
            if self._excluded_packs and _entry_point_pack_name(ep) in self._excluded_packs:
                continue
            pack_name = _entry_point_pack_name(ep)
            try:
                builder_fn = ep.load()
            except Exception:  # pylint: disable=broad-except
                logger.warning("Failed to load context_items entry %r", ep.name, exc_info=True)
                continue

            try:
                contributed = builder_fn(runtime_context, state)
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Extension %r context_items builder raised; continuing.",
                    ep.name,
                    exc_info=True,
                )
                continue

            if contributed:
                for item in contributed:
                    # Tag each evidence record with its source distribution
                    # package name so the broker can enforce pack-scoped
                    # evidence boundaries. Use setattr to support both
                    # Pydantic models and plain dict contributions.
                    if hasattr(item, "pack"):
                        if not item.pack:
                            item.pack = pack_name
                    elif isinstance(item, dict):
                        if not item.get("pack"):
                            item["pack"] = pack_name
                evidence.extend(contributed)

        return evidence


def set_production_extension_item_builder(
    excluded_packs: set[str] | None = None,
) -> None:
    """Install the entry-point-based builder as the broker's extension evidence source.

    Args:
        excluded_packs: Normalized pack names to exclude from evidence
            collection. Forwarded to the ``EntryPointExtensionItemBuilder``
            constructor.
    """
    from sdk.context.context_broker import set_extension_item_builder

    set_extension_item_builder(
        EntryPointExtensionItemBuilder(excluded_packs=excluded_packs)
    )
