"""Composite ``ExtensionItemBuilder`` — merge items from multiple builders.

Mirrors :class:`runtime.callbacks.DelegatingTransport`: the turn loop invokes
one ``build_extension_context_items`` hook per turn, and hosts (the VS Code
sidecar) want to contribute host-supplied context (editor selection,
diagnostics) *alongside* the installed-pack builder's items. This composite
fans the call out to each registered builder in order and concatenates the
results, deduplicating on ``item_id`` so an earlier builder's item wins over
a later one.

Dependency direction: ``sdk.context.composite_item_builder`` →
``kernel.seams.protocols`` (``ExtensionItemBuilder``) only. No runtime or
transport imports.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from kernel.seams.protocols import ExtensionItemBuilder

logger = logging.getLogger(__name__)

__all__: list[str] = ["CompositeExtensionItemBuilder"]


class CompositeExtensionItemBuilder:
    """Concatenate context items from an ordered list of builders.

    The first builder wins the ``item_id`` dedup: pack-contributed items
    are composed before host-supplied ones so a host bridge cannot shadow
    an installed pack's record with the same id.

    A broken or empty builder never blocks assembly — a builder that
    raises is treated as contributing nothing, matching the tolerance the
    entry-point builder applies to individual extensions.
    """

    def __init__(self, builders: Sequence[ExtensionItemBuilder]) -> None:
        self._builders: tuple[ExtensionItemBuilder, ...] = tuple(builders)

    def build_extension_context_items(
        self,
        runtime_context: Any,
        state: Any,
    ) -> list[Any]:
        """Return the merged, deduplicated item list across all builders.

        A builder that raises contributes nothing — logged and swallowed,
        matching the per-extension tolerance of the entry-point builder.
        """
        items: list[Any] = []
        seen: set[str] = set()
        for builder in self._builders:
            try:
                contributed = builder.build_extension_context_items(runtime_context, state)
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Item builder %s raised; continuing without its items.",
                    type(builder).__name__,
                    exc_info=True,
                )
                continue
            for item in contributed or []:
                item_id = getattr(item, "item_id", None)
                if not isinstance(item_id, str) or item_id in seen:
                    continue
                seen.add(item_id)
                items.append(item)
        return items
