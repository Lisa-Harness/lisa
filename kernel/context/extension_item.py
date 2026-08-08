"""Default ``ExtensionItemBuilder`` that returns no context items.

Production hosts override with a builder that walks installed extensions and
collects their context-item contributions. The pass-through default keeps
public hosts runnable without the private extension-item pipeline.
"""

from __future__ import annotations

from typing import Any


class NoOpExtensionItemBuilder:
    """Return an empty item list for any state."""

    def build_extension_context_items(
        self,
        runtime_context: Any,
        state: Any,
    ) -> list[Any]:
        """Return an empty list. Both arguments are accepted for protocol parity."""

        return []
