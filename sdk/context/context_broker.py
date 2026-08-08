"""Context brokerage public API — re-exported from kernel.

This module re-exports the runtime-neutral context helpers from the kernel and
manages the module-level extension evidence builder (which must be in the SDK
layer since it depends on entry-point discovery).

The separation of concerns:
    - kernel.context.context_broker: Runtime-neutral helpers (ambient facts,
      hostname, extension-evidence builder registration)
    - sdk.context.context_broker: Extension builder registration (entry-point
      aware)

Context *selection* is owned by the session-owned
:class:`~kernel.context.context_mapper.ContextMapper`; the legacy namespace
broker and its product-default routing config were removed in favor of that
single engine.
"""

from __future__ import annotations

from kernel.context.context_broker import (
    build_ambient_facts,
    reset_extension_item_builder,
    set_extension_item_builder,
)

__all__ = [
    "build_ambient_facts",
    "reset_extension_item_builder",
    "set_extension_item_builder",
]
