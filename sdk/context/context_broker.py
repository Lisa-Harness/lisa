"""Query-aware context brokerage — public API (re-exported from kernel).

This module re-exports the context brokerage API from the kernel and manages
the module-level extension evidence builder (which must be in the SDK layer
since it depends on entry-point discovery).

The separation of concerns:
    - kernel.context.context_broker: Pure ranking logic (runtime-neutral)
    - sdk.context.context_broker: Extension builder registration (entry-point aware)
"""

from __future__ import annotations

from kernel.context.context_broker import (
    build_ambient_facts,
    build_context_evidence_catalog,
    render_context_packet_markdown,
    reset_extension_evidence_builder,
    select_ranked_context,
    set_extension_evidence_builder,
)

__all__ = [
    "build_ambient_facts",
    "build_context_evidence_catalog",
    "render_context_packet_markdown",
    "reset_extension_evidence_builder",
    "select_ranked_context",
    "set_extension_evidence_builder",
]
