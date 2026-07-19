"""Models pack — natural-language manifest *assignment* editing.

Provides the ``advise_models`` operation, invoked by the ``/models update
<natural language>`` host command (``hosts/core/commands.py``). The operation
reads the current manifest (``models.proposed.yaml`` or ``models.yaml``),
calls an LLM to interpret the instruction, and writes the result back to the
proposal file. The host command then promotes the proposal to the active
manifest.

Scope is deliberately narrow: only ``mode_models`` (the mode→model_id map)
and the ``designations`` field on each model entry may be edited. The
operation cannot add or remove models, reorder the list, or edit other
per-model fields — a post-LLM scope guard rejects any such edit. Use
``/models discover`` to rebuild the catalog, or hand-edit ``models.yaml``
and run ``/models reload`` for anything outside assignments/designations.

Lives in the framework tree at ``sdk/packs/models`` (import path
``sdk.packs.models``) and is discovered via the
``lisa.extensions.content_roots`` entry point.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent
