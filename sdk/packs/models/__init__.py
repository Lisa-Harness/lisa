"""Models pack — natural-language model manifest editing.

Provides the ``advise_models`` operation, invoked by the ``/models update
<natural language>`` host command (``hosts/core/commands.py``). The operation
reads the current manifest (``models.proposed.yaml`` or ``models.yaml``),
calls an LLM to interpret the instruction, edits + validates the manifest,
and writes the result back to the proposal file. The host command then
promotes the proposal to the active manifest.

Lives in the framework tree at ``sdk/packs/models`` (import path
``sdk.packs.models``) and is discovered via the
``lisa.extensions.content_roots`` entry point.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent
