"""Model-picker pack — natural-language model manifest editing.

Provides the ``advise-models`` mode which lets the user edit the model
manifest (``models.proposed.yaml`` or ``models.yaml``) using natural
language instructions instead of hand-editing YAML. The pack calls an
LLM to interpret the instruction, edits the manifest, validates the
result, and returns a diff summary.

Lives in the framework tree at ``sdk/packs/models`` (import path
``sdk.packs.models``) and is discovered via the ``lisa.extensions.content_roots``
entry point.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent
