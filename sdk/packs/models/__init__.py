"""Models pack — natural-language manifest *assignment* editing.

Provides the ``advise_models`` operation, invoked by the ``/models update
<natural language>`` host command (``hosts/core/commands.py``). The operation
reads the current manifest (``models.proposed.yaml`` or ``models.yaml``),
calls an LLM to interpret the instruction, and writes the result back to the
proposal file. The host command then promotes the proposal to the active
manifest.

Scope is deliberately narrow: only ``mode_models`` (the mode→model_id map)
and the top-level ``roles`` map may be edited. The operation cannot add or
remove models, reorder the list, or edit other per-model fields — a post-LLM
scope guard rejects any such edit. Use ``/models discover`` to rebuild the
catalog, or hand-edit ``models.yaml`` and run ``/models reload`` for anything
outside assignments/roles.

This pack also declares the one built-in model role (``deep``) through the
``lisa.extensions.model_roles`` entry point, so the kernel has zero hardcoded
role names. Packs that want to contribute new roles (e.g. ``vision``,
``audio``) register their own entry point in that group.

Lives in the framework tree at ``sdk/packs/models`` (import path
``sdk.packs.models``) and is discovered via the
``lisa.extensions.content_roots`` entry point.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent


def get_declared_roles() -> Mapping[str, str]:
    """Declare the built-in model roles contributed by this pack.

    Returns ``role_name -> default_model_id``. An empty default model id
    means "declare the role name exists; the user must pin it." For ``deep``
    we return an empty string so discovery (which always emits a concrete
    ``deep`` role pin) is the only path that seeds a default — a hand-edited
    manifest with no ``roles`` block simply leaves ``deep`` unresolved until
    the user pins it or runs ``/models discover``.
    """

    return {"deep": ""}
