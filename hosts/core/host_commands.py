"""Host-only command registry and predicate.

This is the canonical source of truth for slash-command prefixes that are
host control directives (operator UI gestures), not conversational content.
Entries whose ``query`` field matches one of these are stripped from the
conversation transcript fed back to the model (see
:func:`hosts.core.thread.build_recent_thread_excerpt` and
:func:`hosts.core.state_provider.FileBackedStateProvider._compact_thread`).

The model already observes the *effect* of these commands via the live
capability manifest / pack lock injected at context-assembly time;
replaying the raw ``/pack disable ...`` text as "conversation" makes the
model treat a UI gesture as something the user said.

Boundary
--------
This module is a leaf in ``hosts.core`` — it imports nothing from other
``hosts.core`` modules, so it can be consumed by both the command dispatch
layer (``handle_message`` / ``commands``) and the persistence layer
(``thread`` / ``state_provider``) without creating a cycle. The
persistence layer must not own command-name vocabulary; this module is the
shared seam.

Public surface
--------------
- :data:`HOST_COMMAND_PREFIXES`
- :data:`HOST_COMMAND_EXACT`
- :func:`is_host_command_query`
"""

from __future__ import annotations

# ── Host-only command prefixes ───────────────────────────────────────
# Case-insensitive, matched after ``str.strip()``. A query is a host
# command if its stripped lower form starts with one of these prefixes OR
# equals one of the bare-word aliases in ``HOST_COMMAND_EXACT``.
#
# ``/deep`` is deliberately excluded: it routes to the model and the query
# body IS the conversation. ``/quit`` is CLI-only and never appended to the
# thread. Bare-word aliases (``help``, ``tokens``, ...) are matched by the
# exact-match form below.
#
# Keep in sync with ``hosts.core.handle_message._dispatch_host_command``.
HOST_COMMAND_PREFIXES: tuple[str, ...] = (
    "/help",
    "/tokens",
    "/settings",
    "/consolidate",
    "/slice",
    "/checkpoint",
    "/restore",
    "/checkpoints",
    "/pack",
    "/debug",
)
HOST_COMMAND_EXACT: frozenset[str] = frozenset(
    {
        "help",
        "tokens",
        "settings",
        "consolidate",
        "slice",
        "checkpoints",
    }
)


def is_host_command_query(query: str) -> bool:
    """Return ``True`` when ``query`` is a host-only command, not dialogue.

    Used by the thread excerpt builder and the thread digest compactor to
    strip operator UI gestures (``/pack disable github``, ``/help``,
    ``/tokens``, ...) from the conversation transcript that gets injected
    back into the model's context. The model observes the *effect* of these
    commands through the live capability manifest and pack lock; replaying
    the raw command text as "recent conversation" makes the model treat a
    UI gesture as something the user said.
    """
    stripped = query.strip().lower()
    if not stripped:
        return False
    if stripped in HOST_COMMAND_EXACT:
        return True
    return stripped.startswith(HOST_COMMAND_PREFIXES)