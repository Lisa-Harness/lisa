"""Host-only command prefix registry.

This module provides a dynamic registry of slash-command prefixes that are
host control directives (operator UI gestures), not conversational content.
Entries whose ``query`` field matches a registered prefix are stripped from
the conversation transcript fed back to the model (see
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
(``thread`` / ``state_provider``) without creating a cycle.

This module owns NO command-name vocabulary. The static list of commands
that existed here previously was a boundary violation: it duplicated the
dispatch table in ``handle_message._dispatch_host_command`` and hardcoded
pack-owned commands (``/checkpoint``, ``/restore``, ``/checkpoints``,
``/pack``) in the host. The dispatch layer is the authority on what gets
intercepted; it registers its prefixes here so the transcript filter can
consult the registry without owning command vocabulary.

``/quit`` is CLI-only and never appended to the thread.

Public surface
--------------
- :func:`register_host_command_prefix`
- :func:`register_host_command_exact`
- :func:`register_host_command_prefixes`
- :func:`register_host_command_exacts`
- :func:`is_host_command_query`
- :func:`host_command_prefixes`
- :func:`host_command_exact`
"""

from __future__ import annotations

# ── Dynamic registry ─────────────────────────────────────────────────
# Populated by the command dispatch layer (``handle_message``) at import
# time. The persistence layer (``thread``, ``state_provider``) reads from
# this registry via :func:`is_host_command_query`.

_prefixes: set[str] = set()
_exact: set[str] = set()


def register_host_command_prefix(prefix: str) -> None:
    """Register a slash-command prefix as a host-only command.

    Called by the command dispatch layer for each prefix it intercepts.
    A query is a host command if its stripped lower form starts with a
    registered prefix (e.g. ``/pack``, ``/help``, ``/checkpoint``) OR
    equals a registered exact form (e.g. ``help``, ``tokens``).
    """
    _prefixes.add(prefix.strip().lower())


def register_host_command_exact(alias: str) -> None:
    """Register a bare-word alias as a host-only command.

    Called by the command dispatch layer for each bare-word alias it
    intercepts (e.g. ``help``, ``tokens``, ``settings``).
    """
    _exact.add(alias.strip().lower())


def register_host_command_prefixes(prefixes: tuple[str, ...]) -> None:
    """Register multiple slash-command prefixes at once."""
    for prefix in prefixes:
        register_host_command_prefix(prefix)


def register_host_command_exacts(aliases: tuple[str, ...]) -> None:
    """Register multiple bare-word aliases at once."""
    for alias in aliases:
        register_host_command_exact(alias)


def host_command_prefixes() -> tuple[str, ...]:
    """Return the currently registered host-command prefixes (sorted)."""
    return tuple(sorted(_prefixes))


def host_command_exact() -> frozenset[str]:
    """Return the currently registered host-command exact aliases."""
    return frozenset(_exact)


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
    if stripped in _exact:
        return True
    return stripped.startswith(tuple(_prefixes)) if _prefixes else False
