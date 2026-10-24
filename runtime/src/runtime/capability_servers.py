"""Capability-based MCP server selection.

A server entry that declares ``capability: <name>`` is an *implementation*
of that capability, competing with every other declared implementation
(built-in host-manifest servers and pack-contributed servers alike) for
ownership of the model-facing tool ids. The pool resolves exactly one
owner per capability at compile time:

- an explicit session selection (``providers.capabilities.<name>``) wins;
- otherwise the built-in (host-manifest) candidate wins — loading a
  competing pack alone never changes what the model sees;
- when no built-in claims the capability, a unique candidate wins, and
  multiple candidates raise (the same explicit-selection philosophy as
  duplicate catalog tool ids);

The winning spec is registered under ``server_id = <capability>`` so the
pool's existing namespacing bridge publishes its tools as
``<capability>.<tool>`` — identical model-facing ids regardless of which
implementation (Python, Go, third-party) is selected. Losers are not
registered at all.

The winner retains its real server id for launch and dispatch. Its
``capability_tools`` declaration defines the stable published contract,
while ``tool_map`` maps canonical names to native wire names.

This module replaces the memory-only ``_select_memory_owner`` path: the
memory pack's backend selection is now one instance of the generic
capability mechanism (``capability: memory`` on the server entry).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from kernel.mcp_launch.policy import MCPServerRegistry, MCPServerSpec

logger = logging.getLogger(__name__)

#: Capability ids double as model-facing tool prefixes; same slug rule as
#: the manifest loader's ``_CAPABILITY_NAME_RE`` (kept local to avoid
#: importing the loader into a selection context).
_CAPABILITY_NAME_RE = re.compile(r"[a-z][a-z0-9_-]*")


def valid_capability_name(name: str) -> bool:
    """Return whether ``name`` is a valid capability slug."""
    return bool(_CAPABILITY_NAME_RE.fullmatch(name or ""))


@dataclass(frozen=True, slots=True)
class CapabilityCandidate:
    """One server spec competing for a capability, with selection metadata."""

    capability: str
    spec: MCPServerSpec
    source: str
    """Manifest layer this candidate was declared in: ``builtin`` (host
    manifest), ``pack`` (pack-contributed), or ``user``."""

    pack_slug: str | None = None
    """Pack slug when ``source == "pack"``; used to match explicit
    ``providers.capabilities`` selections (which name packs)."""

    package_name: str | None = None
    """Pack distribution name when ``source == "pack"``; the second
    identifier explicit selections may name."""

    manifest_path: str | None = None
    """Where the declaring manifest lives, for selection logs/errors."""


@dataclass(frozen=True, slots=True)
class _Selection:
    """Resolved owner for one capability."""

    winner: CapabilityCandidate | None
    losers: tuple[CapabilityCandidate, ...]


def candidate_identifiers(candidate: CapabilityCandidate) -> set[str]:
    """Return the normalized identifiers an explicit selection may match.

    Selections name packs (``providers.capabilities.storage:
    lisa-go-servers``) or, for built-ins, the server_id. Accept the pack
    slug, the distribution name, the ``lisa-pack-``-stripped short form,
    and — always — the spec's own server_id.
    """
    identifiers = {candidate.spec.server_id}
    if candidate.pack_slug:
        identifiers.add(candidate.pack_slug)
    if candidate.package_name:
        identifiers.add(candidate.package_name)
        normalized = candidate.package_name
        if normalized.startswith("lisa-pack-"):
            identifiers.add(normalized[len("lisa-pack-") :])
    return {identifier.strip().lower() for identifier in identifiers if identifier}


def select_capability_owner(
    capability: str,
    candidates: list[CapabilityCandidate],
    *,
    selected_provider: str = "",
) -> _Selection:
    """Resolve one capability to its owning candidate, or None.

    Selection order:

    1. **Explicit selection** — ``selected_provider`` names a pack/server;
       the unique matching candidate wins. No match raises (an explicit
       selection pointing at a non-declared server is a config error, not
       a silent fallback); ambiguous matches raise too.
    2. **Built-in preference** — with no explicit selection, a
       ``source == "builtin"`` candidate wins over pack/user candidates
       (loading a competing pack alone never changes the tool ids the
       model sees). Multiple built-ins claiming one capability raise.
    3. **Unique candidate** — with no built-in, a unique pack/user
       candidate wins.
    4. **Contested** — multiple pack/user candidates and no built-in
       raise, demanding an explicit ``providers.capabilities`` entry (the
       same philosophy as duplicate catalog tool ids).
    """
    if not candidates:
        return _Selection(winner=None, losers=())

    selected = selected_provider.strip().lower()
    if selected:
        matching = [
            candidate for candidate in candidates if selected in candidate_identifiers(candidate)
        ]
        if len(matching) == 1:
            return _Selection(
                winner=matching[0],
                losers=tuple(c for c in candidates if c is not matching[0]),
            )
        if not matching:
            raise CapabilitySelectionError(
                f"providers.capabilities.{capability}={selected_provider!r} does not "
                "match any declared implementation of this capability "
                f"({', '.join(sorted(c.spec.server_id for c in candidates))})"
            )
        raise CapabilitySelectionError(
            f"providers.capabilities.{capability}={selected_provider!r} matches "
            f"multiple declared implementations: "
            f"{', '.join(sorted(c.spec.server_id for c in matching))}"
        )

    builtins = [c for c in candidates if c.source == "builtin"]
    if builtins:
        if len(builtins) > 1:
            raise CapabilitySelectionError(
                f"Multiple built-in servers claim capability {capability!r}: "
                f"{', '.join(sorted(c.spec.server_id for c in builtins))}. "
                "The host manifest must declare exactly one."
            )
        winner = builtins[0]
        return _Selection(winner=winner, losers=tuple(c for c in candidates if c is not winner))

    if len(candidates) == 1:
        return _Selection(winner=candidates[0], losers=())

    raise CapabilitySelectionError(
        f"Multiple servers claim capability {capability!r} and no built-in "
        f"declares it ({', '.join(sorted(c.spec.server_id for c in candidates))}). "
        f"Add providers.capabilities.{capability}: <pack-or-server> to the "
        "session context to select one."
    )


class CapabilitySelectionError(ValueError):
    """Raised when capability ownership cannot be resolved deterministically."""


def winning_spec_as_capability(candidate: CapabilityCandidate) -> MCPServerSpec:
    """Return the winner's spec unchanged, for registration.

    The winner registers under its **real server_id** (``lisa-storage``,
    ``lisa-storage-go``, …) — not the capability — so every downstream
    keyed on server identity keeps working: in-process resolution
    (``mcp_server.registry``), scoped dispatch, log paths, restart
    bookkeeping. The published ``<capability>.<tool>`` naming is a
    **catalog concern**, applied by the pool's
    :class:`~kernel.tools.mcp_client.MCPToolPool` from the
    ``capability`` / ``tool_map`` fields carried on
    :class:`~kernel.tools.mcp_client.MCPClientConfig` (see
    :func:`kernel.mcp_launch.compiler.compile_mcp_config`). Losers are
    never registered, so exactly one implementation is live per
    capability and the published id set is implementation-independent.
    """
    return candidate.spec


def register_capability_server(
    registry: MCPServerRegistry,
    candidate: CapabilityCandidate,
) -> None:
    """Register the winner under its real server id for launch and dispatch."""
    registry.register(winning_spec_as_capability(candidate))


def log_capability_selection(
    capability: str,
    selection: _Selection,
) -> None:
    """Log the selection outcome (winner + skipped losers) at info level."""
    winner_desc = (
        f"{selection.winner.spec.server_id} (source={selection.winner.source}, "
        f"manifest={selection.winner.manifest_path})"
        if selection.winner
        else "(none)"
    )
    logger.info("Selected capability owner %s: %s", capability, winner_desc)
    for loser in selection.losers:
        logger.info(
            "Skipping %s: loses capability %s to %s (source=%s)",
            loser.spec.server_id,
            capability,
            selection.winner.spec.server_id if selection.winner else "(none)",
            loser.source,
        )


__all__ = [
    "CapabilityCandidate",
    "CapabilitySelectionError",
    "log_capability_selection",
    "register_capability_server",
    "select_capability_owner",
    "valid_capability_name",
    "winning_spec_as_capability",
]
