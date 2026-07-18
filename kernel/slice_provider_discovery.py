"""Discovery for pack-registered slice providers.

Loads ``lisa.extensions.slice_providers`` entry points, each of which returns
a callable factory ``(turn_hooks, session_plan) -> StateProvider | None``.
Hosts consult the discovered factories when wiring
``TurnHooks.pre_turn_slice`` / ``post_turn_slice`` / ``explicit_slice`` and
chain any non-``None`` providers ahead of the host's own deterministic
slice (e.g. ``FileBackedStateProvider``).

The factory contract is intentionally narrow:

- It receives the host's ``TurnHooks`` (so it can read
  ``last_context_snapshot`` and call ``execute_operation`` /
  ``write_turn_history_fn``) and the resolved ``SessionPlan`` (so it can
  read its own ``context.history.compaction`` tuning).
- It returns ``None`` when the pack should not contribute a slice this
  session (e.g. ``compaction.enabled`` is ``False``). The host skips
  wiring in that case — no behavior change for sessions that don't opt in.
- It returns a callable matching the duck-typed ``slice(ctx, policy)``
  contract used by ``kernel.slice.SlicePolicy`` and
  ``kernel.turn_runner._fire_slice``. The provider's ``slice()`` method
  must respect ``policy.budget_ms`` and short-circuit when the deadline
  has passed.

This module owns only discovery and pack-name attribution. It does not
construct or invoke the providers — that is the host's job at
``TurnHooks`` assembly time. This mirrors the split in
``kernel.host_command_discovery``.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kernel.pack_activation import normalize_pack_name

logger = logging.getLogger(__name__)

SLICE_PROVIDERS_ENTRY_POINT_GROUP = "lisa.extensions.slice_providers"

#: Factory type: ``(turn_hooks, session_plan) -> provider_or_none``.
SliceProviderFactory = Callable[[Any, Any], "Any | None"]


@dataclass(frozen=True, slots=True)
class PackSliceProviderEntry:
    """One discovered slice-provider factory attributed to its source pack."""

    pack_name: str
    factory: SliceProviderFactory


def _entry_point_pack_name(ep: importlib.metadata.EntryPoint) -> str:
    """Return the distribution package name for an entry point, or empty string."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    name = dist.metadata.get("Name", "") if dist.metadata else ""
    return normalize_pack_name(str(name or ""))


def discover_slice_provider_factories(
    excluded_packs: set[str] | None = None,
) -> list[PackSliceProviderEntry]:
    """Return all pack-registered slice-provider factories, in entry-point order.

    Args:
        excluded_packs: Normalized pack names to exclude. When ``None``,
            no filtering is applied.

    Hosts call each factory in order and chain the first non-``None``
    provider that applies to the current session. The first applicable
    provider wins; later entries are not consulted for the same slot.
    This keeps ordering deterministic and avoids two compaction providers
    racing on the same turn.
    """
    entries: list[PackSliceProviderEntry] = []
    for ep in importlib.metadata.entry_points(group=SLICE_PROVIDERS_ENTRY_POINT_GROUP):
        if excluded_packs and _entry_point_pack_name(ep) in excluded_packs:
            continue
        try:
            factory = ep.load()
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Failed to load slice_providers entry-point %r", ep.name, exc_info=True
            )
            continue
        if not callable(factory):
            logger.warning(
                "slice_providers entry-point %r did not yield a callable factory; skipping",
                ep.name,
            )
            continue
        entries.append(PackSliceProviderEntry(pack_name=_entry_point_pack_name(ep), factory=factory))
    return entries


__all__ = [
    "PackSliceProviderEntry",
    "SLICE_PROVIDERS_ENTRY_POINT_GROUP",
    "SliceProviderFactory",
    "discover_slice_provider_factories",
]