"""Discovery for pack-registered model routers.

Loads ``lisa.extensions.model_routers`` entry points, each of which returns
a callable factory ``(manifest, session_plan) -> ModelRouter | None``.
Hosts consult the discovered factories at ``TurnHooks`` assembly time and
wire the first non-``None`` router into ``ContextSeams.model_router``.

The factory contract is intentionally narrow and mirrors
``kernel.slice_provider_discovery``:

- It receives the loaded ``ModelManifest`` (so it can read the candidate model
  set and per-model metadata like ``cost_tier`` / ``latency_tier`` /
  ``capabilities``) and the resolved ``SessionPlan`` (so it can read
  per-session router tuning if the pack declares any).
- It returns ``None`` when the pack should not contribute a router this
  session (e.g. ``router.backend`` in the manifest does not match the pack's
  backend id). The host skips wiring in that case — no behavior change.
- It returns a ``ModelRouter`` instance whose ``choose(req)`` method the turn
  loop calls every iteration.

This module owns only discovery and pack-name attribution. It does not
construct or invoke the routers — that is the host's job at ``TurnHooks``
assembly time, exactly as with slice providers.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kernel.extension_names import normalize_pack_name

if TYPE_CHECKING:
    from kernel.model_manifest import ModelManifest
    from kernel.seams.protocols import ModelRouter

logger = logging.getLogger(__name__)

MODEL_ROUTERS_ENTRY_POINT_GROUP = "lisa.extensions.model_routers"

#: Factory type: ``(manifest, session_plan) -> router_or_none``.
ModelRouterFactory = Callable[
    ["ModelManifest | None", Any],
    "ModelRouter | None",
]


@dataclass(frozen=True, slots=True)
class PackModelRouterEntry:
    """One discovered model-router factory attributed to its source pack."""

    pack_name: str
    factory: ModelRouterFactory


def _entry_point_pack_name(ep: importlib.metadata.EntryPoint) -> str:
    """Return the distribution package name for an entry point, or empty string."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    name = dist.metadata.get("Name", "") if dist.metadata else ""
    return normalize_pack_name(str(name or ""))


def discover_model_router_factories(
    excluded_packs: set[str] | None = None,
) -> list[PackModelRouterEntry]:
    """Return all pack-registered model-router factories, in entry-point order.

    Args:
        excluded_packs: Normalized pack names to exclude. When ``None``,
            no filtering is applied.

    Hosts call each factory in order and wire the first non-``None`` router
    that applies to the current session. The first applicable factory wins;
    later entries are not consulted. This keeps ordering deterministic and
    avoids two routers racing on the same iteration.
    """
    entries: list[PackModelRouterEntry] = []
    for ep in importlib.metadata.entry_points(group=MODEL_ROUTERS_ENTRY_POINT_GROUP):
        if excluded_packs and _entry_point_pack_name(ep) in excluded_packs:
            continue
        try:
            factory = ep.load()
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to load model_routers entry-point %r", ep.name, exc_info=True)
            continue
        if not callable(factory):
            logger.warning(
                "model_routers entry-point %r did not yield a callable factory; skipping",
                ep.name,
            )
            continue
        entries.append(PackModelRouterEntry(pack_name=_entry_point_pack_name(ep), factory=factory))
    return entries


__all__ = [
    "MODEL_ROUTERS_ENTRY_POINT_GROUP",
    "PackModelRouterEntry",
    "discover_model_router_factories",
]
