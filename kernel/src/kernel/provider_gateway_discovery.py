"""Discovery for pack-registered provider gateway providers.

Loads ``lisa.extensions.provider_gateways`` entry points, each of which returns
a ``ProviderGatewayProvider`` instance. Hosts consult the discovered providers
at ``TurnHooks`` assembly time and wire the first non-``None`` provider into
``ContextSeams.provider_gateways``.

The factory contract mirrors ``kernel.model_router_discovery``:

- The entry point yields a callable or instance. When it's a zero-arg
  callable (factory-factory), the host calls it to get the real
  ``ProviderGatewayProvider``. When it's already an instance with
  ``gateway_for``, it is used directly.
- It returns ``None`` when the pack should not contribute a provider this
  session. The host skips wiring in that case — no behavior change.

This module owns only discovery and pack-name attribution. It does not
construct or invoke the providers — that is the host's job at ``TurnHooks``
assembly time, exactly as with model routers and slice providers.
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass
from typing import Any

from kernel.extension_names import normalize_pack_name

logger = logging.getLogger(__name__)

PROVIDER_GATEWAYS_ENTRY_POINT_GROUP = "lisa.extensions.provider_gateways"


@dataclass(frozen=True, slots=True)
class PackProviderGatewayEntry:
    """One discovered provider-gateway provider attributed to its source pack."""

    pack_name: str
    provider: Any


def _entry_point_pack_name(ep: importlib.metadata.EntryPoint) -> str:
    """Return the distribution package name for an entry point, or empty string."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    name = dist.metadata.get("Name", "") if dist.metadata else ""
    return normalize_pack_name(str(name or ""))


def discover_provider_gateway_factories(
    excluded_packs: set[str] | None = None,
) -> list[PackProviderGatewayEntry]:
    """Return all pack-registered provider-gateway providers, in entry-point order.

    Args:
        excluded_packs: Normalized pack names to exclude. When ``None``,
            no filtering is applied.

    Hosts call each factory in order and wire the first non-``None`` provider
    that applies to the current session. The first applicable factory wins;
    later entries are not consulted. This keeps ordering deterministic.
    """
    entries: list[PackProviderGatewayEntry] = []
    for ep in importlib.metadata.entry_points(group=PROVIDER_GATEWAYS_ENTRY_POINT_GROUP):
        if excluded_packs and _entry_point_pack_name(ep) in excluded_packs:
            continue
        try:
            loaded = ep.load()
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Failed to load provider_gateways entry-point %r", ep.name, exc_info=True
            )
            continue
        entries.append(
            PackProviderGatewayEntry(pack_name=_entry_point_pack_name(ep), provider=loaded)
        )
    return entries


__all__ = [
    "PROVIDER_GATEWAYS_ENTRY_POINT_GROUP",
    "PackProviderGatewayEntry",
    "discover_provider_gateway_factories",
]
