"""Immutable session capability bindings for runtime dispatch."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from kernel.capability_binding import (
    CapabilityBinding,
    CapabilityBindingProvider,
    CapabilityKind,
    CapabilityRef,
)

logger = logging.getLogger(__name__)

EMPTY_CAPABILITY_CONFIGURATION: Mapping[str, Any] = MappingProxyType({})


class SessionCapabilityBindings(CapabilityBindingProvider):
    """Provide immutable configuration bindings for active catalog owners."""

    def __init__(
        self,
        configuration: Mapping[str, Mapping[str, Any]] | None = None,
        capabilities: Iterable[CapabilityRef] | None = None,
        declared_owners: Iterable[str] = (),
    ) -> None:
        raw = configuration or {}
        capability_list = tuple(capabilities) if capabilities is not None else None
        known_owners = {capability.owner_id for capability in capability_list or ()}
        known_owners.update(declared_owners)
        # Exact-set semantics: bindings resolve configuration by the catalog
        # owner id, so a config key without an exact owner match binds as
        # EMPTY even when a slug alias would look equivalent. Keep the
        # missing-owner set exact (2026-11-02 WiDWebsite incident).
        unknown_owners = set(raw) - known_owners if capability_list else set()
        if unknown_owners:
            # Session plans configure optional packs (the shipped default sets a
            # sandbox floor for lisa-pack-bwrap), so an absent owner is inert,
            # not fatal — same contract as an uninstalled `include` entry.
            # Name the catalog capabilities the absent owner still claims: when
            # the owner's ops/tools are live in the catalog while the pack is
            # not active, every dispatch of them fails with "not registered on
            # any MCP server" or a missing backend mid-turn (2026-11-02
            # WiDWebsite profile: memory writes lost, no receipt, warning
            # buried in runtime.jsonl).
            def _aliases(owner: str) -> bool:
                # The catalog can carry the bare directory slug ('memory') while
                # the configuration key is the distribution name
                # ('lisa-pack-memory') — one pack, two id forms.
                normal_owner = owner.strip().lower().replace("_", "-")
                for unknown in unknown_owners:
                    normal_unknown = unknown.strip().lower().replace("_", "-")
                    if normal_unknown == normal_owner:
                        return True
                    if normal_unknown.removeprefix("lisa-pack-") == normal_owner:
                        return True
                return False

            failing = sorted(
                f"{capability.kind}:{capability.capability_id}"
                for capability in capability_list or ()
                if _aliases(capability.owner_id)
            )
            if failing:
                logger.warning(
                    "Session configuration names capability owners absent from the catalog: %s "
                    "— their configuration is inert this session. The catalog still lists "
                    "their capabilities (%s), which will fail at dispatch; activate the pack "
                    "(include/required) or select the owner (providers.capabilities).",
                    ", ".join(sorted(unknown_owners)),
                    ", ".join(failing),
                )
            else:
                logger.warning(
                    "Session configuration names capability owners absent from the catalog: %s "
                    "— their configuration is inert this session.",
                    ", ".join(sorted(unknown_owners)),
                )
        owner_bindings: Mapping[str, CapabilityBinding] = MappingProxyType(
            {
                str(owner_id): CapabilityBinding(owner_id=str(owner_id), configuration=values)
                for owner_id, values in raw.items()
                if str(owner_id).strip() and isinstance(values, Mapping)
            }
        )
        self._bindings: Mapping[tuple[CapabilityKind, str, str], CapabilityBinding] | None = None
        if capability_list is not None:
            self._bindings = MappingProxyType(
                {
                    (capability.kind, capability.capability_id, capability.owner_id): (
                        owner_bindings.get(
                            capability.owner_id,
                            CapabilityBinding(
                                owner_id=capability.owner_id,
                                configuration=EMPTY_CAPABILITY_CONFIGURATION,
                            ),
                        )
                    )
                    for capability in capability_list
                }
            )
        self._owner_bindings = owner_bindings

    def binding_for(self, capability: CapabilityRef) -> CapabilityBinding | None:
        """Return the owner-scoped configuration for one catalog capability."""
        if self._bindings is not None:
            return self._bindings.get(
                (capability.kind, capability.capability_id, capability.owner_id)
            )
        return self._owner_bindings.get(
            capability.owner_id,
            CapabilityBinding(
                owner_id=capability.owner_id,
                configuration=EMPTY_CAPABILITY_CONFIGURATION,
            ),
        )
