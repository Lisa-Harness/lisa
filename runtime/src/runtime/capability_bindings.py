"""Immutable session capability bindings for runtime dispatch."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from kernel.capability_binding import CapabilityBinding, CapabilityBindingProvider, CapabilityRef

EMPTY_CAPABILITY_CONFIGURATION: Mapping[str, Any] = MappingProxyType({})


class SessionCapabilityBindings(CapabilityBindingProvider):
    """Provide immutable configuration bindings for active catalog owners."""

    def __init__(
        self,
        configuration: Mapping[str, Mapping[str, Any]] | None = None,
        capabilities: Iterable[CapabilityRef] | None = None,
    ) -> None:
        raw = configuration or {}
        capability_list = tuple(capabilities) if capabilities is not None else None
        known_owners = {capability.owner_id for capability in capability_list or ()}
        unknown_owners = set(raw) - known_owners if capability_list else set()
        if unknown_owners:
            raise ValueError(
                "Session configuration names unknown capability owners: "
                + ", ".join(sorted(unknown_owners))
            )
        owner_bindings: Mapping[str, CapabilityBinding] = MappingProxyType(
            {
                str(owner_id): CapabilityBinding(owner_id=str(owner_id), configuration=values)
                for owner_id, values in raw.items()
                if str(owner_id).strip() and isinstance(values, Mapping)
            }
        )
        self._bindings: Mapping[tuple[str, str, str], CapabilityBinding] | None = None
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
