"""Execution-time bindings for catalog capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

CapabilityKind = Literal["operation", "tool"]
OwnerId = str


@dataclass(frozen=True, slots=True)
class CapabilityRef:
    """Stable identity of one catalog capability."""

    kind: CapabilityKind
    capability_id: str
    owner_id: OwnerId


@dataclass(frozen=True, slots=True)
class CapabilityBinding:
    """Immutable runtime configuration bound to one catalog capability."""

    owner_id: OwnerId
    configuration: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "configuration", _freeze(self.configuration))


@runtime_checkable
class CapabilityBindingProvider(Protocol):
    """Resolve one concrete catalog capability for the active session."""

    def binding_for(self, capability: CapabilityRef) -> CapabilityBinding | None:
        """Return the binding, or ``None`` when the capability is not active."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


__all__ = [
    "CapabilityBinding",
    "CapabilityBindingProvider",
    "CapabilityKind",
    "CapabilityRef",
    "OwnerId",
]
