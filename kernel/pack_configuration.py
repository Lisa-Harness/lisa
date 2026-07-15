"""Immutable, pack-scoped configuration resolved from a session plan.

Packs receive only their own authored configuration.  The resolver deliberately
has no environment-variable fallback: session policy remains explicit and
portable.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "EMPTY_PACK_CONFIGURATION",
    "PackConfigurationResolver",
    "SessionPackConfigurationResolver",
]


EMPTY_PACK_CONFIGURATION: Mapping[str, Any] = MappingProxyType({})


def _freeze(value: Any) -> Any:
    """Recursively freeze session-authored configuration values."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


@runtime_checkable
class PackConfigurationResolver(Protocol):
    """Resolve immutable configuration for exactly one owning pack."""

    def for_pack(self, pack_id: str) -> Mapping[str, Any]:
        """Return the immutable configuration authored for ``pack_id``."""


class SessionPackConfigurationResolver:
    """Resolve pack configuration from an already-validated session manifest."""

    def __init__(self, configuration: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        raw = configuration or {}
        self._configuration: Mapping[str, Mapping[str, Any]] = MappingProxyType(
            {
                str(pack_id): _freeze(values)
                for pack_id, values in raw.items()
                if str(pack_id).strip() and isinstance(values, Mapping)
            }
        )

    def for_pack(self, pack_id: str) -> Mapping[str, Any]:
        """Return only the requested pack's configuration, or an empty mapping."""

        return self._configuration.get(pack_id.strip(), EMPTY_PACK_CONFIGURATION)