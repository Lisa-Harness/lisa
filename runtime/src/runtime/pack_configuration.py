"""Immutable session configuration snapshots for pack dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

EMPTY_PACK_CONFIGURATION: Mapping[str, Any] = MappingProxyType({})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


class SessionPackConfigurationResolver:
    """Snapshot validated session configuration and look up one pack at a time."""

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
