"""Discovery helpers for installed signal connector extensions."""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InstalledSignalConnector:
    """Minimal installed connector metadata used by kernel registry consumers."""

    connector_id: str
    display_name: str


def discover_installed_signal_connectors() -> dict[str, InstalledSignalConnector]:
    """Return installed connector IDs mapped to basic display metadata."""

    connectors: dict[str, InstalledSignalConnector] = {}
    for ep in importlib.metadata.entry_points(group="lisa.extensions.signal_connectors"):
        try:
            loaded = ep.load()
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to load signal connector %r", ep.name, exc_info=True)
            continue

        connector_id = getattr(loaded, "connector_id", "")
        if not isinstance(connector_id, str) or not connector_id:
            continue

        display_name = getattr(loaded, "display_name", "")
        connectors[connector_id] = InstalledSignalConnector(
            connector_id=connector_id,
            display_name=(
                display_name
                if isinstance(display_name, str) and display_name
                else connector_id.replace("_", " ").title()
            ),
        )

    return connectors


__all__ = ["InstalledSignalConnector", "discover_installed_signal_connectors"]
