"""Discovery helpers for installed signal collector extensions."""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InstalledSignalCollector:
    """Minimal installed collector metadata used by kernel registry consumers."""

    collector_id: str
    display_name: str


def discover_installed_signal_collectors() -> dict[str, InstalledSignalCollector]:
    """Return installed collector IDs mapped to basic display metadata."""

    collectors: dict[str, InstalledSignalCollector] = {}
    for ep in importlib.metadata.entry_points(group="lisa.extensions.signal_connectors"):
        try:
            loaded = ep.load()
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to load signal collector %r", ep.name, exc_info=True)
            continue

        collector_id = getattr(loaded, "collector_id", "")
        if not isinstance(collector_id, str) or not collector_id:
            continue

        display_name = getattr(loaded, "display_name", "")
        collectors[collector_id] = InstalledSignalCollector(
            collector_id=collector_id,
            display_name=(
                display_name
                if isinstance(display_name, str) and display_name
                else collector_id.replace("_", " ").title()
            ),
        )

    return collectors


__all__ = ["InstalledSignalCollector", "discover_installed_signal_collectors"]
