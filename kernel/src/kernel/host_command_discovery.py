"""Discovery for pack-registered host commands.

Loads ``lisa.extensions.host_commands`` entry points, each of which returns
a ``list[HostCommandSpec]``. The host dispatch layer consults the discovered
commands after its own built-in commands.
"""

from __future__ import annotations

import importlib.metadata
import logging

from kernel.extension_names import normalize_pack_name
from kernel.host_commands import HostCommandSpec, PackCommandEntry

logger = logging.getLogger(__name__)

HOST_COMMANDS_ENTRY_POINT_GROUP = "lisa.extensions.host_commands"


def _entry_point_pack_name(ep: importlib.metadata.EntryPoint) -> str:
    """Return the distribution package name for an entry point, or empty string."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    name = dist.metadata.get("Name", "") if dist.metadata else ""
    return normalize_pack_name(str(name or ""))


def discover_pack_host_commands(
    excluded_packs: set[str] | None = None,
) -> list[PackCommandEntry]:
    """Return all pack-registered host command specs.

    Args:
        excluded_packs: Normalized pack names to exclude. When ``None``,
            no filtering is applied.
    """
    entries: list[PackCommandEntry] = []
    for ep in importlib.metadata.entry_points(group=HOST_COMMANDS_ENTRY_POINT_GROUP):
        if excluded_packs and _entry_point_pack_name(ep) in excluded_packs:
            continue
        try:
            loader = ep.load()
            specs_data = loader()
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to load host_commands entry-point %r", ep.name, exc_info=True)
            continue

        specs: list[HostCommandSpec] = []
        if isinstance(specs_data, HostCommandSpec):
            specs = [specs_data]
        elif isinstance(specs_data, list):
            for item in specs_data:
                if isinstance(item, HostCommandSpec):
                    specs.append(item)
        elif callable(specs_data):
            specs = [HostCommandSpec(prefix=str(ep.name), handler=specs_data)]

        if not specs:
            continue
        entries.append(PackCommandEntry(pack_name=_entry_point_pack_name(ep), specs=specs))
    return entries
