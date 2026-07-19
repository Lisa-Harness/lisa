"""Letta (MemGPT) integration pack for Lisa.

Exposes Letta as two layered capabilities:

1. A memory provider behind Lisa's generic ``memory.add`` /
   ``memory.search`` / ``memory.list`` aliases (pack-owned
   ``mcp_servers.yaml``). Durable facts are stored as Letta archival
   memory passages; existing Lisa learn/recall flows work unchanged once
   the pack is selected as the ``memory`` capability owner.

2. Explicit Letta agent operations and a dedicated ``letta`` mode for
   stateful-agent delegation. Normal Lisa chat is not silently routed to
   Letta — delegation is opt-in via the mode and operations.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent