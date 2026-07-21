"""Letta (MemGPT) integration pack for Lisa.

Exposes Letta as two layered capabilities:

1. A memory provider behind Lisa's generic ``memory.add`` /
   ``memory.search`` / ``memory.list`` aliases (pack-owned
   ``mcp_servers.yaml``). Durable facts are stored as Letta archival
   memory passages; existing Lisa learn/recall flows work unchanged once
   the pack is selected as the ``memory`` capability owner.

2. Explicit Letta agent operations and two delegation modes:

   - ``letta`` (chat mode, ``/letta <query>``): the model-driven dispatch
     path. The Lisa model picks among ``letta_send_message``,
     ``letta_search_archival``, ``letta_update_core_memory``,
     ``letta_remember``, ``letta_sync_memory``, ``letta_compact_agent``,
     and ``recall_memory`` based on the user's phrasing. Costs one Lisa
     model round-trip per turn for dispatch plus the Letta agent's own
     internal model call.

   - ``letta-agent`` (action mode, ``/letta-agent <query>``): the
     zero-Lisa-model delegation path. Runs two entry operations in order
     (``letta_ensure_agent`` then ``letta_send_message``) and returns
     the Letta agent's response verbatim. No ``model_call`` stage event
     fires on the Lisa host. Use this for direct "ask the agent" turns
     where you do not need the Lisa model to choose between Letta
     sub-operations.

Normal Lisa chat is not silently routed to Letta — delegation is opt-in
via either mode. ``letta_ensure_agent`` is shared as the first entry
operation of both modes; it is idempotent.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent