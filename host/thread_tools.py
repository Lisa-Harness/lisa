"""Host-owned synthetic tools for thread-store access.

The thread store (``host.thread``) is host-owned, so it cannot live in the
``mcp_server`` package (MCP servers may not import hosts). Instead, these
tools are registered as :class:`~kernel.tools.mcp_client.SyntheticToolSpec`
entries on the MCP tool pool, dispatched in-process without a subprocess.

The model uses these tools to pull older thread entries on demand — the
"on-demand depth" layer of the floor + tool design. The eager floor (the
recent excerpt + thread manifest) is injected by ``context_assembly``;
this tool lets the model read deeper when the manifest shows entries it
needs but the floor did not include.

Public surface
--------------

- :func:`register_thread_tools`: register ``thread_read_range`` on a pool.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernel.tools.mcp_client import MCPToolPool, SyntheticToolSpec

logger = logging.getLogger(__name__)

__all__ = ["register_thread_tools"]


def register_thread_tools(
    pool: "MCPToolPool",
    *,
    thread_path_resolver: "Any",
) -> None:
    """Register ``thread_read_range`` on a started MCP tool pool.

    Args:
        pool: A started :class:`MCPToolPool` (synthetic tools register
            after ``start`` so the handler can dispatch).
        thread_path_resolver: A zero-arg callable returning the per-user
            thread base path (``Path | None``), typically
            ``runtime.thread_path``.
    """
    from kernel.tools.mcp_client import SyntheticToolSpec  # noqa: PLC0415

    pool.add_synthetic_tools([
        SyntheticToolSpec(
            name="thread_read_range",
            description=(
                "Read a range of entries from the conversation thread by "
                "global turn index. Use this to recall prior turns beyond "
                "the recent excerpt injected in the system prompt. Returns "
                "a JSON array of thread entries, each containing mode_id, "
                "query, response_text, handoff_to, transitioned_to, and "
                "effect_receipts. Pass start_turn=0 and end_turn=999999 for "
                "the full thread."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "start_turn": {
                        "type": "integer",
                        "description": (
                            "The global (0-based) turn index to start reading "
                            "from, inclusive."
                        ),
                    },
                    "end_turn": {
                        "type": "integer",
                        "description": (
                            "The global turn index to stop reading at, "
                            "inclusive. Use a large number for 'to the end'."
                        ),
                    },
                },
                "required": ["start_turn", "end_turn"],
            },
            handler=_make_thread_read_range_handler(thread_path_resolver),
        ),
    ])


def _make_thread_read_range_handler(
    thread_path_resolver: "Any",
) -> "Any":
    """Build the async handler closure for ``thread_read_range``."""

    async def _handler(arguments: dict[str, Any]) -> str:
        from host.thread import load_thread_range  # noqa: PLC0415

        start_turn = int(arguments.get("start_turn", 0))
        end_turn = int(arguments.get("end_turn", 0))
        thread_path: Path | None = thread_path_resolver()
        entries = load_thread_range(
            thread_path=thread_path,
            start_turn=start_turn,
            end_turn=end_turn,
        )
        # Project to a compact form to keep the token cost down. The model
        # rarely needs the full response_text of every entry; the compact
        # form carries mode_id, query, a truncated response, handoff, and
        # receipt capability ids — enough to decide whether to read deeper.
        compact = [_compact_entry(e) for e in entries]
        return json.dumps(compact, ensure_ascii=False)

    return _handler


def _compact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Project a full thread entry to a compact form for the model."""
    response = str(entry.get("response_text") or "")
    # Truncate the response to keep the tool output bounded. The model
    # can call thread_read_range with a tighter range if it needs the
    # full text of a specific turn.
    max_chars = 2000
    if len(response) > max_chars:
        response = response[:max_chars] + "\n[…truncated…]"
    receipts = entry.get("effect_receipts") or []
    receipt_caps = [
        str(r.get("capability_id") or "") for r in receipts if isinstance(r, dict)
    ]
    return {
        "turn": entry.get("turn"),
        "mode_id": entry.get("mode_id"),
        "query": entry.get("query"),
        "response_text": response,
        "handoff_to": entry.get("handoff_to"),
        "transitioned_to": entry.get("transitioned_to"),
        "receipt_capability_ids": receipt_caps,
    }