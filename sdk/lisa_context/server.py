"""Lisa Context MCP server — query-aware context brokerage.

Owns the context broker logic only. Tools:

- ``broker_context`` — query-aware context brokerage

The former memory tools (``add_memory`` / ``search_memory`` /
``list_memories`` / ``get_surface_context``) moved to the dedicated
``lisa_memory`` server — the minimal built-in memory backend. ``lisa_context``
no longer owns a SQLite store; it operates on a :class:`~kernel.models.UnifiedState`
payload supplied by the caller.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from kernel.models import (
    ContextBrokerRequest,
    ContextBrokerResponse,
    UnifiedState,
)
from sdk.lisa_context.context_broker import (
    render_context_packet_markdown,
    select_ranked_context,
)

__all__ = ["main", "mcp"]

logger = logging.getLogger(__name__)

mcp = FastMCP("Lisa Context Server")


def _state_from_request(payload: dict[str, Any]) -> UnifiedState:
    state_payload = payload.get("state")
    if isinstance(state_payload, UnifiedState):
        return state_payload
    if isinstance(state_payload, dict):
        return UnifiedState.model_validate(state_payload)
    return UnifiedState(compiled_at="", context_markdown="")


@mcp.tool()
async def broker_context(payload: dict[str, Any]) -> str:
    """Return a brokered context packet and markdown for the current turn."""
    request = ContextBrokerRequest.model_validate(payload)
    state = _state_from_request(payload)
    packet = select_ranked_context(
        state=state,
        query=request.query,
        mode_id=request.mode_id,
        stage_id=request.stage_id,
        connectors=request.connectors or None,
    )
    response = ContextBrokerResponse(
        packet=packet,
        packet_markdown=render_context_packet_markdown(packet),
        context_tag=request.context_budget.get("context_tag", ""),
    )
    return response.model_dump_json()


def main() -> None:
    """Entry point for the lisa-context MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper. The
    broker is stateless, so ``--data-dir`` is accepted for contract
    consistency but unused today.
    """
    from sdk.lisa_mcp_config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-context", supports_readonly=True)
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-context")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-context starting (data_dir=%s, readonly=%s)", data_dir, args.readonly)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

