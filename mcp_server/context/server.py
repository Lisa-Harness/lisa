"""Lisa Context MCP server — query-aware context brokerage.

Owns the context broker logic only. Tools:

- ``broker_context`` — query-aware context brokerage

The former memory tools (``add_memory`` / ``search_memory`` /
``list_memories`` / ``get_surface_context``) moved to the dedicated
``lisa_memory`` server — the minimal built-in memory backend. ``lisa_context``
no longer owns a SQLite store; it operates on a :class:`~kernel.models.UnifiedState`
payload supplied by the caller.

``--readonly`` is accepted for CLI-contract consistency but is inert: the
broker is stateless and read-only by design. ``--data-dir`` is accepted for
contract consistency; the broker does not persist anything today.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from kernel.context.context_broker import (
    build_ambient_facts,
    render_context_packet_markdown,
    select_ranked_context,
)
from kernel.models import (
    ContextBrokerResponse,
    UnifiedState,
)

__all__ = ["BrokerContextRequest", "main", "mcp"]

logger = logging.getLogger(__name__)

mcp = FastMCP("Lisa Context Server")
_SERVER_TIMEZONE = "UTC"


class BrokerContextRequest(BaseModel):
    """Typed MCP request schema for ``broker_context``.

    Mirrors the kernel-level :class:`ContextBrokerRequest` fields plus the
    ``state`` payload needed to build a :class:`UnifiedState`. FastMCP uses
    this model's fields as the tool's input schema — each field becomes a
    top-level tool argument.
    """

    query: str
    mode_id: str = "default"
    stage_id: str = ""
    surface: dict[str, Any] = Field(default_factory=dict)
    user_slug: str = "default"
    workspace_id: str = "default"
    connectors: list[str] = Field(default_factory=list)
    context_budget: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    # Pack-scoped evidence boundary: the caller passes the active mode's
    # provider pack and explicitly allowed packs so the broker can filter
    # extension evidence to only what the mode should see.
    mode_provider_id: str = ""
    context_evidence_packs: list[str] = Field(default_factory=list)


def _state_from_request(request: BrokerContextRequest) -> UnifiedState:
    """Build a ``UnifiedState`` from the request's ``state`` field.

    Fails closed: if ``state`` is present but malformed (not a valid
    ``UnifiedState``), a ``ValidationError`` propagates to the caller. If
    ``state`` is absent or empty, an empty state is used explicitly.
    """
    state_data = request.state
    if not state_data:
        return UnifiedState(compiled_at="", context_markdown="")
    return UnifiedState.model_validate(state_data)


@mcp.tool()
async def broker_context(
    query: str,
    mode_id: str = "default",
    stage_id: str = "",
    surface: dict[str, Any] | None = None,
    user_slug: str = "default",
    workspace_id: str = "default",
    connectors: list[str] | None = None,
    context_budget: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    mode_provider_id: str = "",
    context_evidence_packs: list[str] | None = None,
) -> str:
    """Return a brokered context packet and markdown for the current turn.

    Each argument maps to a field on :class:`BrokerContextRequest`. The
    ``state`` argument carries the caller-supplied ``UnifiedState`` snapshot
    used for evidence selection. Malformed ``state`` raises a validation
    error instead of silently degrading to an empty state.

    ``mode_provider_id`` and ``context_evidence_packs`` carry the active
    mode's pack-scoped evidence boundary so the broker can filter extension
    evidence to only what the mode should see.
    """
    request = BrokerContextRequest(
        query=query,
        mode_id=mode_id,
        stage_id=stage_id,
        surface=surface or {},
        user_slug=user_slug,
        workspace_id=workspace_id,
        connectors=connectors or [],
        context_budget=context_budget or {},
        state=state or {},
        mode_provider_id=mode_provider_id or "",
        context_evidence_packs=context_evidence_packs or [],
    )
    unified_state = _state_from_request(request)
    packet = select_ranked_context(
        state=unified_state,
        query=request.query,
        mode_id=request.mode_id,
        stage_id=request.stage_id,
        connectors=request.connectors or None,
        mode_provider_id=request.mode_provider_id,
        context_evidence_packs=request.context_evidence_packs or None,
    )
    response = ContextBrokerResponse(
        packet=packet,
        packet_markdown=(
            "## Ambient Facts\n\n"
            + build_ambient_facts(timezone=_SERVER_TIMEZONE, host_name="lisa-context")
            + "\n\n"
            + render_context_packet_markdown(packet)
        ),
        context_tag=request.context_budget.get("context_tag", ""),
    )
    return response.model_dump_json()


def main() -> None:
    """Entry point for the lisa-context MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper. The
    broker is stateless, so ``--data-dir`` is accepted for contract
    consistency but unused today.
    """
    from mcp_server.config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-context", supports_readonly=True)
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="IANA timezone for ambient facts (for example, UTC or America/New_York).",
    )
    args = parser.parse_args()

    global _SERVER_TIMEZONE
    _SERVER_TIMEZONE = str(args.timezone or "UTC")

    log = configure_logging(args.log_level, server_name="lisa-context")
    data_dir = resolve_data_dir(args.data_dir)
    log.info(
        "lisa-context starting (data_dir=%s, readonly=%s, timezone=%s)",
        data_dir,
        args.readonly,
        _SERVER_TIMEZONE,
    )
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
