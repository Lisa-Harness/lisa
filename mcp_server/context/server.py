"""Lisa Context MCP server — query-aware context brokerage.

Owns the context broker logic only. Tools:

- ``broker_context`` — query-aware context brokerage

The former memory tools (``add_memory`` / ``search_memory`` /
``list_memories`` / ``get_memory`` / ``delete_memory``) moved to the dedicated
``lisa_memory`` server — the minimal built-in memory backend. ``lisa_context``
no longer owns a SQLite store; it operates on a :class:`~kernel.models.UnifiedState`
payload supplied by the caller.

``--readonly`` is accepted for CLI-contract consistency but is inert: the
broker is stateless and read-only by design. ``--data-dir`` is accepted for
contract consistency; the broker does not persist anything today.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from kernel.models import (
    ContextBrokerResponse,
    UnifiedState,
)
from sdk.context.context_broker import (
    build_ambient_facts,
    render_context_packet_markdown,
    select_ranked_context,
)

__all__ = ["BrokerContextRequest", "ContextDeps", "main", "mcp", "set_deps_for_test"]

logger = logging.getLogger(__name__)


# ── Lifespan deps ───────────────────────────────────────────────────────────


@dataclass
class ContextDeps:
    """Server-scoped dependencies owned by the lifespan.

    Carries the IANA timezone for ambient-facts rendering. The broker is
    stateless, so this is the only per-server config.
    """

    timezone: str = "UTC"


# Test seam: populated by the lifespan in production, or by
# ``set_deps_for_test`` in tests (direct ``call_tool`` does not run the
# lifespan). This is an explicit seam, not a hidden production singleton.
_deps: ContextDeps | None = None


def set_deps_for_test(deps: ContextDeps) -> None:
    """Test-only seam: inject ``ContextDeps`` without spawning stdio.

    The lifespan owns this in production; tests bypass it because the
    SDK's direct ``call_tool`` path does not run the lifespan.
    """
    global _deps  # noqa: PLW0603 — test seam; lifespan owns this in production
    _deps = deps


def _resolve_deps(ctx: Context[Any, Any] | None) -> ContextDeps:
    """Return the active deps, preferring lifespan context when available."""
    if ctx is not None:
        try:
            rc = ctx.request_context
            deps = rc.lifespan_context
            if isinstance(deps, ContextDeps):
                return deps
        except ValueError:
            pass  # direct call_tool path — no request_context
    if _deps is None:
        # Default to UTC for the test path when no seam was set.
        return ContextDeps()
    return _deps


# ── CLI config holder (populated by _init before run()) ────────────────────


@dataclass
class _ContextConfig:
    timezone: str = "UTC"


_cfg = _ContextConfig()


@asynccontextmanager
async def _lifespan(app: MCPServer[Any]) -> AsyncIterator[ContextDeps]:
    """Yield the resolved timezone for ambient-facts rendering."""
    yield ContextDeps(timezone=_cfg.timezone)


mcp = MCPServer("Lisa Context Server", lifespan=_lifespan)


class BrokerContextRequest(BaseModel):
    """Typed MCP request schema for ``broker_context``.

    Mirrors the kernel-level :class:`ContextBrokerRequest` fields plus the
    ``state`` payload needed to build a :class:`UnifiedState`. MCPServer uses
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
    ``UnifiedState``), a :class:`ToolError` propagates to the caller with
    the validation path so the host gets a usable message instead of an
    opaque internal error. If ``state`` is absent or empty, an empty state
    is used explicitly.
    """
    state_data = request.state
    if not state_data:
        return UnifiedState(compiled_at="", context_markdown="")
    try:
        return UnifiedState.model_validate(state_data)
    except Exception as exc:
        raise ToolError(f"malformed state: {exc}") from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
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
    ctx: Context[ContextDeps, Any] | None = None,
) -> str:
    """Return a brokered context packet and markdown for the current turn.

    Each argument maps to a field on :class:`BrokerContextRequest`. The
    ``state`` argument carries the caller-supplied ``UnifiedState`` snapshot
    used for evidence selection. Malformed ``state`` raises a
    :class:`ToolError` instead of silently degrading to an empty state.

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
    deps = _resolve_deps(ctx)
    response = ContextBrokerResponse(
        packet=packet,
        packet_markdown=(
            "## Ambient Facts\n\n"
            + build_ambient_facts(timezone=deps.timezone, host_name="lisa-context")
            + "\n\n"
            + render_context_packet_markdown(packet)
        ),
        context_tag=request.context_budget.get("context_tag", ""),
    )
    return response.model_dump_json()


def _add_context_args(parser: Any) -> None:
    """Add ``--timezone`` to the standard parser."""
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="IANA timezone for ambient facts (for example, UTC or America/New_York).",
    )


def _init(args: Any) -> None:
    """Populate ``_cfg`` from parsed args so the lifespan can read the timezone."""
    _cfg.timezone = str(getattr(args, "timezone", None) or "UTC")
    # Also populate the test seam so the server is usable immediately after
    # init (the lifespan only runs during a real stdio session).
    set_deps_for_test(ContextDeps(timezone=_cfg.timezone))


def main() -> None:
    """Entry point for the lisa-context MCP server (stdio transport).

    Parses the standard CLI contract via the shared ``run()`` helper. The
    broker is stateless, so ``--data-dir`` is accepted for contract
    consistency but unused today. ``--timezone`` selects the IANA timezone
    for ambient-facts rendering.
    """
    from mcp_server.config.cli import run

    run("lisa-context", mcp, _init, supports_readonly=True, add_args=_add_context_args)


if __name__ == "__main__":
    main()
