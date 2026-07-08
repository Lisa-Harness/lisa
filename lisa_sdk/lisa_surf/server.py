"""Lisa Surf MCP server — surf workflow tools.

Wraps ``kernel.surf_workflow`` as MCP tools so external MCP clients can
manage iterative browsing sessions:

- ``surf_url`` — create or update a surf session for a URL
- ``flush_session`` — delete the current surf session
- ``get_session_state`` — inspect the current surf session state
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from kernel.models import SurfaceContext
from lisa_sdk.lisa_surf.surf_workflow import (
    build_surf_session_summary,
    create_surf_session,
    delete_surf_session,
    load_surf_session,
)

logger = logging.getLogger(__name__)

__all__ = ["mcp", "main"]


mcp = FastMCP("Lisa Surf Server")


@mcp.tool()
async def surf_url(
    workspace_id: str,
    surface_id: str,
    url: str,
    surface_type: str = "personal_dm",
    memory_scope: str = "personal",
) -> str:
    """Create (or refresh) a surf session for one URL.

    Returns the session ID and a summary. The actual page fetch is performed
    by the host's browser runtime; this tool establishes the session state.
    """
    surface = SurfaceContext(
        surface_type=surface_type,  # type: ignore[arg-type]
        surface_id=surface_id,
        memory_scope=memory_scope,  # type: ignore[arg-type]
    )
    session = create_surf_session(workspace_id=workspace_id, surface_context=surface)
    session.current_page_url = url
    session.last_action = "surf_url"
    return json.dumps(
        {
            "session_id": session.session_id,
            "url": url,
            "summary": build_surf_session_summary(session),
        },
        default=str,
    )


@mcp.tool()
async def flush_session(
    workspace_id: str,
    surface_id: str,
    surface_type: str = "personal_dm",
    memory_scope: str = "personal",
    blob_io: Any = None,
) -> str:
    """Delete the current surf session for the given surface."""
    surface = SurfaceContext(
        surface_type=surface_type,  # type: ignore[arg-type]
        surface_id=surface_id,
        memory_scope=memory_scope,  # type: ignore[arg-type]
    )
    deleted = delete_surf_session(
        workspace_id=workspace_id,
        surface_context=surface,
        blob_io=blob_io,
    )
    return json.dumps({"deleted": deleted, "surface_id": surface_id})


@mcp.tool()
async def get_session_state(
    workspace_id: str,
    surface_id: str,
    surface_type: str = "personal_dm",
    memory_scope: str = "personal",
    blob_io: Any = None,
) -> str:
    """Return the current surf session state as JSON, or 'no session'."""
    surface = SurfaceContext(
        surface_type=surface_type,  # type: ignore[arg-type]
        surface_id=surface_id,
        memory_scope=memory_scope,  # type: ignore[arg-type]
    )
    session = load_surf_session(
        workspace_id=workspace_id,
        surface_context=surface,
        blob_io=blob_io,
    )
    if session is None:
        return json.dumps({"session": None, "reason": "no session or expired"})
    return json.dumps(
        {
            "session_id": session.session_id,
            "current_page_url": session.current_page_url,
            "current_page_title": session.current_page_title,
            "summary": session.summary,
            "pages": len(session.pages),
            "expires_at": session.expires_at,
        },
        default=str,
    )


def main() -> None:
    """Entry point for the lisa-surf MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper. No
    path config is needed today, but ``--data-dir`` is required for
    consistency across all first-party servers.
    """
    from lisa_sdk.lisa_mcp_config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-surf")
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-surf")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-surf starting (data_dir=%s)", data_dir)

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()