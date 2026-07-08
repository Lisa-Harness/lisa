"""Lisa Search MCP server — web search tool.

Wraps ``kernel.web_search.search()`` as an MCP tool so external MCP clients
can perform Brave Search queries through the kernel.
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from lisa_sdk.lisa_search import web_search as _web_search

logger = logging.getLogger(__name__)

__all__ = ["main", "mcp"]


mcp = FastMCP("Lisa Search Server")


@mcp.tool()
async def web_search(query: str, count: int = 5) -> str:
    """Search the web using Brave Search.

    Returns a JSON array of results (name, url, snippet). Requires
    ``WEB_SEARCH_API_KEY`` to be configured in the kernel environment.
    """
    try:
        response = await _web_search.search(query)
    except RuntimeError as exc:
        return f"Web search not configured: {exc}"
    except Exception as exc:
        logger.warning("web_search failed: %s", exc)
        return f"Search error: {exc}"
    return response.model_dump_json()


@mcp.tool()
async def web_search_configured() -> str:
    """Return whether web search is configured (has an API key)."""
    return json.dumps({"configured": _web_search.is_configured()})


def main() -> None:
    """Entry point for the lisa-search MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper.
    ``WEB_SEARCH_API_KEY`` stays in env (behavioral config); ``--data-dir``
    is required for consistency but not used for path resolution today.
    """
    from lisa_sdk.lisa_mcp_config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-search")
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-search")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-search starting (data_dir=%s)", data_dir)

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
