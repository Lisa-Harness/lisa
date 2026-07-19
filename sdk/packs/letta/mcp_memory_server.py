"""Letta memory adapter MCP server.

Exposes the three generic Lisa memory tools (``add_memory``,
``search_memory``, ``list_memories``) over Letta archival memory. The
host wires this server behind the generic ``memory.*`` synthetic
aliases when ``lisa-pack-letta`` is selected as the ``memory``
capability owner (see ``mcp_servers.yaml``).

Resolution policy:
- Each Lisa identity (``workspace_id`` + ``user_slug`` + ``surface_id``)
  maps to one Letta agent via :class:`IdentityMap`.
- On first use, if no agent is mapped, a new Letta agent is created
  (auto-create) so ``synthesize_and_store_learning`` does not fail on a
  fresh user. A strict mode may be added later via env.
- All durable facts are stored as Letta archival memory passages.

Responses are returned as Letta-shaped JSON; the alias-boundary
``result_maps`` normalize them to canonical Lisa ``MemoryRecord`` JSON.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from sdk.packs.letta.utils.client import LettaClient, LettaError
from sdk.packs.letta.utils.identity_map import IdentityEntry, IdentityMap, identity_key

__all__ = ["init_server", "main", "mcp"]

logger = logging.getLogger(__name__)

mcp = FastMCP("Lisa Letta Memory Server")

_client: LettaClient | None = None
_identity: IdentityMap | None = None
_default_agent_name: str = "lisa-agent"


def init_server(
    data_dir: Path,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    default_agent_name: str | None = None,
    transport: Any | None = None,
) -> None:
    """Initialize the module-level client and identity map.

    Called by :func:`main` after arg parsing. Tests may call this directly
    with a ``tmp_path`` and an injected ``transport`` (e.g.
    ``httpx.MockTransport``) to pin behavior without spawning the server.
    """
    global _client, _identity, _default_agent_name
    _identity = IdentityMap(data_dir)
    resolved_name = default_agent_name or os.getenv(
        "LETTA_DEFAULT_AGENT_NAME", "lisa-agent"
    )
    _default_agent_name = str(resolved_name)
    _client = LettaClient(
        base_url=str(base_url or os.getenv("LETTA_BASE_URL", "http://localhost:8283")),
        api_key=api_key if api_key is not None else os.getenv("LETTA_API_KEY", ""),
        transport=transport,
    )


def _require() -> tuple[LettaClient, IdentityMap]:
    if _client is None or _identity is None:
        raise RuntimeError(
            "lisa-letta-memory is not initialized. "
            "Call init_server(data_dir) from main() before serving tools."
        )
    return _client, _identity


async def _resolve_agent(
    *,
    workspace_id: str,
    user_slug: str,
    surface_id: str,
) -> str:
    """Return the mapped Letta agent id, creating one if needed.

    Auto-create keeps the generic ``memory.add`` contract simple: a fresh
    user can store a learning on their first turn without a separate
    provisioning step. Failures (API down, auth error) surface as a
    ``LettaError`` that the tool handlers wrap in an error envelope.
    """
    client, identity = _require()
    key = identity_key(
        workspace_id=workspace_id,
        user_slug=user_slug,
        surface_id=surface_id,
    )
    entry = identity.get(key)
    if entry is not None and entry.agent_id:
        return entry.agent_id

    agent = await client.create_agent(name=_default_agent_name)
    agent_id = str(agent.get("id") or "")
    if not agent_id:
        raise LettaError("Letta create_agent returned no agent id")
    identity.upsert(
        key,
        IdentityEntry(
            agent_id=agent_id,
            agent_name=str(agent.get("name") or _default_agent_name),
        ),
    )
    return agent_id


@mcp.tool()
async def add_memory(
    user_slug: str,
    content: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    memory_tier: str = "long_term",
    operation_id: str = "letta.add",
    instruction_id: str = "letta.v1",
    content_json: dict[str, Any] | None = None,
    salience: float = 0.0,
) -> str:
    """Store a memory snippet as a Letta archival memory passage.

    Mirrors the first-party ``lisa_memory.add_memory`` signature so the
    generic alias needs no backend-specific arg translation. Returns a
    JSON object describing the created passage; the alias result map
    normalizes it to a Lisa ``MemoryRecord``.
    """
    import json

    client, _ = _require()
    try:
        agent_id = await _resolve_agent(
            workspace_id=workspace_id,
            user_slug=user_slug,
            surface_id=surface_id,
        )
        metadata = {
            "memory_scope": memory_scope,
            "workspace_id": workspace_id,
            "surface_id": surface_id,
            "memory_tier": memory_tier,
            "operation_id": operation_id,
            "instruction_id": instruction_id,
            "salience": salience,
            "content_json": content_json or {},
        }
        passage = await client.create_passage(agent_id, text=content)
        if isinstance(passage, dict):
            passage.setdefault("metadata", metadata)
        return json.dumps(passage or {"id": "", "text": content})
    except LettaError as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def search_memory(
    query: str,
    user_slug: str = "default",
    workspace_id: str = "default",
    surface_id: str = "",
    limit: int = 10,
) -> str:
    """Search Letta archival memory for the mapped agent.

    Returns a JSON array of passages. The alias result map normalizes it
    to a bare array of ``MemoryRecord`` dicts for ``recall_memory``.
    """
    import json

    client, _ = _require()
    try:
        agent_id = await _resolve_agent(
            workspace_id=workspace_id,
            user_slug=user_slug,
            surface_id=surface_id,
        )
        results = await client.search_passages(agent_id, query=query)
        if isinstance(results, list):
            return json.dumps(results[: max(1, int(limit or 10))])
        return json.dumps([])
    except LettaError as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def list_memories(
    user_slug: str = "default",
    workspace_id: str = "default",
    surface_id: str = "",
    limit: int = 20,
) -> str:
    """List recent Letta archival memory passages for the mapped agent."""
    import json

    client, _ = _require()
    try:
        agent_id = await _resolve_agent(
            workspace_id=workspace_id,
            user_slug=user_slug,
            surface_id=surface_id,
        )
        results = await client.list_passages(agent_id)
        if isinstance(results, list):
            return json.dumps(results[: max(1, int(limit or 20))])
        return json.dumps([])
    except LettaError as exc:
        return json.dumps({"error": str(exc)})


def main() -> None:
    """Console-script entry point for the ``lisa-letta-memory`` server."""
    from mcp_server.config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-letta-memory")
    args = parser.parse_args()
    log = configure_logging(args.log_level, server_name="lisa-letta-memory")
    data_dir = resolve_data_dir(args.data_dir)

    log.info(
        "lisa-letta-memory starting (data_dir=%s, base_url=%s)",
        data_dir,
        os.getenv("LETTA_BASE_URL", "http://localhost:8283"),
    )
    init_server(data_dir)
    mcp.run()