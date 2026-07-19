"""Sync selected Lisa memories into the mapped Letta agent's archival memory."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext
from sdk.packs.letta.utils.client import LettaClient, LettaError
from sdk.packs.letta.utils.identity_map import IdentityMap, identity_key

logger = logging.getLogger(__name__)


class Parameters(BaseModel):
    query: str = Field(
        default="",
        description="Optional query to select which Lisa memories to sync. Empty syncs recent memories.",
    )
    limit: int = Field(default=10, ge=1, le=50, description="Max memories to sync.")


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Pull Lisa memories via memory.search and push them into Letta archival memory."""
    if not callable(ctx.execute_tool_fn):
        return OperationResult(output_text="⛔ Tool execution not injected.")

    identity = IdentityMap(Path(os.getenv("LISA_DATA_DIR", "")))
    key = identity_key(
        workspace_id=ctx.workspace_id,
        user_slug=ctx.user_slug,
        surface_id=ctx.surface_id,
        conversation_id=ctx.conversation_id,
    )
    entry = identity.get(key)
    if entry is None or not entry.agent_id:
        return OperationResult(
            output_text="⛔ No Letta agent mapped for this identity. Run letta_ensure_agent first.",
            output_json={"error": "no_mapped_agent"},
        )

    query = str(ctx.params.get("query") or ctx.query or "")
    limit = int(ctx.params.get("limit", 10) or 10)

    # Pull from Lisa memory via the generic alias (routes to whichever
    # backend is the default — which may be Letta itself). The point of
    # sync is to ensure high-salience Lisa-distilled facts are present as
    # Letta archival passages even when the default backend is local.
    tool_result = await ctx.execute_tool_fn(
        "memory.search",
        ToolExecutionContext(
            tool_id="memory.search",
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            query=query,
            parameters={"query": query, "user_slug": ctx.user_slug, "limit": limit},
        ),
    )
    if getattr(tool_result, "error", None):
        return OperationResult(output_text=f"⛔ memory.search failed: {tool_result.error}")

    try:
        records = json.loads(tool_result.output_text or "[]")
    except (json.JSONDecodeError, TypeError):
        records = []
    if isinstance(records, dict) and "results" in records:
        records = records["results"]
    if not isinstance(records, list):
        records = []

    base_url = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
    api_key = os.getenv("LETTA_API_KEY", "")
    synced = 0
    failures: list[str] = []
    async with LettaClient(base_url=base_url, api_key=api_key) as client:
        for record in records[:limit]:
            if not isinstance(record, dict):
                continue
            content = str(record.get("content") or "")
            if not content:
                continue
            try:
                await client.create_passage(entry.agent_id, text=content)
                synced += 1
            except LettaError as exc:
                failures.append(str(exc))

    from datetime import UTC, datetime
    current = identity.get(key)
    identity.update(
        key,
        last_sync_turn=(current.last_sync_turn + 1) if current else 0,
        last_sync_at=datetime.now(UTC).isoformat(),
    )

    summary = f"✅ Synced {synced} memory passage(s) to Letta agent {entry.agent_id}."
    if failures:
        summary += f" ⚠ {len(failures)} failure(s): {failures[0]}"
    return OperationResult(
        output_text=summary,
        output_json={
            "agent_id": entry.agent_id,
            "synced": synced,
            "failures": failures,
            "candidates": len(records),
        },
    )