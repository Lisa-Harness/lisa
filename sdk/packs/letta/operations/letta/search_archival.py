"""Direct Letta archival memory search."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult
from sdk.packs.letta.utils.client import LettaClient, LettaError
from sdk.packs.letta.utils.identity_map import IdentityMap, identity_key

logger = logging.getLogger(__name__)


class Parameters(BaseModel):
    query: str = Field(..., description="Free-text query against Letta archival memory.")
    limit: int = Field(default=10, ge=1, le=50, description="Max passages to return.")


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Search Letta archival memory for the mapped agent."""
    query = str(ctx.params.get("query") or ctx.query or "").strip()
    if not query:
        return OperationResult(output_text="⛔ No query for Letta archival search.")

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
            output_text="⛔ No Letta agent mapped for this identity.",
            output_json={"error": "no_mapped_agent"},
        )

    limit = int(ctx.params.get("limit", 10) or 10)
    base_url = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
    api_key = os.getenv("LETTA_API_KEY", "")
    async with LettaClient(base_url=base_url, api_key=api_key) as client:
        try:
            passages = await client.search_passages(entry.agent_id, query=query)
        except LettaError as exc:
            return OperationResult(
                output_text=f"⛔ Letta archival search failed: {exc}",
                output_json={"error": str(exc), "agent_id": entry.agent_id},
            )

    if not isinstance(passages, list):
        passages = []
    passages = passages[:limit]
    lines = [f"Letta archival memory ({len(passages)} passages):"]
    for p in passages:
        if isinstance(p, dict):
            lines.append(f"- [{p.get('id', '?')}] {str(p.get('text', ''))[:200]}")
    return OperationResult(
        output_text="\n".join(lines),
        output_json={"agent_id": entry.agent_id, "passages": passages},
    )