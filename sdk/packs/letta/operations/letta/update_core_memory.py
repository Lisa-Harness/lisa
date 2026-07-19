"""Update a pinned Letta core memory block for the mapped agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult
from sdk.packs.letta.utils.client import LettaClient, LettaError
from sdk.packs.letta.utils.identity_map import IdentityMap, identity_key

logger = logging.getLogger(__name__)


class Parameters(BaseModel):
    block_id: str = Field(
        default="",
        description="Letta block id to update. If empty, the operation looks up a mapped block by label.",
    )
    block_label: str = Field(
        default="",
        description="Label of the block to update (e.g. 'human', 'persona'). Used when block_id is empty.",
    )
    value: str = Field(..., description="New block value.")


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Update one Letta core memory block."""
    block_id = str(ctx.params.get("block_id") or "")
    block_label = str(ctx.params.get("block_label") or "")
    value = str(ctx.params.get("value") or "")

    if not value:
        return OperationResult(output_text="⛔ No block value provided.")

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

    if not block_id:
        block_id = entry.core_block_ids.get(block_label, "")
    if not block_id:
        return OperationResult(
            output_text=f"⛔ No core block mapped for label {block_label!r}. "
            "Use letta_ensure_agent first or pass an explicit block_id.",
            output_json={"error": "no_block_id", "label": block_label},
        )

    base_url = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
    api_key = os.getenv("LETTA_API_KEY", "")
    async with LettaClient(base_url=base_url, api_key=api_key) as client:
        try:
            updated = await client.update_block(block_id, value=value)
        except LettaError as exc:
            return OperationResult(
                output_text=f"⛔ Letta block update failed: {exc}",
                output_json={"error": str(exc), "block_id": block_id},
            )

    return OperationResult(
        output_text=f"✅ Letta core block {block_id} updated.",
        output_json={"block_id": block_id, "updated": True, "block": updated},
    )