"""Push a single passage into the mapped Letta agent's archival memory.

Unlike ``letta_sync_memory`` (which pulls many Lisa memories and bulk-pushes
them), this operation writes one explicit user-supplied passage. It is the
backend for the ``/letta remember <text>`` slash command: a deliberate,
operator-curated "remember this" action.
"""

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
    text: str = Field(
        default="",
        description="Passage text to remember. Defaults to the current user query.",
    )


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Create one archival memory passage on the mapped Letta agent."""
    text = str(ctx.params.get("text") or ctx.query or "").strip()
    if not text:
        return OperationResult(output_text="⛔ Nothing to remember — no text provided.")

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
            output_text="⛔ No Letta agent mapped for this identity. Run `/letta sync` or a letta-mode turn to create one.",
            output_json={"error": "no_mapped_agent"},
        )

    base_url = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
    api_key = os.getenv("LETTA_API_KEY", "")
    async with LettaClient(base_url=base_url, api_key=api_key) as client:
        try:
            passage = await client.create_passage(entry.agent_id, text=text)
        except LettaError as exc:
            return OperationResult(
                output_text=f"⛔ Letta remember failed: {exc}",
                output_json={"error": str(exc), "agent_id": entry.agent_id},
            )
        except Exception as exc:  # pylint: disable=broad-except
            return OperationResult(
                output_text=f"⛔ Letta is not reachable at {base_url}: {type(exc).__name__}: {exc}",
                output_json={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "base_url": base_url,
                },
            )

    passage_id = ""
    if isinstance(passage, dict):
        passage_id = str(passage.get("id") or passage.get("passage_id") or "")
    return OperationResult(
        output_text=f"✅ Remembered by Letta agent {entry.agent_id}"
        + (f" (passage {passage_id})" if passage_id else "")
        + f": {text[:200]}",
        output_json={
            "agent_id": entry.agent_id,
            "passage_id": passage_id,
            "text": text,
        },
    )