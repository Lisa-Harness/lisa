"""Send a message to the mapped Letta agent and return its response."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult
from sdk.packs.letta.utils.client import LettaClient, LettaError
from sdk.packs.letta.utils.identity_map import IdentityMap, identity_key

logger = logging.getLogger(__name__)


class Parameters(BaseModel):
    message: str = Field(
        default="",
        description="Message to send. Defaults to the current user query.",
    )
    agent_id: str = Field(
        default="",
        description="Optional explicit Letta agent id. Defaults to the mapped agent for this identity.",
    )


def _resolve_agent_id(ctx: OperationExecutionContext, explicit: str) -> str | None:
    if explicit:
        return explicit
    data_dir = Path(os.getenv("LISA_DATA_DIR", ""))
    identity = IdentityMap(data_dir)
    key = identity_key(
        workspace_id=ctx.workspace_id,
        user_slug=ctx.user_slug,
        surface_id=ctx.surface_id,
        conversation_id=ctx.conversation_id,
    )
    entry = identity.get(key)
    return entry.agent_id if entry else None


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Delegate the turn to the mapped Letta agent."""
    message = str(ctx.params.get("message") or ctx.query or "").strip()
    if not message:
        return OperationResult(output_text="⛔ No message to send to Letta.")

    agent_id = _resolve_agent_id(ctx, str(ctx.params.get("agent_id") or ""))
    if not agent_id:
        return OperationResult(
            output_text="⛔ No Letta agent mapped for this identity. Run letta_ensure_agent first.",
            output_json={"error": "no_mapped_agent"},
        )

    base_url = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
    api_key = os.getenv("LETTA_API_KEY", "")
    async with LettaClient(base_url=base_url, api_key=api_key) as client:
        try:
            response = await client.send_message(agent_id, message=message)
        except LettaError as exc:
            return OperationResult(
                output_text=f"⛔ Letta send_message failed: {exc}",
                output_json={"error": str(exc), "agent_id": agent_id},
            )

    messages = response.get("messages") if isinstance(response, dict) else None
    assistant_text = ""
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("message_type") or "") == "assistant_message":
                assistant_text = str(msg.get("content") or "")
                break
    if not assistant_text:
        assistant_text = json.dumps(response, ensure_ascii=False)[:2000]

    return OperationResult(
        output_text=assistant_text,
        output_json={
            "agent_id": agent_id,
            "stop_reason": response.get("stop_reason") if isinstance(response, dict) else None,
            "usage": response.get("usage") if isinstance(response, dict) else None,
            "raw_messages": messages if isinstance(messages, list) else [],
        },
    )