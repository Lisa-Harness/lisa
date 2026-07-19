"""Trigger Letta message compaction for the mapped agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from kernel.models import OperationExecutionContext, OperationResult
from sdk.packs.letta.utils.client import LettaClient, LettaError
from sdk.packs.letta.utils.identity_map import IdentityMap, identity_key

logger = logging.getLogger(__name__)


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Call Letta summarize/compact for the mapped agent."""
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

    base_url = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
    api_key = os.getenv("LETTA_API_KEY", "")
    async with LettaClient(base_url=base_url, api_key=api_key) as client:
        try:
            result = await client.summarize(entry.agent_id)
        except LettaError as exc:
            return OperationResult(
                output_text=f"⛔ Letta compaction failed: {exc}",
                output_json={"error": str(exc), "agent_id": entry.agent_id},
            )
        except Exception as exc:  # pylint: disable=broad-except
            return OperationResult(
                output_text=f"⛔ Letta is not reachable at {base_url}: {type(exc).__name__}: {exc}",
                output_json={"error": str(exc), "error_type": type(exc).__name__, "base_url": base_url},
            )

    from datetime import UTC, datetime
    identity.update(
        key,
        last_compact_turn=entry.last_compact_turn + 1,
        last_compact_at=datetime.now(UTC).isoformat(),
    )

    before = result.get("num_messages_before") if isinstance(result, dict) else None
    after = result.get("num_messages_after") if isinstance(result, dict) else None
    return OperationResult(
        output_text=f"✅ Letta agent {entry.agent_id} compacted ({before} → {after} messages).",
        output_json={
            "agent_id": entry.agent_id,
            "num_messages_before": before,
            "num_messages_after": after,
            "summary": result.get("summary") if isinstance(result, dict) else None,
        },
    )