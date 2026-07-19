"""Resolve or create the Letta agent for the current Lisa identity."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult
from sdk.packs.letta.utils.client import LettaClient, LettaError
from sdk.packs.letta.utils.identity_map import IdentityEntry, IdentityMap, identity_key

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_NAME = "lisa-agent"


class Parameters(BaseModel):
    agent_name: str = Field(
        default_factory=lambda: os.getenv("LETTA_DEFAULT_AGENT_NAME", _DEFAULT_AGENT_NAME),
        description="Name for a newly created Letta agent. Ignored when an agent is already mapped.",
    )


def _identity_map(runtime_data_dir: Any) -> IdentityMap:
    # Operations resolve the data dir from the environment; the host wires
    # the same data dir when launching the adapter MCP server.
    data_dir = os.getenv("LISA_DATA_DIR", str(runtime_data_dir or ""))
    return IdentityMap(Path(data_dir))


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Return the Letta agent_id for this Lisa identity, creating one if needed."""
    agent_name = str(ctx.params.get("agent_name") or os.getenv(
        "LETTA_DEFAULT_AGENT_NAME", _DEFAULT_AGENT_NAME
    ))
    key = identity_key(
        workspace_id=ctx.workspace_id,
        user_slug=ctx.user_slug,
        surface_id=ctx.surface_id,
        conversation_id=ctx.conversation_id,
    )
    identity = _identity_map(ctx.workspace_id)

    # If LETTA_AGENT_ID is set, use it directly — no lookup or creation.
    explicit_agent_id = os.getenv("LETTA_AGENT_ID", "").strip()
    if explicit_agent_id:
        identity.upsert(
            key,
            IdentityEntry(agent_id=explicit_agent_id, agent_name=agent_name),
        )
        return OperationResult(
            output_text=f"Letta agent resolved (explicit): {explicit_agent_id}",
            output_json={"agent_id": explicit_agent_id, "created": False, "explicit": True, "key": key},
        )

    existing = identity.get(key)
    if existing is not None and existing.agent_id:
        return OperationResult(
            output_text=f"Letta agent resolved: {existing.agent_id}",
            output_json={"agent_id": existing.agent_id, "created": False, "key": key},
        )

    base_url = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
    api_key = os.getenv("LETTA_API_KEY", "")
    async with LettaClient(base_url=base_url, api_key=api_key) as client:
        try:
            # Look up an existing agent by name before creating — Letta
            # Cloud's free plan allows only 3 agents, and creating a
            # duplicate may 402.
            found = await client.find_agent_by_name(agent_name)
            if found is not None and found.get("id"):
                agent_id = str(found["id"])
                identity.upsert(
                    key,
                    IdentityEntry(
                        agent_id=agent_id,
                        agent_name=str(found.get("name") or agent_name),
                    ),
                )
                return OperationResult(
                    output_text=f"✅ Letta agent resolved (existing): {agent_id}",
                    output_json={"agent_id": agent_id, "created": False, "found_by_name": True, "key": key},
                )

            agent = await client.create_agent(name=agent_name)
        except LettaError as exc:
            return OperationResult(
                output_text=f"⛔ Letta agent creation failed: {exc}",
                output_json={"error": str(exc)},
            )
        except Exception as exc:  # pylint: disable=broad-except
            # Catch httpx.ConnectError, httpx.TimeoutException, and any
            # transport-level failure so the operation surfaces a clear
            # "Letta not reachable" message instead of crashing the turn.
            return OperationResult(
                output_text=(
                    f"⛔ Letta is not reachable at {base_url}: "
                    f"{type(exc).__name__}: {exc}"
                ),
                output_json={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "base_url": base_url,
                },
            )
    agent_id = str(agent.get("id") or "")
    if not agent_id:
        return OperationResult(
            output_text="⛔ Letta create_agent returned no agent id",
            output_json={"error": "no agent id"},
        )
    identity.upsert(
        key,
        IdentityEntry(agent_id=agent_id, agent_name=str(agent.get("name") or agent_name)),
    )
    return OperationResult(
        output_text=f"✅ Letta agent created: {agent_id}",
        output_json={"agent_id": agent_id, "created": True, "key": key},
    )