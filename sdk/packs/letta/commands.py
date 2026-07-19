"""Letta-pack slash-command handlers.

Operator-facing commands for inspecting and driving the Letta
integration. These handlers live in the pack and are discovered through
the ``lisa.extensions.host_commands`` entry-point group. They delegate
to the Letta operations or read the identity-map sidecar.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernel.host_commands import HostCommandSpec

if TYPE_CHECKING:
    from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)

__all__ = [
    "get_host_commands",
    "handle_letta_status",
    "handle_letta_agent",
    "handle_letta_sync",
    "handle_letta_compact",
]


def get_host_commands() -> list[HostCommandSpec]:
    """Register the Letta pack's operator slash commands.

    These are admin/introspection subcommands only. The bare ``/letta``
    prefix is deliberately NOT registered: ``/letta <query>`` must fall
    through host-command dispatch so ``kernel.ipsar.parse_route`` can
    match the ``letta`` mode's ``command_pattern`` and delegate the turn
    to the mapped Letta agent. Registering a bare ``/letta`` here would
    shadow that delegation (it would ``startswith``-match every
    ``/letta ...`` line and return a status panel instead of routing).

    Order matters: the host dispatcher matches prefixes with
    ``str.startswith``, so longer/more-specific prefixes are listed
    BEFORE shorter ones as a defensive habit, though with no bare
    ``/letta`` registered there is no shadowing risk today.
    """
    return [
        HostCommandSpec(
            prefix="/letta status",
            handler=handle_letta_status,
            help_text="Show Letta connection status, identity map size, and env config.",
        ),
        HostCommandSpec(
            prefix="/letta compact",
            handler=handle_letta_compact,
            help_text="Run letta_compact_agent for the current identity.",
        ),
        HostCommandSpec(
            prefix="/letta agent",
            handler=handle_letta_agent,
            help_text="Show the Letta agent mapped to the current Lisa identity.",
        ),
        HostCommandSpec(
            prefix="/letta sync",
            handler=handle_letta_sync,
            help_text="Run letta_sync_memory for the current identity.",
        ),
    ]


def _identity_map(runtime: HostRuntime) -> Any:
    from sdk.packs.letta.utils.identity_map import IdentityMap

    return IdentityMap(Path(runtime.data_dir))


def _status_line(label: str, value: Any) -> str:
    return f"- {label}: {value}"


async def handle_letta_status(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Report Letta adapter configuration and identity-map size."""
    base_url = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
    has_key = "yes" if os.getenv("LETTA_API_KEY") else "no (local/unauth)"
    default_name = os.getenv("LETTA_DEFAULT_AGENT_NAME", "lisa-agent")
    identity = _identity_map(runtime)
    entries = identity.all_entries()
    lines = [
        "## Letta Status",
        "",
        _status_line("base_url", base_url),
        _status_line("api_key_set", has_key),
        _status_line("default_agent_name", default_name),
        _status_line("mapped_agents", len(entries)),
    ]
    if entries:
        lines.append("")
        lines.append("Mapped identities:")
        for key, entry in list(entries.items())[:10]:
            lines.append(f"- `{key}` → agent `{entry.agent_id}` ({entry.agent_name})")
    return "\n".join(lines)


async def handle_letta_agent(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Show the mapped Letta agent for the current identity, if any."""
    from sdk.packs.letta.utils.identity_map import identity_key

    identity = _identity_map(runtime)
    key = identity_key(
        workspace_id="default",
        user_slug=str(getattr(runtime.config, "user_slug", "default") or "default"),
        surface_id=str(getattr(runtime.config, "surface_id", "") or ""),
    )
    entry = identity.get(key)
    if entry is None:
        return f"No Letta agent mapped for identity `{key}`. Run `/letta sync` or `/letta <query>` in letta mode to create one."
    lines = [
        "## Letta Agent",
        "",
        _status_line("identity_key", key),
        _status_line("agent_id", entry.agent_id),
        _status_line("agent_name", entry.agent_name),
        _status_line("last_sync_turn", entry.last_sync_turn),
        _status_line("last_sync_at", entry.last_sync_at or "(never)"),
        _status_line("last_compact_turn", entry.last_compact_turn),
        _status_line("last_compact_at", entry.last_compact_at or "(never)"),
    ]
    if entry.core_block_ids:
        lines.append("")
        lines.append("Core blocks:")
        for label, block_id in entry.core_block_ids.items():
            lines.append(f"- `{label}` → `{block_id}`")
    return "\n".join(lines)


async def handle_letta_sync(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Run the letta_sync_memory operation for the current identity."""
    from kernel.models import OperationExecutionContext
    from kernel.operations.registry import execute_operation

    ctx = OperationExecutionContext(
        workspace_id="default",
        user_slug=str(getattr(runtime.config, "user_slug", "default") or "default"),
        surface_id=str(getattr(runtime.config, "surface_id", "") or ""),
        mode_id="letta",
        query=stripped or "",
        execute_tool_fn=getattr(runtime, "execute_tool_fn", None),
    )
    result = await execute_operation("letta_sync_memory", ctx)
    return result.output_text or "sync completed (no output)"


async def handle_letta_compact(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Run the letta_compact_agent operation for the current identity."""
    from kernel.models import OperationExecutionContext
    from kernel.operations.registry import execute_operation

    ctx = OperationExecutionContext(
        workspace_id="default",
        user_slug=str(getattr(runtime.config, "user_slug", "default") or "default"),
        surface_id=str(getattr(runtime.config, "surface_id", "") or ""),
        mode_id="letta",
    )
    result = await execute_operation("letta_compact_agent", ctx)
    return result.output_text or "compaction completed (no output)"