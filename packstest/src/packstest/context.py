"""Fully-wired ``ToolExecutionContext`` factory for pack tools.

This is the generalized form of the ``_context()`` helper pattern proven in
pack unit tests: build a real ``ToolExecutionContext`` whose ``pack_config``
dict points at a temporary workspace root, then call the tool's ``execute()``
directly. Everything the tool may touch (artifact store, event bus, state
provider, data dir) is wired with deterministic in-memory doubles.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel.capability_binding import ExecutionFloorView
from kernel.models import ToolExecutionContext, ToolResult
from kernel.testing.stubs import NullEventBus, StaticStateProvider
from sdk.workproducts import WorkProductStore

from packstest.store import MemoryWorkProductStore

__all__ = [
    "DEFAULT_IDENTITY",
    "ToolContextFactory",
    "make_context",
    "make_workspace",
]

DEFAULT_IDENTITY = {
    "workspace_id": "default",
    "user_id": "test-user",
    "user_slug": "default",
    "surface_type": "personal_dm",
    "surface_id": "packstest",
    "memory_scope": "personal",
    "session_plan_digest": "packstest-digest",
}


@dataclass(slots=True)
class ToolContextFactory:
    """Builds one or many execution contexts over shared doubles.

    Attributes:
        root: Workspace root directory (a Git worktree by the time the
            factory is used).
        pack_config: Immutable policy dict handed to pack tools.
        identity: Host identity fields stamped into every context so
            ownership checks (``WorkProduct.belongs_to``) pass.
        data_dir: Canonical host data root injected into contexts.
        artifact_store: In-memory ``WorkProductStore``.
        parameters: Default tool parameters merged (shallow) per call.
        mode_id: Mode id stamped on every context.
        tool_id: Tool id stamped on every context.
        approvals: Optional mutable dict the tool may read for
            approval-service seams; harness tests never require it.
        mcp_tool_call: Optional double for ``ctx.mcp_tool_call`` — tools
            that delegate to an MCP capability (``fetch_page`` calls
            ``fetch.fetch_url``) fail loudly without one.
    """

    root: Path
    pack_config: dict[str, Any]
    identity: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_IDENTITY))
    data_dir: Path | None = None
    artifact_store: WorkProductStore = field(default_factory=MemoryWorkProductStore)
    parameters: dict[str, Any] = field(default_factory=dict)
    mode_id: str = "packstest"
    tool_id: str = "packstest-tool"
    execution_floor_for: Callable[[str], ExecutionFloorView | None] | None = None
    delegated_tool_dispatch: Callable[[str, Mapping[str, Any]], Awaitable[ToolResult]] | None = None
    mcp_tool_call: Callable[[str, Mapping[str, Any]], Awaitable[ToolResult]] | None = None

    def context(
        self,
        parameters: dict[str, Any] | None = None,
        *,
        tool_id: str | None = None,
        approval_request_id: str = "",
    ) -> ToolExecutionContext:
        """Return a new execution context for one tool invocation."""

        merged = dict(self.parameters)
        if parameters:
            merged.update(parameters)
        effective_tool_id = tool_id or self.tool_id
        return ToolExecutionContext(
            tool_id=effective_tool_id,
            mode_id=self.mode_id,
            parameters=merged,
            pack_config=dict(self.pack_config),
            execution_floor_for=self.execution_floor_for,
            delegated_tool_dispatch=self.delegated_tool_dispatch,
            mcp_tool_call=self.mcp_tool_call,
            approval_service=None,
            approval_request_id=approval_request_id,
            task_id=self.identity.get("task_id", ""),
            session_plan_digest=self.identity["session_plan_digest"],
            code_change_artifact_store=self.artifact_store,
            event_bus=NullEventBus(),
            state_provider=StaticStateProvider(None),
            stream_output=False,
            data_dir=str(self.data_dir or self.root / ".packstest-data"),
            workspace_id=self.identity["workspace_id"],
            user_id=self.identity["user_id"],
            user_slug=self.identity["user_slug"],
            surface_type=self.identity["surface_type"],  # type: ignore[arg-type]
            surface_id=self.identity["surface_id"],
            memory_scope=self.identity["memory_scope"],  # type: ignore[arg-type]
        )


def make_workspace(tmp_path: Path, *, name: str = "source") -> Path:
    """Create a bare (non-Git) directory usable as a workspace root.

    Git-aware fixtures live in :mod:`packstest.git`; this helper is for
    tools that only need an existing directory.
    """

    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_context(
    root: Path,
    parameters: dict[str, Any] | None = None,
    *,
    pack_config: dict[str, Any] | None = None,
    tool_id: str = "packstest-tool",
    mode_id: str = "packstest",
) -> ToolExecutionContext:
    """One-shot context builder for quick unit tests.

    Convenience wrapper around :class:`ToolContextFactory` for tests that
    need exactly one context and do not care about sharing stores.
    """

    factory = ToolContextFactory(
        root=root,
        pack_config=pack_config or {"workspace_root": str(root)},
        tool_id=tool_id,
        mode_id=mode_id,
    )
    return factory.context(parameters or {})
