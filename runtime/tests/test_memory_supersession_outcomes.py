"""Verify retirement receipts through the in-process MCP and operation path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from kernel.models import OperationExecutionContext, ToolExecutionContext, ToolResult
from kernel.seams.host_adapter import build_callable_host
from kernel.tools.mcp_client import InProcessServerEntry, MCPClientConfig, MCPToolPool
from kernel.tools.synthetic import mcp_alias_tool
from mcp_server.registry import get_in_process_server
from runtime.testing import build_test_runtime
from runtime.turn_hooks_builder import _execute_mcp_tool
from sdk.packs.memory.operations.memory import dispute, mine_thread_segment


def _resolve_server(server_id: str) -> InProcessServerEntry | None:
    entry = get_in_process_server(server_id)
    if entry is None:
        return None
    return InProcessServerEntry(server_id=entry.server_id, mcp=entry.mcp, init=entry.init)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["dispute", "mining"])
@pytest.mark.parametrize(
    ("race", "expected_retired"),
    [("none", True), ("retire", False), ("delete", False), ("unreported", None)],
)
async def test_retirement_outcome_survives_mcp_and_lookup_write_races(
    tmp_path: Path, operation: str, race: str, expected_retired: bool | None
) -> None:
    data_dir = tmp_path / "memory"
    config = MCPClientConfig(
        server_id="lisa-memory",
        command="",
        args=["--data-dir", str(data_dir)],
        env={},
        warm=True,
        required=True,
        runtime_kind="in_process",
        data_dir_path=str(data_dir),
        ensure_data_dir=True,
    )
    pool = MCPToolPool([config], in_process_resolver=_resolve_server)
    runtime = build_test_runtime(tmp_path / "host", user_slug="test-user")
    runtime.resources._mcp_tool_pool = pool
    scope = {
        "user_slug": "test-user",
        "workspace_id": "default",
        "memory_scope": "personal",
        "surface_id": "test-surface",
    }

    try:
        await pool.start()
        pool.add_synthetic_tools(
            [
                mcp_alias_tool(
                    name=f"memory.{name}",
                    description="memory alias",
                    pool=pool,
                    target_tool=f"{name}_memory",
                    target_server_id="lisa-memory",
                    result_map=(
                        (lambda text: text.splitlines()[-1])
                        if name == "add" and race == "unreported"
                        else None
                    ),
                )
                for name in ("add", "get", "search")
            ]
        )
        original = await pool.call_tool(
            "lisa-memory.add_memory", {**scope, "content": "original fact"}
        )
        assert not original.is_error
        original_id = int(original.output_text.rsplit("ID ", 1)[-1])
        sentinel = await pool.call_tool("lisa-memory.add_memory", {**scope, "content": "sentinel"})
        assert not sentinel.is_error

        async def execute_tool(tool_id: str, tool_ctx: ToolExecutionContext) -> ToolResult:
            if tool_id == "memory.add" and race in {"retire", "delete"}:
                changed = await pool.call_tool(
                    f"lisa-memory.{race}_memory", {**scope, "memory_id": original_id}
                )
                assert not changed.is_error
            return await _execute_mcp_tool(runtime, tool_id, tool_ctx)

        async def complete(messages: list[dict[str, Any]], **_kwargs: Any) -> str:
            return json.dumps(
                {
                    "facts": [
                        {
                            "action": "update",
                            "fact": "corrected fact",
                            "supersedes_id": original_id,
                        }
                    ]
                }
            )

        host = build_callable_host(
            execute_tool_fn=execute_tool,
            complete_fn=complete,
            resolve_model_fn=lambda _model_id: {"model_id": "test-model"},
            read_authored_text_fn=lambda _path: "test prompt",
        )
        params: dict[str, Any]
        if operation == "dispute":
            params = {"memory_id": str(original_id), "correction": "corrected fact"}
        else:
            params = {
                **scope,
                "turns": [{"turn": 1, "query": "original fact", "response_text": "corrected"}],
            }
        ctx = OperationExecutionContext(
            operation_id=operation,
            user_id="test-user",
            params=params,
            host=host,
            **scope,
        )

        if operation == "dispute":
            result = await dispute.execute(ctx)
            assert result.output_json is not None
            assert result.output_json["retired"] is expected_retired
        else:
            result = await mine_thread_segment.execute(ctx)
            assert result.extra["retired_ids"] == (
                [original_id] if expected_retired is True else []
            )
            assert result.extra["not_retired_by_write_ids"] == (
                [original_id] if expected_retired is False else []
            )
            assert result.extra["retirement_unknown_ids"] == (
                [original_id] if expected_retired is None else []
            )
            assert result.extra["supersession_outcomes"] == [
                {
                    "memory_id": result.extra["updated_ids"][0],
                    "supersedes_id": original_id,
                    "retired": expected_retired,
                }
            ]

        assert result.status == "success"
        active = await pool.call_tool("lisa-memory.list_memories", scope)
        assert not active.is_error
        corrected = [
            record
            for record in json.loads(active.output_text)
            if record["content"] == "corrected fact"
        ]
        assert len(corrected) == 1
        assert corrected[0]["supersedes_id"] == original_id
    finally:
        await pool.aclose()
