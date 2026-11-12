"""Pack history semantics through a real ``run_turn`` loop.

A re-read supersedes the earlier read; the supersession epoch folds the
stale result to a ref stub in the live array, the next model request
carries the stub, and the turn result exposes the history block the host
stores for replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.approval import InMemoryApprovalService
from kernel.history import HistoryPolicy
from kernel.models import ToolDefinition, ToolExecutionContext, ToolResult
from kernel.turn_hooks import (
    ContextHooks,
    HostServiceHandles,
    ModelHooks,
    ToolHooks,
    TurnIdentity,
)
from kernel.turn_runner import TurnHooks, run_turn
from pydantic import BaseModel

from kernel import config


class _Parameters(BaseModel):
    path: str


@dataclass(frozen=True)
class _Completion:
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None


def _read_call(call_id: str, path: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "peek_file", "arguments": json.dumps({"path": path})},
    }


@pytest.mark.asyncio
async def test_superseded_read_folds_mid_turn_and_lands_in_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def execute(ctx: ToolExecutionContext) -> ToolResult:
        path = ctx.parameters["path"]
        return ToolResult(output_text=f"line of {path}\n" * 200, output_json={"path": path})

    monkeypatch.setattr(
        "kernel.tools.registry._TOOL_CATALOG",
        {
            "peek_file": ToolDefinition(
                tool_id="peek_file",
                description="Read a file.",
                pack_id="lisa-pack-x",
                owner_id="",
                parameters_schema=_Parameters,
                implementation_module=SimpleNamespace(execute=execute),
                history=HistoryPolicy(key="peek:{path}", anchor="peek_file:{path}"),
            )
        },
    )
    monkeypatch.setattr(config.settings, "history_epoch_min_tokens", 100)
    # The default mode routes a stage hint; keep the enforcer out of this test.
    monkeypatch.setattr(config.settings, "forced_action_non_compliant_cap", 99)
    store: dict[str, str] = {}

    def persist(record: dict[str, Any]) -> str:
        ref = "r" + hashlib.sha256(record["content"].encode()).hexdigest()
        store[ref] = record["content"]
        return ref

    script = [
        [_read_call("c1", "a.py")],
        [_read_call("c2", "a.py")],
        [_read_call("c3", "b.py")],
    ]

    async def complete(**_kwargs: Any) -> _Completion:
        return _Completion(tool_calls=script.pop(0)) if script else _Completion(content="done")

    result = await run_turn(
        mode_id="default",
        query="read twice",
        hooks=TurnHooks(
            models=ModelHooks(complete=complete),
            context=ContextHooks(),
            tools=ToolHooks(persist_tool_result=persist),
            services=HostServiceHandles(approval_service=InMemoryApprovalService()),
            identity=TurnIdentity(
                workspace_id="w",
                user_id="u",
                user_slug="u",
                surface_id="s",
                session_plan_digest="p",
                task_id="task-1",
            ),
        ),
    )

    assert result.response_text.endswith("done")
    tool_messages = {
        m["tool_call_id"]: m["content"]
        for m in result.model_requests[-1]["messages"]
        if m.get("role") == "tool"
    }
    stub = tool_messages["c1"]
    assert stub.startswith("[superseded by call c2") and "peek_file:a.py" in stub
    assert tool_messages["c2"].startswith("line of a.py")
    records = {r["call_id"]: r for r in result.history["records"]}
    assert set(records) == {"c1", "c2", "c3"}
    # Lossless: the stub's ref is the stored original text.
    assert records["c1"]["ref"] in stub
    assert store[records["c1"]["ref"]].startswith("line of a.py\n" * 200)
    assert all(r["ref"] in store for r in records.values())
