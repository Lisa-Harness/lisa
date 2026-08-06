"""Tests for the core-pack fork operation handler.

Exercises the ``fork`` operation through its ``execute(ctx)`` function
directly — the same entry point the kernel's operation registry dispatcher
calls. The ``host.fork`` seam is replaced with a fake so the tests run
in isolation without spawning the catalog MCP subprocess or recursing
into ``run_turn``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kernel.models import (
    ForkChildResult,
    ForkResult,
    ForkSpec,
    ForkTarget,
    OperationExecutionContext,
    OperationResult,
)
from kernel.seams.host_adapter import build_callable_host
from sdk.packs.core.operations.delegation import fork as fork_mod


def _patched_catalog(monkeypatch: pytest.MonkeyPatch, modes: set[str]) -> None:
    """Patch ``get_workflow_catalog`` to return a stub catalog."""

    def _stub_catalog() -> dict[str, SimpleNamespace]:
        return {
            m: SimpleNamespace(start_stage_id="start", satisfies_intents=[])
            for m in modes
        }

    monkeypatch.setattr(fork_mod, "get_workflow_catalog", _stub_catalog)


def _ctx_with_host(
    *,
    mode_id: str = "default",
    query: str = "",
    params: dict[str, Any] | None = None,
    fork_fn: Any = None,
) -> OperationExecutionContext:
    """Build an ``OperationExecutionContext`` with a ``_CallableHost``.

    The host wires only ``fork_fn``; the other seams are left unwired
    (the fork operation does not call them).
    """
    host = build_callable_host(fork_fn=fork_fn)
    return OperationExecutionContext(
        mode_id=mode_id,
        query=query,
        params=params or {},
        host=host,
    )


@pytest.mark.asyncio
async def test_fork_returns_ok_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful fan-out returns the children's outputs and structured payload."""
    _patched_catalog(monkeypatch, {"remember", "markdown"})

    async def fake_fork_fn(spec: ForkSpec) -> ForkResult:
        assert len(spec.targets) == 2
        assert spec.targets[0].mode_id == "remember"
        assert spec.targets[1].mode_id == "markdown"
        return ForkResult(
            children=[
                ForkChildResult(
                    target=spec.targets[0],
                    status="ok",
                    output_text="remembered",
                    called_tool_names=["memory.add"],
                    duration_ms=10,
                ),
                ForkChildResult(
                    target=spec.targets[1],
                    status="ok",
                    output_text="wrote doc",
                    called_tool_names=["write_file"],
                    duration_ms=20,
                ),
            ],
            succeeded=2,
        )

    ctx = _ctx_with_host(
        params={
            "targets": [
                {"mode_id": "remember", "query": "save this"},
                {"mode_id": "markdown", "query": "write a doc"},
            ],
        },
        fork_fn=fake_fork_fn,
    )
    result = await fork_mod.execute(ctx)
    assert isinstance(result, OperationResult)
    assert result.output_json["succeeded"] == 2
    assert len(result.output_json["children"]) == 2
    assert result.output_json["children"][0]["status"] == "ok"
    assert result.output_json["children"][1]["called_tool_names"] == ["write_file"]
    assert "remembered" in result.output_text
    assert "wrote doc" in result.output_text
    assert "2/2" in result.output_text


@pytest.mark.asyncio
async def test_fork_fails_loudly_on_unknown_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown target mode returns an error, not a silent fallback."""
    _patched_catalog(monkeypatch, {"remember"})

    async def fake_fork_fn(spec: Any) -> ForkResult:
        raise AssertionError("fork should not be called for an unknown mode")

    ctx = _ctx_with_host(
        params={"targets": [{"mode_id": "nonexistent", "query": "test"}]},
        fork_fn=fake_fork_fn,
    )
    result = await fork_mod.execute(ctx)
    assert result.output_json["status"] == "error"
    assert "nonexistent" in result.output_text


@pytest.mark.asyncio
async def test_fork_fails_loudly_when_hook_not_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing host.fork returns an error, not a silent fallback."""
    _patched_catalog(monkeypatch, {"remember"})

    # No host wired at all.
    ctx = OperationExecutionContext(
        mode_id="default",
        params={"targets": [{"mode_id": "remember", "query": "test"}]},
    )
    result = await fork_mod.execute(ctx)
    assert result.output_json["status"] == "error"
    assert "host.fork" in result.output_text


@pytest.mark.asyncio
async def test_fork_fails_loudly_on_invalid_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing required targets return a validation error."""
    _patched_catalog(monkeypatch, {"remember"})

    async def fake_fork_fn(spec: Any) -> ForkResult:
        raise AssertionError("fork should not be called for invalid params")

    ctx = _ctx_with_host(
        params={},  # missing targets
        fork_fn=fake_fork_fn,
    )
    result = await fork_mod.execute(ctx)
    assert result.output_json["status"] == "error"
    assert "validation" in result.output_text.lower()


@pytest.mark.asyncio
async def test_fork_rejects_unexpected_result_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-ForkResult return from host.fork surfaces a clear error."""
    _patched_catalog(monkeypatch, {"remember"})

    async def fake_fork_fn(spec: Any) -> str:
        return "not a ForkResult"

    ctx = _ctx_with_host(
        params={"targets": [{"mode_id": "remember", "query": "test"}]},
        fork_fn=fake_fork_fn,
    )
    result = await fork_mod.execute(ctx)
    assert result.output_json["status"] == "error"
    assert "unexpected result type" in result.output_text


@pytest.mark.asyncio
async def test_fork_renders_no_tools_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``no_tools`` child is rendered with its status and error message."""
    _patched_catalog(monkeypatch, {"markdown"})

    async def fake_fork_fn(spec: Any) -> ForkResult:
        return ForkResult(
            children=[
                ForkChildResult(
                    target=spec.targets[0],
                    status="no_tools",
                    output_text="",
                    error_message="0 tool calls — hallucinated success",
                    duration_ms=5,
                ),
            ],
            succeeded=0,
        )

    ctx = _ctx_with_host(
        params={"targets": [{"mode_id": "markdown", "query": "write a doc"}]},
        fork_fn=fake_fork_fn,
    )
    result = await fork_mod.execute(ctx)
    assert result.output_json["succeeded"] == 0
    assert result.output_json["children"][0]["status"] == "no_tools"
    assert "`no_tools`" in result.output_text
    assert "0/1" in result.output_text


@pytest.mark.asyncio
async def test_fork_falls_back_to_ctx_query_for_target_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a target has no query, the operation falls back to ctx.query."""
    _patched_catalog(monkeypatch, {"remember"})

    captured: list[ForkSpec] = []

    async def fake_fork_fn(spec: Any) -> ForkResult:
        captured.append(spec)
        return ForkResult(children=[], succeeded=0)

    ctx = _ctx_with_host(
        query="remember the meeting notes",
        params={"targets": [{"mode_id": "remember"}]},  # no query
        fork_fn=fake_fork_fn,
    )
    result = await fork_mod.execute(ctx)
    assert result.output_json["succeeded"] == 0
    assert len(captured) == 1
    assert captured[0].targets[0].query == "remember the meeting notes"