"""Tests for the core-pack delegate operation handler.

Exercises the ``delegate`` operation through its ``execute(ctx)`` function
directly — the same entry point the kernel's operation registry dispatcher
calls. The ``host.delegate`` seam is replaced with a fake so the tests run
in isolation without spawning the catalog MCP subprocess or recursing into
``run_turn``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kernel.delegation import DelegationResult, DelegationSpec
from kernel.models import OperationExecutionContext, OperationResult
from kernel.seams.host_adapter import build_callable_host
from sdk.packs.core.operations.delegation import delegate as delegate_mod


def _patched_catalog(monkeypatch: pytest.MonkeyPatch, modes: set[str]) -> None:
    """Patch ``get_workflow_catalog`` to return a stub catalog."""

    def _stub_catalog() -> dict[str, SimpleNamespace]:
        return {m: SimpleNamespace() for m in modes}

    monkeypatch.setattr(delegate_mod, "get_workflow_catalog", _stub_catalog)


def _ctx_with_host(
    *,
    mode_id: str = "default",
    query: str = "",
    params: dict[str, Any] | None = None,
    delegate_fn: Any = None,
) -> OperationExecutionContext:
    """Build an ``OperationExecutionContext`` with a ``_CallableHost``.

    The host wires only ``delegate_fn``; the other seams are left unwired
    (the delegate operation does not call them).
    """
    host = build_callable_host(delegate_fn=delegate_fn)
    return OperationExecutionContext(
        mode_id=mode_id,
        query=query,
        params=params or {},
        host=host,
    )


@pytest.mark.asyncio
async def test_delegate_returns_ok_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful delegation returns the child's output and structured payload."""
    _patched_catalog(monkeypatch, {"search"})

    async def fake_delegate_fn(spec: DelegationSpec) -> DelegationResult:
        assert spec.target_mode_id == "search"
        assert spec.query == "find rust async news"
        return DelegationResult(
            target_mode_id="search",
            status="ok",
            output_text="rust async runtime news",
            tool_call_count=2,
            called_tool_names=["web_search", "fetch_page"],
            duration_ms=42,
            depth=1,
        )

    ctx = _ctx_with_host(
        params={"target_mode_id": "search", "query": "find rust async news"},
        delegate_fn=fake_delegate_fn,
    )
    result = await delegate_mod.execute(ctx)
    assert isinstance(result, OperationResult)
    assert result.output_json["status"] == "ok"
    assert result.output_json["target_mode_id"] == "search"
    assert result.output_json["tool_call_count"] == 2
    assert result.output_json["called_tool_names"] == ["web_search", "fetch_page"]
    assert "rust async runtime news" in result.output_text
    assert "`ok`" in result.output_text


@pytest.mark.asyncio
async def test_delegate_fails_loudly_on_unknown_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown target mode returns an error, not a silent fallback."""
    _patched_catalog(monkeypatch, {"search"})

    async def fake_delegate_fn(spec: DelegationSpec) -> DelegationResult:
        raise AssertionError("delegate should not be called for an unknown mode")

    ctx = _ctx_with_host(
        params={"target_mode_id": "nonexistent", "query": "test"},
        delegate_fn=fake_delegate_fn,
    )
    result = await delegate_mod.execute(ctx)
    assert result.output_json["status"] == "error"
    assert "nonexistent" in result.output_text


@pytest.mark.asyncio
async def test_delegate_fails_loudly_when_hook_not_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing host.delegate returns an error, not a silent fallback."""
    _patched_catalog(monkeypatch, {"search"})

    # No host wired at all.
    ctx = OperationExecutionContext(
        mode_id="default",
        params={"target_mode_id": "search", "query": "test"},
    )
    result = await delegate_mod.execute(ctx)
    assert result.output_json["status"] == "error"
    assert "host.delegate" in result.output_text


@pytest.mark.asyncio
async def test_delegate_fails_loudly_on_invalid_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing required parameters return a validation error."""
    _patched_catalog(monkeypatch, {"search"})

    async def fake_delegate_fn(spec: DelegationSpec) -> DelegationResult:
        raise AssertionError("delegate should not be called for invalid params")

    ctx = _ctx_with_host(
        params={"target_mode_id": "search"},  # missing query
        delegate_fn=fake_delegate_fn,
    )
    result = await delegate_mod.execute(ctx)
    assert result.output_json["status"] == "error"
    assert "validation" in result.output_text.lower()


@pytest.mark.asyncio
async def test_delegate_falls_back_to_ctx_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """When params has no query, the operation falls back to ctx.query."""
    _patched_catalog(monkeypatch, {"search"})

    captured: list[str] = []

    async def fake_delegate_fn(spec: DelegationSpec) -> DelegationResult:
        captured.append(spec.query)
        return DelegationResult(target_mode_id="search", status="ok", output_text="ok")

    ctx = _ctx_with_host(
        query="delegate to search for rust async news",
        params={"target_mode_id": "search"},  # no query → fall back to ctx.query
        delegate_fn=fake_delegate_fn,
    )
    result = await delegate_mod.execute(ctx)
    assert result.output_json["status"] == "ok"
    assert captured == ["delegate to search for rust async news"]


@pytest.mark.asyncio
async def test_delegate_rejects_unexpected_result_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host.delegate returning a non-DelegationResult fails loudly."""
    _patched_catalog(monkeypatch, {"search"})

    async def fake_delegate_fn(spec: DelegationSpec) -> object:
        return "not a DelegationResult"

    ctx = _ctx_with_host(
        params={"target_mode_id": "search", "query": "test"},
        delegate_fn=fake_delegate_fn,
    )
    result = await delegate_mod.execute(ctx)
    assert result.output_json["status"] == "error"
    assert "unexpected result type" in result.output_text.lower()


@pytest.mark.asyncio
async def test_delegate_renders_no_tools_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A no_tools child renders the status and error message honestly."""
    _patched_catalog(monkeypatch, {"remember"})

    async def fake_delegate_fn(spec: DelegationSpec) -> DelegationResult:
        return DelegationResult(
            target_mode_id="remember",
            status="no_tools",
            output_text="I learned it (unverified)",
            error_message="remember child made 3 tool calls but none were action tools",
            tool_call_count=3,
            called_tool_names=["recall_memory", "read_file", "list_files"],
            duration_ms=100,
            depth=1,
        )

    ctx = _ctx_with_host(
        params={"target_mode_id": "remember", "query": "learn about rust"},
        delegate_fn=fake_delegate_fn,
    )
    result = await delegate_mod.execute(ctx)
    assert result.output_json["status"] == "no_tools"
    assert "`no_tools`" in result.output_text
    assert "unverified" in result.output_text.lower() or "error" in result.output_text.lower()
