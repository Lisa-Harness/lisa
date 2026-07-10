"""Tests for the extended self_model operation with introspection telemetry."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from kernel.models import (
    IntrospectSnapshot,
    McpServerHealth,
    OperationExecutionContext,
)
from sdk_packs.core.operations.self_model.self_model import build_self_model_vibe


def _make_artifact():
    """Build a minimal LisaSelfModelArtifact for vibe testing."""
    from kernel.models import LisaSelfModelArtifact

    return LisaSelfModelArtifact(
        generated_at=datetime.now(UTC).isoformat(),
        workspace_id="test",
        user_slug="test_user",
        identity_summary="Test Lisa with 5 modes, 3 operations, 10 tools.",
        grounding_summary="Host: test. Timezone: UTC.",
        identity=[],
        capabilities=[],
        limitations=[],
        architecture=[],
        memory_awareness=[],
        grounding=[],
        known_unknowns=[],
    )


def test_build_self_model_vibe_includes_runtime_telemetry() -> None:
    """The vibe string includes runtime telemetry when telemetry fields are set."""
    artifact = _make_artifact()
    artifact.uptime_seconds = 3600.0
    artifact.asyncio_tasks = 47
    artifact.python_threads = 5
    artifact.process_rss_mb = 187.3
    artifact.mcp_pool_health = {
        "mem0": McpServerHealth(server_id="mem0", status="failed", warm=True),
    }

    vibe = build_self_model_vibe(artifact)
    assert "Runtime:" in vibe
    assert "uptime=3600s" in vibe
    assert "tasks=47" in vibe
    assert "rss=187MB" in vibe
    assert "mcp_down=mem0" in vibe


def test_build_self_model_vibe_without_telemetry() -> None:
    """The vibe string works without telemetry (backward compatible)."""
    artifact = _make_artifact()
    vibe = build_self_model_vibe(artifact)
    assert "## Self Model" in vibe
    assert "Test Lisa" in vibe
    assert "Runtime:" not in vibe


@pytest.mark.asyncio
async def test_self_model_execute_with_introspect_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    """self_model.execute() populates telemetry fields when introspect_fn is wired."""
    from sdk_packs.core.operations.self_model.self_model import execute
    from sdk_packs.core.operations.self_model import self_model as self_model_module

    monkeypatch.setattr(self_model_module, "get_workflow_catalog", lambda: {})

    async def _mock_introspect() -> IntrospectSnapshot:
        return IntrospectSnapshot(
            captured_at=datetime.now(UTC).isoformat(),
            uptime_seconds=120.0,
            asyncio_tasks=10,
            python_threads=3,
            python_modules=500,
            process_rss_mb=42.5,
            process_cpu_percent=1.2,
            mcp_pool_health={
                "test_server": McpServerHealth(
                    server_id="test_server", status="warm", warm=True
                ),
            },
            token_usage_24h=None,
            recent_log_errors=[],
            logfile_tail_lines=0,
        )

    ctx = OperationExecutionContext(
        operation_id="self_model",
        user_slug="test_user",
        introspect_fn=_mock_introspect,
    )

    result = await execute(ctx)
    artifact_json = result.output_json
    assert isinstance(artifact_json, dict)
    assert artifact_json["uptime_seconds"] == 120.0
    assert artifact_json["asyncio_tasks"] == 10
    assert artifact_json["python_threads"] == 3
    assert artifact_json["process_rss_mb"] == pytest.approx(42.5)
    assert "test_server" in artifact_json["mcp_pool_health"]


@pytest.mark.asyncio
async def test_self_model_execute_without_introspect_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    """self_model.execute() works without introspect_fn — telemetry fields are empty."""
    from sdk_packs.core.operations.self_model.self_model import execute
    from sdk_packs.core.operations.self_model import self_model as self_model_module

    monkeypatch.setattr(self_model_module, "get_workflow_catalog", lambda: {})

    ctx = OperationExecutionContext(
        operation_id="self_model",
        user_slug="test_user",
        # introspect_fn is None by default
    )

    result = await execute(ctx)
    artifact_json = result.output_json
    assert isinstance(artifact_json, dict)
    assert artifact_json["uptime_seconds"] == 0.0
    assert artifact_json["asyncio_tasks"] == 0
    assert artifact_json["python_threads"] == 0
    assert artifact_json["process_rss_mb"] == 0.0
    assert artifact_json["mcp_pool_health"] == {}