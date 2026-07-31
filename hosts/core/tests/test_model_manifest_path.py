"""Tests for session-context-driven model manifest path override.

Verifies that ``context.models.path`` in a session plan:
- resolves a session-relative path against the session source directory,
- resolves an absolute path directly,
- falls back to ``data_dir/models.yaml`` when omitted,
- rejects missing files and wrong extensions at host boot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hosts.core.config import HostConfig
from hosts.core.model_runtime import (
    active_manifest_path,
    resolve_session_manifest_path,
)
from hosts.core.runtime import HostRuntime
from hosts.core.session_context import compile_selected_session_plan


def _write_session(tmp_path: Path, *, models_path: str | None) -> Path:
    """Write a session_context.md with an optional context.models block."""
    session = tmp_path / "session_context.md"
    models_block = ""
    if models_path is not None:
        models_block = f"context:\n  models:\n    path: {models_path}\n"
    session.write_text(
        "---\n"
        "schema_version: 2\n"
        "name: models-path-test\n"
        "surface: { type: personal_dm, id: host-console, memory_scope: testing }\n"
        f"{models_block}"
        "---\n"
        "# Session\n\nAct precisely.\n",
        encoding="utf-8",
    )
    return session


class _MockModels:
    def __init__(self, path: str) -> None:
        self.path = path


class _MockContext:
    def __init__(self, models: Any) -> None:
        self.models = models


class _MockManifest:
    def __init__(self, models: Any) -> None:
        self.context = _MockContext(models)


class _MockPlan:
    def __init__(self, models: Any) -> None:
        self.manifest = _MockManifest(models)
        self.source_dir = None


def test_resolve_session_manifest_path_falls_back_when_plan_is_none() -> None:
    """No session plan → data_dir/models.yaml."""
    resolved = resolve_session_manifest_path(
        session_plan=None,
        session_source_dir=None,
        data_dir=Path("C:/data"),
    )
    assert resolved == active_manifest_path(data_dir=Path("C:/data"))


def test_resolve_session_manifest_path_falls_back_when_models_is_none() -> None:
    """Session plan without context.models → data_dir/models.yaml."""
    plan = _MockPlan(models=None)
    resolved = resolve_session_manifest_path(
        session_plan=plan,
        session_source_dir=None,
        data_dir=Path("C:/data"),
    )
    assert resolved == Path("C:/data/models.yaml")


def test_resolve_session_manifest_path_resolves_session_relative(tmp_path: Path) -> None:
    """A relative context.models.path resolves against the session source dir."""
    roster = tmp_path / "roster.yaml"
    roster.write_text("models: []\n", encoding="utf-8")
    plan = _MockPlan(models=_MockModels("./roster.yaml"))
    plan.source_dir = str(tmp_path)
    resolved = resolve_session_manifest_path(
        session_plan=plan,
        session_source_dir=tmp_path,
        data_dir=Path("C:/data"),
    )
    assert resolved == roster.resolve()


def test_resolve_session_manifest_path_resolves_absolute(tmp_path: Path) -> None:
    """An absolute context.models.path resolves directly."""
    roster = tmp_path / "roster.yaml"
    roster.write_text("models: []\n", encoding="utf-8")
    plan = _MockPlan(models=_MockModels(str(roster.resolve())))
    resolved = resolve_session_manifest_path(
        session_plan=plan,
        session_source_dir=None,
        data_dir=Path("C:/data"),
    )
    assert resolved == roster.resolve()


def test_resolve_session_manifest_path_rejects_missing_file(tmp_path: Path) -> None:
    """A context.models.path that doesn't exist raises ValueError."""
    plan = _MockPlan(models=_MockModels("./nope.yaml"))
    plan.source_dir = str(tmp_path)
    with pytest.raises(ValueError, match="not a file"):
        resolve_session_manifest_path(
            session_plan=plan,
            session_source_dir=tmp_path,
            data_dir=Path("C:/data"),
        )


def test_resolve_session_manifest_path_rejects_wrong_extension(tmp_path: Path) -> None:
    """A context.models.path with a wrong extension raises ValueError."""
    bad = tmp_path / "roster.txt"
    bad.write_text("x", encoding="utf-8")
    plan = _MockPlan(models=_MockModels("./roster.txt"))
    plan.source_dir = str(tmp_path)
    with pytest.raises(ValueError, match=".yaml"):
        resolve_session_manifest_path(
            session_plan=plan,
            session_source_dir=tmp_path,
            data_dir=Path("C:/data"),
        )


def test_resolve_session_manifest_path_rejects_relative_without_source_dir() -> None:
    """A relative context.models.path with no source dir raises ValueError."""
    plan = _MockPlan(models=_MockModels("./roster.yaml"))
    with pytest.raises(ValueError, match="relative"):
        resolve_session_manifest_path(
            session_plan=plan,
            session_source_dir=None,
            data_dir=Path("C:/data"),
        )


def test_host_runtime_models_path_uses_session_override(tmp_path: Path) -> None:
    """HostRuntime.models_path returns the session-declared roster path."""
    roster = tmp_path / "roster.yaml"
    roster.write_text(
        "models:\n  - model_id: default\n    provider: azure_openai\n    context_window: 8000\n",
        encoding="utf-8",
    )
    session = _write_session(tmp_path, models_path="./roster.yaml")
    config = HostConfig.from_env(data_dir=str(tmp_path), session_context=str(session))
    runtime = HostRuntime(config)
    assert runtime.models_path == roster.resolve()


def test_host_runtime_models_path_falls_back_to_data_dir(tmp_path: Path) -> None:
    """Without context.models, HostRuntime.models_path is data_dir/models.yaml."""
    session = _write_session(tmp_path, models_path=None)
    config = HostConfig.from_env(data_dir=str(tmp_path), session_context=str(session))
    runtime = HostRuntime(config)
    assert runtime.models_path == (tmp_path / "models.yaml").resolve()


def test_compile_selected_session_plan_threads_source_dir(tmp_path: Path) -> None:
    """compile_selected_session_plan sets source_dir from the resolved file."""
    roster = tmp_path / "roster.yaml"
    roster.write_text("models: []\n", encoding="utf-8")
    session = _write_session(tmp_path, models_path="./roster.yaml")
    plan = compile_selected_session_plan(session_context=str(session), data_dir=tmp_path)
    assert plan.source_dir == str(tmp_path.resolve())
    assert plan.manifest.context.models is not None
    assert plan.manifest.context.models.path == "./roster.yaml"


def test_host_runtime_models_path_rejects_missing_session_roster(tmp_path: Path) -> None:
    """A session context.models.path pointing at a missing file fails at boot."""
    session = _write_session(tmp_path, models_path="./missing.yaml")
    config = HostConfig.from_env(data_dir=str(tmp_path), session_context=str(session))
    with pytest.raises(ValueError, match="not a file"):
        HostRuntime(config)
