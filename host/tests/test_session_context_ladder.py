"""Tests for host.session_context.resolve_session_context_source.

The resolution ladder is the load-bearing startup path for every host:
an explicit path, else LISA_SESSION_CONTEXT_PATH, else the data-dir
config file, else the bundled core default. These tests pin each rung
and the loud-failure contract (a missing explicit or env path raises
instead of silently falling through).

No MCP subprocess is spawned; the bundled-default rung is exercised
through the real filesystem layout of the framework checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from host.session_context import (
    bundled_default_session_path,
    resolve_session_context_source,
)

_MINIMAL_SESSION = (
    "---\n"
    "schema_version: 2\n"
    "name: ladder-test\n"
    "surface: { type: personal_dm, id: host-console, memory_scope: testing }\n"
    "---\n"
    "# Session\n"
)


def _write_session(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_MINIMAL_SESSION, encoding="utf-8")
    return path


# ── Rung 1: explicit CLI path ─────────────────────────────────────────


def test_explicit_path_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = _write_session(tmp_path / "explicit.md")
    # Even with the env var set and a data-dir file present, explicit wins.
    env_file = _write_session(tmp_path / "env.md")
    data_dir = tmp_path / "data"
    _write_session(data_dir / "config" / "session_context.md")
    monkeypatch.setenv("LISA_SESSION_CONTEXT_PATH", str(env_file))

    source = resolve_session_context_source(session_context=str(explicit), data_dir=data_dir)
    assert source.cached_path == explicit.resolve()
    assert "ladder-test" in source.text


def test_explicit_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        resolve_session_context_source(
            session_context=str(tmp_path / "nope.md"), data_dir=tmp_path
        )


def test_explicit_directory_raises(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    with pytest.raises(ValueError, match="Markdown file"):
        resolve_session_context_source(session_context=str(tmp_path / "adir"), data_dir=tmp_path)


# ── Rung 2: LISA_SESSION_CONTEXT_PATH ────────────────────────────────


def test_env_path_used_when_no_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = _write_session(tmp_path / "env-context.md")
    monkeypatch.setenv("LISA_SESSION_CONTEXT_PATH", str(env_file))
    source = resolve_session_context_source(session_context="", data_dir=tmp_path / "data")
    assert source.cached_path == env_file.resolve()


def test_env_path_missing_file_raises_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured-but-missing env path is an error, not a silent fallthrough."""
    monkeypatch.setenv("LISA_SESSION_CONTEXT_PATH", str(tmp_path / "missing.md"))
    with pytest.raises(FileNotFoundError, match="LISA_SESSION_CONTEXT_PATH"):
        resolve_session_context_source(session_context="", data_dir=tmp_path)


# ── Rungs 3+4: data-dir config file, bundled default ─────────────────


def test_data_dir_config_file_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LISA_SESSION_CONTEXT_PATH", raising=False)
    data_dir = tmp_path / "data"
    authored = _write_session(data_dir / "config" / "session_context.md")
    source = resolve_session_context_source(session_context="", data_dir=data_dir)
    assert source.cached_path == authored.resolve()


def test_bundled_default_when_nothing_authored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LISA_SESSION_CONTEXT_PATH", raising=False)
    # Ensure no cwd/session_context.md interferes: pytest tmp chdir is not
    # used, so just verify the rung resolves to the bundled default file.
    source = resolve_session_context_source(session_context="", data_dir=tmp_path)
    assert source.cached_path == bundled_default_session_path().resolve()
    assert source.text  # the bundled session is non-empty