"""Tests for CLI host restart behavior after pack lifecycle commands.

Covers the platform-aware restart contract:
- Unix: true in-place ``os.execv`` re-exec.
- Windows: do NOT spawn (parent-link loss); print a notice and exit with
  ``_RESTART_REQUIRED_EXIT`` so the user manually re-runs ``lisa``.
- One-shot paths (``once``, ``pack``) exit with ``_RESTART_REQUIRED_EXIT``
  on all platforms instead of re-execing.
- Resume-message env var is consumed once on REPL startup, then cleared.
"""

from __future__ import annotations

# pylint: disable=protected-access
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from host.runtime import HostRuntime

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path."""
    from kernel.testing.runtime_factory import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


# ── _restart_current_process ──────────────────────────────────────────


def test_restart_current_process_unix_uses_execv(tmp_path: Path) -> None:
    """On Unix, _restart_current_process calls os.execv in-place."""
    from hosts.cli import main as cli_main

    captured: dict[str, Any] = {}

    def fake_execv(path: str, args: list[str]) -> None:
        captured["path"] = path
        captured["args"] = args

    with (
        patch.object(cli_main.os, "name", "posix"),
        patch.object(cli_main.os, "execv", fake_execv),
        patch.object(cli_main.sys, "executable", "/usr/bin/python"),
        patch.object(cli_main.sys, "argv", ["lisa", "--debug"]),
    ):
        cli_main._restart_current_process()

    assert captured["path"] == "/usr/bin/python"
    assert captured["args"] == ["/usr/bin/python", "lisa", "--debug"]


def test_restart_current_process_windows_exits_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows, _restart_current_process does NOT spawn; exits with code."""
    import typer
    from hosts.cli import main as cli_main

    spawn_calls: list[Any] = []
    execv_calls: list[Any] = []

    monkeypatch.setattr(cli_main.os, "name", "nt")
    monkeypatch.setattr(
        cli_main.subprocess,
        "Popen",
        lambda *a, **k: spawn_calls.append((a, k)),
    )
    monkeypatch.setattr(cli_main.os, "execv", lambda *a: execv_calls.append(a))

    with pytest.raises(typer.Exit) as exc_info:
        cli_main._restart_current_process()

    assert exc_info.value.exit_code == cli_main._RESTART_REQUIRED_EXIT
    assert spawn_calls == [], "Windows path must NOT spawn a subprocess"
    assert execv_calls == [], "Windows path must NOT call os.execv"


# ── _exit_restart_required (one-shot paths) ───────────────────────────


def test_exit_restart_required_raises_typer_exit(tmp_path: Path) -> None:
    """_exit_restart_required raises typer.Exit with the restart code."""
    import typer
    from hosts.cli import main as cli_main

    with pytest.raises(typer.Exit) as exc_info:
        cli_main._exit_restart_required()

    assert exc_info.value.exit_code == cli_main._RESTART_REQUIRED_EXIT


# ── _confirm_restart (REPL yes/no before exiting to reload) ───────────


def test_confirm_restart_yes_proceeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the user answers yes, _confirm_restart returns True."""
    from hosts.cli import repl as cli_repl
    from rich.console import Console

    runtime = _make_runtime(tmp_path)
    runtime.request_restart("pack install: lisa-pack-cloudflare")
    console = Console(quiet=True)

    monkeypatch.setattr(cli_repl.Confirm, "ask", staticmethod(lambda *a, **k: True))

    assert cli_repl._confirm_restart(runtime, console) is True


def test_confirm_restart_no_defers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the user answers no, _confirm_restart returns False (defer)."""
    from hosts.cli import repl as cli_repl
    from rich.console import Console

    runtime = _make_runtime(tmp_path)
    runtime.request_restart("pack install: lisa-pack-cloudflare")
    console = Console(quiet=True)

    monkeypatch.setattr(cli_repl.Confirm, "ask", staticmethod(lambda *a, **k: False))

    assert cli_repl._confirm_restart(runtime, console) is False


def test_confirm_restart_eof_defers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On EOF (Ctrl+D), _confirm_restart defers rather than forcing exit."""

    def _raise_eof(*a: Any, **k: Any) -> bool:
        raise EOFError

    from hosts.cli import repl as cli_repl
    from rich.console import Console

    runtime = _make_runtime(tmp_path)
    runtime.request_restart("pack enable: lisa-pack-cloudflare")
    console = Console(quiet=True)

    monkeypatch.setattr(cli_repl.Confirm, "ask", staticmethod(_raise_eof))

    assert cli_repl._confirm_restart(runtime, console) is False


def test_confirm_restart_keyboard_interrupt_defers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Ctrl+C, _confirm_restart defers rather than forcing exit."""

    def _raise_interrupt(*a: Any, **k: Any) -> bool:
        raise KeyboardInterrupt

    from hosts.cli import repl as cli_repl
    from rich.console import Console

    runtime = _make_runtime(tmp_path)
    runtime.request_restart("pack enable: lisa-pack-cloudflare")
    console = Console(quiet=True)

    monkeypatch.setattr(cli_repl.Confirm, "ask", staticmethod(_raise_interrupt))

    assert cli_repl._confirm_restart(runtime, console) is False


# ── Resume-message consumption ────────────────────────────────────────


def test_consume_restart_resume_message_stashes_and_clears_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resume env var is popped and stashed on the runtime."""
    from hosts.cli import main as cli_main

    monkeypatch.setenv(cli_main._RESTART_RESUME_MESSAGE_ENV, "check cloudflare status")
    runtime = _make_runtime(tmp_path)
    assert runtime.pending_resume_message is None

    cli_main._consume_restart_resume_message(runtime)

    assert runtime.pending_resume_message == "check cloudflare status"
    assert cli_main._RESTART_RESUME_MESSAGE_ENV not in os.environ


def test_consume_restart_resume_message_noop_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the env var is absent, nothing is stashed and no error is raised."""
    from hosts.cli import main as cli_main

    monkeypatch.delenv(cli_main._RESTART_RESUME_MESSAGE_ENV, raising=False)
    runtime = _make_runtime(tmp_path)

    cli_main._consume_restart_resume_message(runtime)

    assert runtime.pending_resume_message is None
    assert cli_main._RESTART_RESUME_MESSAGE_ENV not in os.environ


def test_consume_restart_resume_message_clears_after_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After consumption, a second call does not re-stash a stale message."""
    from hosts.cli import main as cli_main

    monkeypatch.setenv(cli_main._RESTART_RESUME_MESSAGE_ENV, "run again")
    runtime = _make_runtime(tmp_path)

    cli_main._consume_restart_resume_message(runtime)
    assert runtime.pending_resume_message == "run again"

    runtime.pending_resume_message = None
    cli_main._consume_restart_resume_message(runtime)
    assert runtime.pending_resume_message is None


# ── Runtime field ─────────────────────────────────────────────────────


def test_host_runtime_has_pending_resume_message_field(tmp_path: Path) -> None:
    """HostRuntime exposes pending_resume_message, defaulting to None."""
    runtime = _make_runtime(tmp_path)
    assert hasattr(runtime, "pending_resume_message")
    assert runtime.pending_resume_message is None
