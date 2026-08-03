"""Tests for the CLI ``models`` subcommand delegation to the core handler.

The CLI ``lisa models`` command mirrors the in-REPL ``/models`` command by
delegating the workflow subcommands (discover/apply/reload/edit/help) to
:func:`hosts.core.commands.handle_models_command`. ``show`` keeps the Rich
table renderer. These tests verify the dispatch wiring without paying the
full ``_initialize_runtime`` cost: we stub the runtime builder and the core
handler, then invoke the Typer callback directly.
"""

from __future__ import annotations

# pylint: disable=protected-access

from pathlib import Path
from typing import Any
from unittest.mock import patch

from rich.console import Console

from hosts.core.config import HostConfig
from hosts.core.runtime import HostRuntime


# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> HostRuntime:
    from kernel.testing.runtime_factory import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


def _invoke_models(
    tmp_path: Path,
    action: str = "show",
    force: bool = False,
    captured: dict[str, Any] | None = None,
) -> str:
    """Call the CLI _models_command with a stubbed runtime + core handler.

    ``captured`` (if provided) records the ``stripped`` slash string passed
    to ``handle_models_command`` so tests can assert the dispatch wiring.
    Returns the core handler's response text.
    """
    from hosts.cli import main as cli_main

    runtime = _make_runtime(tmp_path)

    async def _fake_handle_models(_runtime, stripped, message):
        if captured is not None:
            captured["stripped"] = stripped
            captured["message"] = message
        return f"[core-response for {stripped}]"

    with (
        patch.object(
            cli_main,
            "_initialize_runtime_or_exit",
            return_value=runtime,
        ),
        patch(
            "hosts.core.commands.handle_models_command",
            new=_fake_handle_models,
        ),
        patch.object(cli_main, "console", new=Console(quiet=True)),
    ):
        # When action == "show" and render_models succeeds, the core
        # handler is NOT called. For non-show actions (or a show fallback
        # failure) the delegation path runs. We force the show path to
        # fall through to the core handler by stubbing render_models to
        # raise, so the 'show' delegation is also observable.
        if action in {"", "show"}:
            with patch.object(
                cli_main,
                "render_models",
                side_effect=RuntimeError("no manifest"),
            ):
                cli_main._models_command(
                    action=action,
                    force=force,
                    debug=False,
                    log_level="",
                    log="",
                    no_log=False,
                    data_dir=str(tmp_path),
                    thread_dir="",
                    no_thread=False,
                    session_context="",
                    session_context_refresh=False,
                    timezone="",
                    state_snapshot="",
                    streaming="none",
                    host_log="",
                )
        else:
            cli_main._models_command(
                action=action,
                force=force,
                debug=False,
                log_level="",
                log="",
                no_log=False,
                data_dir=str(tmp_path),
                thread_dir="",
                no_thread=False,
                session_context="",
                session_context_refresh=False,
                timezone="",
                state_snapshot="",
                streaming="none",
                host_log="",
            )
    return captured["stripped"] if captured else ""


# ── dispatch wiring ───────────────────────────────────────────────────


def test_models_discover_delegates_to_core(tmp_path: Path) -> None:
    """'discover' action delegates to handle_models_command with the slash string."""
    captured: dict[str, Any] = {}
    stripped = _invoke_models(tmp_path, action="discover", captured=captured)
    assert stripped == "/models discover"
    assert captured["message"] == "/models discover"


def test_models_discover_force_passes_flag(tmp_path: Path) -> None:
    """'discover --force' builds the '/models discover --force' slash string."""
    captured: dict[str, Any] = {}
    stripped = _invoke_models(tmp_path, action="discover", force=True, captured=captured)
    assert stripped == "/models discover --force"


def test_models_apply_delegates_to_core(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    stripped = _invoke_models(tmp_path, action="apply", captured=captured)
    assert stripped == "/models apply"


def test_models_help_delegates_to_core(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    stripped = _invoke_models(tmp_path, action="help", captured=captured)
    assert stripped == "/models help"


def _invoke_rejected_models(
    tmp_path: Path,
    action: str,
) -> list[str]:
    """Invoke the CLI _models_command for a REPL-only action and capture console lines.

    'reload' and 'edit' are rejected with a guidance message instead of
    delegating to the core handler, so they should NOT call
    handle_models_command.
    """
    from hosts.cli import main as cli_main

    runtime = _make_runtime(tmp_path)
    printed: list[str] = []

    class _CapturingConsole:
        def print(self, *args, **kwargs):  # noqa: D401, ANN002
            # Rich print renders markup; capture the plain text of the
            # first positional arg (a str here, not a Rich renderable).
            for a in args:
                if isinstance(a, str):
                    printed.append(a)

    handler_calls: list[str] = []

    async def _fake_handle_models(_runtime, stripped, message):
        handler_calls.append(stripped)
        return ""

    with (
        patch.object(
            cli_main,
            "_initialize_runtime_or_exit",
            return_value=runtime,
        ),
        patch(
            "hosts.core.commands.handle_models_command",
            new=_fake_handle_models,
        ),
        patch.object(cli_main, "console", new=_CapturingConsole()),
    ):
        cli_main._models_command(
            action=action,
            force=False,
            debug=False,
            log_level="",
            log="",
            no_log=False,
            data_dir=str(tmp_path),
            thread_dir="",
            no_thread=False,
            session_context="",
            session_context_refresh=False,
            timezone="",
            state_snapshot="",
            streaming="none",
            host_log="",
        )
    assert handler_calls == [], f"{action!r} must NOT delegate to handle_models_command"
    return printed


def test_models_reload_rejected_with_guidance(tmp_path: Path) -> None:
    """'reload' is REPL-only; the CLI prints guidance instead of delegating."""
    printed = _invoke_rejected_models(tmp_path, action="reload")
    assert any("reload" in line and "no durable effect" in line for line in printed)
    assert any("lisa models apply" in line for line in printed)


def test_models_edit_rejected_with_guidance(tmp_path: Path) -> None:
    """'edit' is REPL-only; the CLI prints guidance instead of delegating."""
    printed = _invoke_rejected_models(tmp_path, action="edit")
    assert any("edit" in line and "no durable effect" in line for line in printed)
    assert any("lisa models apply" in line for line in printed)


def test_models_show_falls_back_to_core_on_render_failure(
    tmp_path: Path,
) -> None:
    """When render_models raises (no active manifest), 'show' delegates to core."""
    captured: dict[str, Any] = {}
    stripped = _invoke_models(tmp_path, action="show", captured=captured)
    assert stripped == "/models show"


def test_models_show_renders_table_when_manifest_present(
    tmp_path: Path,
) -> None:
    """When render_models succeeds, 'show' prints the Rich table and skips core."""
    from hosts.cli import main as cli_main

    runtime = _make_runtime(tmp_path)
    rendered: list[bool] = []

    def _fake_render_models(_runtime, *, console=None):
        rendered.append(True)
        return None

    with (
        patch.object(
            cli_main,
            "_initialize_runtime_or_exit",
            return_value=runtime,
        ),
        patch.object(cli_main, "render_models", new=_fake_render_models),
        patch.object(cli_main, "console", new=Console(quiet=True)),
    ):
        cli_main._models_command(
            action="show",
            force=False,
            debug=False,
            log_level="",
            log="",
            no_log=False,
            data_dir=str(tmp_path),
            thread_dir="",
            no_thread=False,
            session_context="",
            session_context_refresh=False,
            timezone="",
            state_snapshot="",
            streaming="none",
            host_log="",
        )
    assert rendered == [True]
