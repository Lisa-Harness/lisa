"""Tests for the `desktop` CLI subcommand that launches the web host.

These tests verify CLI registration behavior, so they live under
``hosts/cli/tests`` (not ``hosts/web/tests``) — the CLI is the top-level
orchestrator that registers and launches every host, including web.
See docs/BOUNDARIES.md.
"""

from __future__ import annotations


def test_cli_desktop_command_is_registered() -> None:
    """The main `lisa` CLI exposes a `desktop` subcommand for the web host."""
    from hosts.cli.main import app as cli_app

    names = {cmd.name for cmd in cli_app.registered_commands}
    assert "desktop" in names


def test_cli_desktop_command_help_lists_web_flags() -> None:
    import typer.testing

    from hosts.cli.main import app as cli_app

    runner = typer.testing.CliRunner()
    result = runner.invoke(cli_app, ["desktop", "--help"])
    assert result.exit_code == 0
    out = result.output
    assert "--port" in out
    assert "--host" in out
    assert "--token" in out
    assert "--no-open-browser" in out