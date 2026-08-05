"""Tests for the ``_render_turn_footer`` post-response summary in the REPL.

The footer renders a one-line ``mode · model · tokens · elapsed`` summary
sourced from ``runtime.last_turn_context`` plus a handoff note when the
model switched modes mid-turn.
"""

from __future__ import annotations

# pylint: disable=protected-access
import io
from pathlib import Path

from host.runtime import HostRuntime, TurnContext
from hosts.cli.repl import _render_turn_footer
from rich.console import Console


def _make_runtime(tmp_path: Path) -> HostRuntime:
    from kernel.testing.runtime_factory import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


def _make_console() -> Console:
    # A StringIO-backed console records output via capture() (quiet=True
    # suppresses capture entirely).
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def test_footer_quiet_when_no_completion_context(tmp_path: Path) -> None:
    """No last_turn_context (or no completion dict) means no footer."""
    runtime = _make_runtime(tmp_path)
    console = _make_console()
    runtime.last_turn_context = None
    with console.capture() as capture:
        _render_turn_footer(console, runtime, elapsed_ms=340.0)
    assert capture.get().strip() == ""


def test_footer_renders_mode_model_tokens_time(tmp_path: Path) -> None:
    """The footer shows mode, model, total tokens, and elapsed ms."""
    runtime = _make_runtime(tmp_path)
    console = _make_console()
    runtime.last_turn_context = TurnContext(
        mode_id="coding",
        completion={
            "mode_id": "coding",
            "total_tokens": 1200,
            "resolved_model": {"model_id": "haiku", "provider": "anthropic"},
        },
    )
    runtime.current_mode_id = "coding"
    with console.capture() as capture:
        _render_turn_footer(console, runtime, elapsed_ms=340.0)
    out = capture.get().strip()
    assert "▾ coding" in out
    assert "haiku" in out
    assert "1200 tok" in out
    assert "340ms" in out
    # No handoff — current == entry mode.
    assert "→" not in out


def test_footer_shows_handoff_when_mode_changed_mid_turn(tmp_path: Path) -> None:
    """When runtime.current_mode_id differs from the entry mode, the footer
    appends ``→  <new mode>`` so the mid-turn transition is visible.
    """
    runtime = _make_runtime(tmp_path)
    console = _make_console()
    runtime.last_turn_context = TurnContext(
        mode_id="default",
        completion={
            "mode_id": "default",
            "total_tokens": 900,
            "resolved_model": {"model_id": "haiku", "provider": "anthropic"},
        },
    )
    # Simulate handle_message having applied a mid-turn handoff.
    runtime.current_mode_id = "coding"
    with console.capture() as capture:
        _render_turn_footer(console, runtime, elapsed_ms=500.0)
    out = capture.get().strip()
    assert "▾ default" in out
    assert "→ coding" in out
