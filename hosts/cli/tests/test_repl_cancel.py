"""Tests for mid-turn Ctrl+C cancellation in the CLI REPL.

Covers the contract that a ``KeyboardInterrupt`` raised during
``handle_message`` is caught by ``_process_repl_turn`` so the prompt is
returned quickly without tearing down the session (the warm MCP pool,
thread, and session-level accumulators survive). Before this handler
existed, ``KeyboardInterrupt`` (a ``BaseException``) propagated past
``_run_repl_loop`` to ``run_repl``'s ``finally: await runtime.aclose()``,
killing the whole session on every mid-turn Ctrl+C.
"""

from __future__ import annotations

# pylint: disable=protected-access
from pathlib import Path
from typing import Any

import pytest
from host.runtime import HostRuntime

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path."""
    from kernel.testing.runtime_factory import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


# ── _process_repl_turn: KeyboardInterrupt ─────────────────────────────


def test_process_repl_turn_keyboard_interrupt_returns_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C mid-turn cancels the turn and returns without re-raising.

    A KeyboardInterrupt raised by handle_message must be caught, the
    spinner stopped, a cancellation marker printed, and the function
    return normally so the REPL loop re-renders the prompt. The session
    must NOT be torn down (no aclose on the runtime).
    """
    from hosts.cli import repl as cli_repl
    from hosts.cli.transport import RichTransport
    from rich.console import Console

    runtime = _make_runtime(tmp_path)
    console = Console(quiet=True)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")

    async def _raise_interrupt(*a: Any, **k: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_repl, "handle_message", _raise_interrupt)

    aclose_called: list[Any] = []

    async def _aclose(*a: Any, **k: Any) -> None:
        aclose_called.append(True)

    monkeypatch.setattr(HostRuntime, "aclose", _aclose)

    # Must not raise.
    import asyncio

    asyncio.run(cli_repl._process_repl_turn(runtime, transport, console, "typo msg"))

    # Spinner was stopped.
    assert transport._status is None
    # Session-level accounting: turn counted as cancelled, not errored.
    assert runtime.tokens.session_turn_count == 1
    assert runtime.tokens.session_cancelled_count == 1
    assert runtime.tokens.session_error_count == 0
    assert runtime.tokens.session_success_count == 0
    # The runtime was NOT closed — the pool survives for the next turn.
    assert aclose_called == []
    # Route preview reset to the start mode with neutral model placeholders
    # (the same defaults HostRuntime.__init__ sets; the next turn repopulates).
    assert runtime.current_mode_id == runtime.start_mode_id
    assert runtime.next_model_id == "-"
    assert runtime.next_model_provider == "-"
    # last_turn_context cleared so a stale half-built context isn't surfaced.
    assert runtime.last_turn_context is None


def test_process_repl_turn_cancelled_error_returns_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """asyncio.CancelledError (from asyncio.run's SIGINT handler) is caught.

    On Python 3.11+, ``asyncio.run()`` installs a SIGINT handler that calls
    ``main_task.cancel()`` on the first Ctrl+C, which raises
    ``CancelledError`` inside the coroutine — not ``KeyboardInterrupt``.
    This is the primary interrupt path on all platforms. The handler must
    catch ``CancelledError``, render the cancellation marker, and return
    normally so the REPL loop re-renders the prompt.
    """
    import asyncio

    from hosts.cli import repl as cli_repl
    from hosts.cli.transport import RichTransport
    from rich.console import Console

    runtime = _make_runtime(tmp_path)
    console = Console(quiet=True)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("model_call")

    async def _raise_cancelled(*a: Any, **k: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(cli_repl, "handle_message", _raise_cancelled)

    asyncio.run(cli_repl._process_repl_turn(runtime, transport, console, "typo msg"))

    assert transport._status is None
    assert runtime.tokens.session_cancelled_count == 1
    assert runtime.tokens.session_error_count == 0
    assert runtime.current_mode_id == runtime.start_mode_id
    assert runtime.last_turn_context is None


def test_process_repl_turn_keyboard_interrupt_stops_spinner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The thinking spinner is stopped even when the turn is interrupted."""
    from hosts.cli import repl as cli_repl
    from hosts.cli.transport import RichTransport
    from rich.console import Console

    runtime = _make_runtime(tmp_path)
    console = Console(quiet=True)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("model_call")
    assert transport._status is not None

    async def _raise_interrupt(*a: Any, **k: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_repl, "handle_message", _raise_interrupt)

    import asyncio

    asyncio.run(cli_repl._process_repl_turn(runtime, transport, console, "x"))

    assert transport._status is None


def test_process_repl_turn_normal_exception_still_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal Exception is still counted as an error, not a cancellation."""
    from hosts.cli import repl as cli_repl
    from hosts.cli.transport import RichTransport
    from rich.console import Console

    runtime = _make_runtime(tmp_path)
    console = Console(quiet=True)
    transport = RichTransport(console=console, debug=False)

    async def _raise_value_error(*a: Any, **k: Any) -> Any:
        raise ValueError("boom")

    monkeypatch.setattr(cli_repl, "handle_message", _raise_value_error)

    import asyncio

    asyncio.run(cli_repl._process_repl_turn(runtime, transport, console, "x"))

    assert runtime.tokens.session_error_count == 1
    assert runtime.tokens.session_cancelled_count == 0
    assert runtime.tokens.session_success_count == 0


# ── _reset_runtime_after_cancelled_turn ──────────────────────────────


def test_reset_runtime_after_cancelled_turn_clears_per_turn_state(tmp_path: Path) -> None:
    """The reset helper clears per-turn counters, context, and route preview."""
    from hosts.cli import repl as cli_repl

    runtime = _make_runtime(tmp_path)
    # Dirty per-turn state from a half-run turn.
    runtime.tokens.turn_prompt_tokens = 123
    runtime.tokens.turn_completion_tokens = 456
    runtime.last_turn_context = object()  # type: ignore[assignment]
    runtime.current_mode_id = "search"
    runtime.next_model_id = "stale-model"

    cli_repl._reset_runtime_after_cancelled_turn(runtime)

    assert runtime.tokens.turn_prompt_tokens == 0
    assert runtime.tokens.turn_completion_tokens == 0
    assert runtime.last_turn_context is None
    assert runtime.current_mode_id == runtime.start_mode_id


# ── TokenCounters.session_cancelled_count ────────────────────────────


def test_token_counters_has_cancelled_field(tmp_path: Path) -> None:
    """TokenCounters exposes a session_cancelled_count field defaulting to 0."""
    runtime = _make_runtime(tmp_path)
    assert runtime.tokens.session_cancelled_count == 0
    runtime.tokens.session_cancelled_count += 1
    assert runtime.tokens.session_cancelled_count == 1


def test_session_exit_summary_includes_cancelled_when_present(tmp_path: Path) -> None:
    """format_session_exit_summary_lines surfaces cancelled count when > 0."""
    from host.commands import format_token_stats

    runtime = _make_runtime(tmp_path)
    runtime.tokens.session_cancelled_count = 2
    summary = format_token_stats(runtime)
    assert "cancelled: 2" in summary


def test_session_exit_summary_omits_cancelled_when_zero(tmp_path: Path) -> None:
    """When no turns were cancelled, the cancelled suffix is omitted."""
    from host.commands import format_token_stats

    runtime = _make_runtime(tmp_path)
    summary = format_token_stats(runtime)
    assert "cancelled" not in summary
