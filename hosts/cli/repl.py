"""Interactive REPL for the CLI host.

This is the terminal transport's read-eval-print loop. It wraps the
transport-neutral :func:`hosts.core.handle_message` with a Rich-based
:class:`hosts.cli.transport.RichTransport` and a prompt_toolkit-backed
input session (with fallback to plain ``console.input``).

Lifecycle::

    runtime = HostRuntime(config)
    await runtime.start()           # starts MCP pool in this event loop
    await run_repl(runtime)         # runs the loop until /quit or EOF
    await runtime.aclose()          # closes MCP pool in same loop

The REPL is transport-specific: it owns the Rich console, the status line,
the prompt, and the session-exit summary. It delegates all routing and
turn execution to ``hosts.core.handle_message``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.text import Text

from hosts.cli.transport import RichTransport
from hosts.core.callbacks import TransportCallbacks
from hosts.core.handle_message import handle_message
from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)

# ── Visual style constants (CLI host palette) ─────────────────────────
STYLE_TEXT_PRIMARY = "bright_white"
STYLE_TEXT_MUTED = "grey58"
STYLE_BORDER_SESSION = "steel_blue3"
STYLE_BORDER_RESPONSE = "dark_sea_green4"
STYLE_BORDER_CONTEXT = "gold3"
STYLE_RULE = "grey42"
STYLE_PROMPT = "bold steel_blue3"
STYLE_STATUS_INFO = "bright_white on steel_blue3"
STYLE_STATUS_METRIC = "bright_white on grey27"
STYLE_STATUS_DEBUG_ON = "black on dark_khaki"
STYLE_STATUS_DEBUG_OFF = "bright_white on grey35"
STYLE_STATUS_SEPARATOR = "grey42"

# ── Status-line compact levels ───────────────────────────────────────
_STATUS_COMPACT_FULL = 0
_STATUS_COMPACT_ABBREVIATED = 1
_STATUS_COMPACT_TRIMMED = 2
_STATUS_COMPACT_MINIMAL = 3
_TIMEZONE_INITIALISM_MIN_LEN = 2
_STATUS_TIME_PADDING_WIDTH = 2

# ── REPL file history ────────────────────────────────────────────────
_REPL_HISTORY_FILE_NAME = "repl_history.txt"

# ── Host-only REPL shortcuts (not catalog modes) ────────────────────
_HOST_REPL_SHORTCUTS: tuple[str, ...] = (
    "/help",
    "/help capabilities",
    "/pack list",
    "/pack enable ",
    "/pack disable ",
    "/pack install ",
    "/pack uninstall ",
    "/models show",
    "/models discover",
    "/models apply",
    "/models reload",
    "/models edit",
    "/settings",
    "/tokens",
    "/debug context",
    "/debug frontdoor",
    "/consolidate",
    "/checkpoint",
    "/restore",
    "/checkpoints",
    "/deep ",
    "/quit",
)


# ── prompt_toolkit: optional dependency with graceful fallback ───────
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
    from prompt_toolkit.history import FileHistory, History
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
except ImportError:  # pragma: no cover - optional dependency
    PromptSession = None
    WordCompleter = None
    FileHistory = None
    History = None
    KeyBindings = None
    CursorShape = None
    ModalCursorShapeConfig = None
    Style = None


class _SurrogateSafeFileHistory(History):
    """FileHistory wrapper that sanitises unpaired surrogates before persisting.

    On Windows, the console can emit unpaired surrogate characters for emoji.
    Python's utf-8 codec rejects those, which crashes prompt_toolkit's
    FileHistory.store_string. This wrapper strips them.
    """

    def __init__(self, filename: str) -> None:
        super().__init__()
        self._inner = FileHistory(filename)

    def load_history_strings(self) -> list[str]:
        return self._inner.load_history_strings()

    def store_string(self, string: str) -> None:
        safe = string.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
        self._inner.store_string(safe)

    def append_string(self, string: str) -> None:
        safe = string.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
        self._inner.append_string(safe)


def _repl_command_hints() -> tuple[str, ...]:
    """Build REPL completion hints from host shortcuts plus catalog mode commands."""
    from kernel.ipsar_fsm import get_workflow_catalog
    from hosts.core.commands import mode_support_gap

    catalog = get_workflow_catalog()
    mode_hints: list[str] = []
    for mode_id, mode in sorted(catalog.items()):
        if mode_support_gap(mode) is not None:
            continue
        if mode.command_pattern and mode_id != "default":
            mode_hints.append(f"/{mode_id} ")
    return (*_HOST_REPL_SHORTCUTS, *mode_hints)


async def run_repl(runtime: HostRuntime, *, console: Console | None = None) -> bool:
    """Run the interactive REPL loop.

    Args:
        runtime: A constructed ``HostRuntime`` whose ``start()`` has not
            yet been called (this function calls ``start()`` and ``aclose()``).
        console: Optional Rich Console. Defaults to a new ``Console()``.

    Returns:
        ``True`` if a restart was requested (caller may re-exec), ``False``
        on clean exit.
    """
    console = console or Console()
    # Stash the runtime on the console so the status-line renderer can read
    # route preview + token counters without threading the runtime through
    # every status-segment call.
    console._lisa_runtime = runtime  # type: ignore[attr-defined]
    transport = RichTransport(console=console, debug=runtime.config.debug)

    # Set the initial route preview so the status line shows the
    # resolved model before the first turn is executed. When start_mode
    # is set in the session context, use it instead of "default".
    from hosts.core.commands import set_runtime_route_preview
    from hosts.core.commands import frontdoor_model_alias, model_alias_for_mode

    initial_mode = runtime.config.start_mode or "default"
    initial_model_alias = (
        model_alias_for_mode(initial_mode)
        if runtime.config.start_mode
        else frontdoor_model_alias()
    )
    set_runtime_route_preview(
        runtime,
        mode_id=initial_mode,
        preferred_model_id=initial_model_alias,
    )

    # Start the MCP pool in the background so the prompt appears immediately
    # while warm servers spawn. The first turn (or any slash command reading
    # the pool) awaits ensure_pool_ready() in handle_message before touching
    # the pool. Anyio cancel-scope binding is preserved: the task runs on
    # this event loop and is awaited on this same loop.
    try:
        runtime.start_pool_deferred()
    except Exception as exc:  # pylint: disable=broad-except
        _render_runtime_error(console, exc, title="MCP Startup Error")
        return False

    _print_startup_banner(console)
    console.print(
        Panel(
            "Type /help, /help capabilities, /debug context, "
            "/debug frontdoor, /pack list, /models discover, /search <query>, /deep <query>, or /quit.",
            title="Session",
            border_style=STYLE_BORDER_SESSION,
        )
    )

    prompt_session = _build_repl_prompt_session(runtime, console)
    if prompt_session is None:
        console.print(f"[{STYLE_TEXT_MUTED}]prompt_toolkit unavailable; using basic input mode.[/]")

    # Replay a pending resume message (set by `/pack ... --resume <msg>` in
    # the prior session via the LISA_RESTART_RESUME_MESSAGE env var) as the
    # first turn. Cleared after one replay.
    pending_resume = runtime.pending_resume_message
    runtime.pending_resume_message = None

    restart_requested = False
    try:
        if pending_resume:
            console.print(
                Panel(
                    f"Resuming after restart: {pending_resume}",
                    title="Resume",
                    border_style=STYLE_BORDER_SESSION,
                )
            )
            await _process_repl_turn(runtime, transport, console, pending_resume)
            if runtime.restart_required:
                _render_restart_required_notice_if_needed(runtime, console)
                if not _confirm_restart(runtime, console):
                    runtime.restart_required = False
                    runtime.restart_reason = ""
                    console.print(
                        f"[{STYLE_TEXT_MUTED}]Restart deferred. "
                        f"Run /quit then relaunch `lisa` manually when ready.[/]"
                    )
                else:
                    _render_session_exit_summary(runtime, console)
                    console.print(
                        f"[{STYLE_TEXT_MUTED}]Applying restart now...[/]"
                    )
                    return True
        restart_requested = await _run_repl_loop(runtime, transport, console, prompt_session)
    finally:
        await runtime.aclose()

    return restart_requested


async def _run_repl_loop(
    runtime: HostRuntime,
    transport: TransportCallbacks,
    console: Console,
    prompt_session: Any,
) -> bool:
    """Process REPL turns until the user quits or sends EOF."""
    while True:
        message = await _read_repl_input(console, prompt_session)
        if message is None:
            _render_session_exit_summary(runtime, console)
            return False
        if not message:
            continue
        if _is_repl_quit_command(message):
            _render_session_exit_summary(runtime, console)
            console.print(f"[{STYLE_TEXT_MUTED}]Session closed.[/]")
            return False
        await _process_repl_turn(runtime, transport, console, message)
        if runtime.restart_required:
            _render_restart_required_notice_if_needed(runtime, console)
            if not _confirm_restart(runtime, console):
                runtime.restart_required = False
                runtime.restart_reason = ""
                console.print(
                    f"[{STYLE_TEXT_MUTED}]Restart deferred. "
                    f"Run /quit then relaunch `lisa` manually when ready.[/]"
                )
                continue
            _render_session_exit_summary(runtime, console)
            if os.name == "nt":
                console.print(
                    f"[{STYLE_TEXT_MUTED}]Please restart `lisa` manually to "
                    f"reload packs and MCP manifests.[/]"
                )
            elif runtime.restart_reason:
                console.print(f"[{STYLE_TEXT_MUTED}]Applying restart now...[/]")
            else:
                console.print(
                    f"[{STYLE_TEXT_MUTED}]Restarting is required before continuing this session.[/]"
                )
            return True


async def _read_repl_input(console: Console, prompt_session: Any) -> str | None:
    """Read one line of REPL input, returning None on EOF."""
    try:
        _render_status_line(console)
        if prompt_session is not None:
            cursor = (
                ModalCursorShapeConfig()
                if ModalCursorShapeConfig is not None
                else None
            )
            return (
                await prompt_session.prompt_async(
                    [(f"class:prompt", "❯ ")],
                    cursor=cursor,
                )
            ).strip()
        return console.input(f"[{STYLE_PROMPT}]❯ [/]").strip()
    except KeyboardInterrupt:
        console.print()
        return ""
    except EOFError:
        console.print(Rule(style=STYLE_RULE))
        return None


def _is_repl_quit_command(message: str) -> bool:
    """Return True if the message is a quit/exit command."""
    return message.lower() in {"exit", "quit", "/exit", "/quit"}


async def _process_repl_turn(
    runtime: HostRuntime,
    transport: TransportCallbacks,
    console: Console,
    message: str,
) -> None:
    """Execute one REPL turn via handle_message and render the response."""
    turn_start = time.perf_counter()
    # Start the thinking spinner immediately so the terminal is not dead
    # between submit and the first pipeline stage callback. The spinner
    # is stopped by the transport on the first on_stage/on_response/
    # on_error/on_text_chunk.
    if isinstance(transport, RichTransport):
        transport.start_thinking("routing")
    try:
        result = await handle_message(runtime, message, transport)
    except Exception as exc:  # pylint: disable=broad-except
        if isinstance(transport, RichTransport):
            transport.stop_thinking()
        elapsed_ms = (time.perf_counter() - turn_start) * 1000.0
        runtime.tokens.session_error_count += 1
        runtime.tokens.session_total_latency_ms += elapsed_ms
        logger.error("REPL turn failed: %s: %s", type(exc).__name__, exc)
        _render_runtime_error(console, exc, title="Turn Error")
        console.print(Rule(style=STYLE_RULE))
        return
    elapsed_ms = (time.perf_counter() - turn_start) * 1000.0
    runtime.tokens.session_turn_count += 1
    runtime.tokens.session_success_count += 1
    runtime.tokens.session_total_latency_ms += elapsed_ms
    mode_id = runtime.current_mode_id
    runtime.tokens.session_mode_counts[mode_id] = (
        runtime.tokens.session_mode_counts.get(mode_id, 0) + 1
    )

    # Debug context is delivered via transport.on_debug_context during the
    # turn; render the last-turn context here too when debug is on, to
    # match the Little Lisa post-turn panel.
    if runtime.config.debug and runtime.last_turn_context is not None:
        _render_turn_context(console, runtime.last_turn_context)

    # The transport's on_response already rendered the response panel during
    # the turn. If on_response was a no-op (e.g. NullTransport), render here
    # as a fallback — but RichTransport already printed it.
    if result.text and not isinstance(transport, RichTransport):
        await transport.on_response(result.text)

    console.print(Rule(style=STYLE_RULE))


# ── Rendering helpers ────────────────────────────────────────────────


def _print_startup_banner(console: Console) -> None:
    """Print a compact startup banner."""
    title = Text()
    title.append("Ready", style="bold steel_blue3")
    title.append("  ")
    title.append("interactive terminal host", style=STYLE_TEXT_MUTED)
    console.print(title)
    console.print()


def _render_turn_context(console: Console, context: Any) -> None:
    """Render a turn-context dict or TurnContext as a Rich Panel."""
    if isinstance(context, dict):
        payload = context
    else:
        # TurnContext dataclass → dict
        payload = {
            k: getattr(context, k)
            for k in (
                "timestamp_utc",
                "raw_message",
                "mode_id",
                "mode_kind",
                "query",
                "deep_route",
                "preferred_model_id",
                "handoff",
            )
        }
    import json

    console.print(
        Panel(
            json.dumps(payload, indent=2, ensure_ascii=True, default=str),
            title="Turn Context",
            border_style=STYLE_BORDER_CONTEXT,
        )
    )


def _render_runtime_error(
    console: Console, exc: Exception, *, title: str = "Error"
) -> None:
    error_text = f"{type(exc).__name__}: {exc}"
    logger.error("%s: %s: %s", title, type(exc).__name__, exc)
    console.print(
        Panel(
            error_text,
            title=title,
            border_style="indian_red",
        )
    )


def _render_restart_required_notice_if_needed(
    runtime: HostRuntime, console: Console
) -> None:
    """Render a one-shot restart-required notice after pack lifecycle changes."""
    if not runtime.restart_required:
        return
    reason = runtime.restart_reason.strip() or "runtime change"
    console.print(
        Panel(
            "Restart required to reload entry points, pack catalogs, and MCP manifests.\n"
            f"Reason: {reason}",
            title="Restart Required",
            border_style="gold3",
        )
    )


def _confirm_restart(runtime: HostRuntime, console: Console) -> bool:
    """Ask the user yes/no before exiting the session to apply a restart.

    Returns ``True`` to proceed with the restart (exit/relaunch), or
    ``False`` to defer and keep the current session alive. On EOF
    (Ctrl+D) or KeyboardInterrupt, the restart is deferred so the
    user is not forced out of the session unexpectedly.
    """
    reason = runtime.restart_reason.strip() or "runtime change"
    prompt_text = f"Exit now to reload ({reason})? [Y/n]"
    try:
        return bool(
            Confirm.ask(
                prompt_text,
                console=console,
                default=True,
                show_choices=False,
                show_default=False,
            )
        )
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False


def _render_session_exit_summary(runtime: HostRuntime, console: Console) -> None:
    """Render the session-exit summary panel from the runtime's token counters."""
    from hosts.core.commands import format_token_stats

    summary = format_token_stats(runtime)
    console.print(
        Panel(
            Markdown(summary),
            title="Session",
            border_style=STYLE_BORDER_SESSION,
        )
    )


def _build_repl_prompt_session(runtime: HostRuntime, console: Console) -> Any:
    """Build a prompt_toolkit PromptSession with history + completion.

    Returns ``None`` when prompt_toolkit is unavailable. When available,
    the session is multi-line: Enter submits the turn, Esc→Enter inserts
    a newline (chat-box semantics). A bottom toolbar shows live char/line
    count and control hints while the prompt is active.
    """
    if PromptSession is None or FileHistory is None or WordCompleter is None:
        return None
    history_path = runtime.data_dir / _REPL_HISTORY_FILE_NAME
    history_path.parent.mkdir(parents=True, exist_ok=True)
    bindings = _build_repl_key_bindings() if KeyBindings is not None else None
    return PromptSession(
        history=_SurrogateSafeFileHistory(str(history_path)),
        completer=WordCompleter(list(_repl_command_hints()), ignore_case=True),
        complete_while_typing=True,
        multiline=True,
        key_bindings=bindings,
        prompt_continuation=[("class:prompt-continuation", "… ")],
        bottom_toolbar=_repl_bottom_toolbar,
        style=Style.from_dict(_REPL_PROMPT_STYLE) if Style is not None else None,
    )


# prompt_toolkit style: maps the class: names used in the prompt and
# toolbar to hex colors (prompt_toolkit does not accept Rich color names
# like 'steel_blue3'). Steel_blue3 is ANSI-256 #74 -> #5fafff; grey58
# is #5f5f5f; grey39 is #626262.
_REPL_PROMPT_STYLE = {
    "prompt": "bold #5fafff",
    "prompt-continuation": "#5f5f5f",
    "bottom-toolbar": "reverse",
    "bottom-toolbar.text": "#626262",
    "bottom-toolbar.metric": "bold #5fafff",
}


def _repl_bottom_toolbar() -> Any:
    """Render the bottom toolbar: live char/line count + control hints.

    Called by prompt_toolkit on every buffer mutation. Returns
    ``FormattedText`` (a list of ``(style, text)`` tuples) so the toolbar
    updates in real time as the user types.
    """
    # Import inside the function so the module imports cleanly when
    # prompt_toolkit is absent (the toolbar is only used when it is present).
    from prompt_toolkit.formatted_text import FormattedText

    # prompt_toolkit stores the active buffer on the current application.
    # We reach it via get_app() rather than capturing a session closure so
    # the toolbar works regardless of which PromptSession calls it.
    try:
        from prompt_toolkit.application import get_app
        buffer = get_app().current_buffer
        text = buffer.text
    except Exception:  # pylint: disable=broad-except
        text = ""

    char_count = len(text)
    line_count = text.count("\n") + 1 if text else 0

    return FormattedText([
        ("class:bottom-toolbar", " Enter submit · Esc+Enter newline · ↑/↓ history · Tab complete "),
        ("class:bottom-toolbar.text", "│ "),
        ("class:bottom-toolbar.metric", f"{line_count} line{'s' if line_count != 1 else ''}, {char_count} char{'s' if char_count != 1 else ''}"),
    ])


def _build_repl_key_bindings() -> Any:
    """Build chat-box key bindings for the multi-line REPL prompt.

    - Enter: submit the current buffer (chat-box semantics).
    - Esc→Enter: insert a newline (for multi-line input).

    Without these bindings, prompt_toolkit's default ``multiline=True``
    behaviour is the reverse (Enter inserts a newline; Esc→Enter submits),
    which surprises chat users.
    """
    if KeyBindings is None:
        return None
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


# ── Status line ──────────────────────────────────────────────────────


def _render_status_line(console: Console) -> None:
    """Render a powerline-style status row above the input prompt."""
    from rich.cells import cell_len
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    from datetime import datetime

    left = Text()
    segments = _fitted_status_segments(console)
    for index, (label, style) in enumerate(segments):
        left.append(f" {label} ", style=style)
        if index < len(segments) - 1:
            left.append("", style=STYLE_STATUS_SEPARATOR)

    time_label, time_style = _status_time_badge(console)
    right = Text(f" {time_label} ", style=time_style)

    left_width = cell_len(left.plain)
    right_width = cell_len(right.plain)
    padding = max(1, int(console.size.width) - left_width - right_width)
    line = Text()
    line.append(left)
    line.append(" " * padding)
    line.append(right)
    console.print(line, no_wrap=True)


def _status_segments(
    runtime_console: Console, compact_level: int = _STATUS_COMPACT_FULL
) -> list[tuple[str, str]]:
    """Build status segments from the runtime's route preview + token counters.

    Note: this reads from a module-global placeholder runtime set by
    ``run_repl``. We avoid threading the runtime through every status call
    by stashing it on the console via a private attribute.
    """
    runtime: HostRuntime | None = getattr(runtime_console, "_lisa_runtime", None)
    if runtime is None:
        return [("init", STYLE_STATUS_METRIC)]

    user = (runtime.config.user_slug or "").strip()
    user_segment = (f"👤 {user}", STYLE_STATUS_INFO) if user else None
    mode_id = runtime.current_mode_id
    stage_id = runtime.current_stage_id
    mode_badge = f"mode {mode_id}{'' if stage_id == '-' else '/' + stage_id}"
    mode_badge_short = f"m {mode_id}{'' if stage_id == '-' else '/' + stage_id}"
    model_badge = f"{runtime.next_model_provider}:{runtime.next_model_id}"
    turn_token_total = runtime.tokens.turn_total_tokens
    session_token_total = runtime.tokens.session_total_tokens
    token_badge = f"tokens {turn_token_total}/{session_token_total}"
    token_badge_short = f"tc {turn_token_total}/{session_token_total}"
    debug_enabled = runtime.config.debug

    if compact_level <= _STATUS_COMPACT_FULL:
        return _finalize_status_segments(
            segments=[
                (mode_badge, STYLE_STATUS_METRIC),
                (model_badge, STYLE_STATUS_METRIC),
                (token_badge, STYLE_STATUS_METRIC),
            ],
            user_segment=user_segment,
            debug_enabled=debug_enabled,
            abbreviated_debug=False,
        )
    if compact_level == _STATUS_COMPACT_ABBREVIATED:
        return _finalize_status_segments(
            segments=[
                (mode_badge_short, STYLE_STATUS_METRIC),
                (model_badge, STYLE_STATUS_METRIC),
                (token_badge_short, STYLE_STATUS_METRIC),
            ],
            user_segment=user_segment,
            debug_enabled=debug_enabled,
            abbreviated_debug=True,
        )
    if compact_level == _STATUS_COMPACT_TRIMMED:
        return _finalize_status_segments(
            segments=[
                (mode_badge_short, STYLE_STATUS_METRIC),
                (model_badge, STYLE_STATUS_METRIC),
                (token_badge_short, STYLE_STATUS_METRIC),
            ],
            user_segment=None,
            debug_enabled=debug_enabled,
            abbreviated_debug=True,
        )
    return _finalize_status_segments(
        segments=[
            (mode_badge_short, STYLE_STATUS_METRIC),
            (model_badge, STYLE_STATUS_METRIC),
            (token_badge_short, STYLE_STATUS_METRIC),
        ],
        user_segment=None,
        debug_enabled=debug_enabled,
        abbreviated_debug=True,
    )


def _finalize_status_segments(
    *,
    segments: list[tuple[str, str]],
    user_segment: tuple[str, str] | None,
    debug_enabled: bool,
    abbreviated_debug: bool,
) -> list[tuple[str, str]]:
    if user_segment is not None:
        segments.insert(1, user_segment)
    if debug_enabled:
        debug_label = "dbg on" if abbreviated_debug else "debug on"
        segments.append((debug_label, STYLE_STATUS_DEBUG_ON))
    return segments


def _status_time_badge(console: Console) -> tuple[str, str]:
    """Build the timezone-aware clock badge for the status line."""
    from datetime import datetime
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    runtime: HostRuntime | None = getattr(console, "_lisa_runtime", None)
    timezone_name = runtime.config.timezone if runtime is not None else "UTC"
    try:
        status_dt = datetime.now(ZoneInfo(timezone_name))
    except ZoneInfoNotFoundError:
        status_dt = datetime.now().astimezone()
    timezone_label = timezone_name
    if "/" in timezone_name:
        tz_abbrev_raw = str(status_dt.tzname() or "").strip()
        if tz_abbrev_raw:
            if " " in tz_abbrev_raw:
                initials = "".join(
                    token[0] for token in tz_abbrev_raw.split() if token and token[0].isalpha()
                ).upper()
                timezone_label = (
                    initials
                    if len(initials) >= _TIMEZONE_INITIALISM_MIN_LEN
                    else tz_abbrev_raw.upper()
                )
            else:
                timezone_label = tz_abbrev_raw.upper()
        elif timezone_label == timezone_name:
            tail = timezone_name.rsplit("/", 1)[-1]
            timezone_label = (
                "".join(part[:1] for part in tail.split("_") if part).upper() or tail
            )
        if " " in timezone_label:
            is_daylight = bool(status_dt.dst())
            us_zone_abbrevs = {
                "America/New_York": ("EST", "EDT"),
                "America/Chicago": ("CST", "CDT"),
                "America/Denver": ("MST", "MDT"),
                "America/Los_Angeles": ("PST", "PDT"),
            }
            pair = us_zone_abbrevs.get(timezone_name)
            if pair is not None:
                timezone_label = pair[1] if is_daylight else pair[0]
    now_local = status_dt.strftime("%H:%M:%S")
    return f"🕒 {now_local} {timezone_label}", STYLE_STATUS_INFO


def _estimated_status_line_width(segments: list[tuple[str, str]]) -> int:
    """Estimate rendered width for status segments and separators."""
    from rich.cells import cell_len

    if not segments:
        return 0
    label_width = sum(cell_len(label) + 2 for label, _ in segments)
    separator_width = max(0, len(segments) - 1)
    return label_width + separator_width


def _fitted_status_segments(console: Console) -> list[tuple[str, str]]:
    """Return the richest status segments that fit the current terminal width."""
    from rich.cells import cell_len

    time_label, _ = _status_time_badge(console)
    time_width = cell_len(time_label) + _STATUS_TIME_PADDING_WIDTH
    max_width = max(0, int(console.size.width) - time_width - 1)
    for compact_level in (
        _STATUS_COMPACT_FULL,
        _STATUS_COMPACT_ABBREVIATED,
        _STATUS_COMPACT_TRIMMED,
        _STATUS_COMPACT_MINIMAL,
    ):
        segments = _status_segments(console, compact_level)
        if _estimated_status_line_width(segments) <= max_width:
            return segments
    return _status_segments(console, _STATUS_COMPACT_MINIMAL)


__all__ = ["run_repl"]