"""hosts.web.main — Typer app and entry points for the localhost web host.

This is the web-transport entry point for the Lisa framework. It wraps the
transport-neutral :mod:`hosts.core` runtime with an aiohttp web server and
a minimal Typer CLI.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Annotated, Any

import typer

from hosts.web.server import (
    DEFAULT_WEB_HOST,
    DEFAULT_WEB_PORT,
    generate_web_token,
    run_web_server,
)

if TYPE_CHECKING:
    from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run the Lisa framework localhost web host.",
)


def _get_version() -> str:
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("lisa-host-web")
    except Exception:  # pylint: disable=broad-except
        return "0.1.0"


_VERSION_OPTION = typer.Option(
    False,
    "--version",
    help="Print the lisa-web version and exit.",
    is_eager=True,
)

# ---------------------------------------------------------------------------

# Shared Typer option aliases â€” mirror hosts.cli.main so the web host
# accepts the same runtime flags as the terminal host.

DebugOption = Annotated[
    bool,
    typer.Option("--debug", help="Print turn context above each response."),
]
LogLevelOption = Annotated[
    str,
    typer.Option("--log-level", help="Log threshold: debug, info, warning, or error."),
]
LogOption = Annotated[str, typer.Option("--log", help="Append JSONL turn logs to this file.")]
NoLogOption = Annotated[bool, typer.Option("--no-log", help="Disable turn logging entirely.")]
HostLogOption = Annotated[
    str,
    typer.Option(
        "--host-log",
        help="Stdlib host log file. Default: <data-dir>/host.jsonl",
    ),
]
DataDirOption = Annotated[
    str,
    typer.Option(
        "--data-dir",
        help="Data root for artifacts, memory, and logs. Default: ~/.lisa",
    ),
]
OverlaysDirOption = Annotated[
    str,
    typer.Option(
        "--overlays-dir",
        help="User-authored overlays directory for mode and content overrides. Default: <data-dir>/overlays",
    ),
]
ThreadDirOption = Annotated[
    str,
    typer.Option("--thread-dir", help="Thread log directory."),
]
NoThreadOption = Annotated[bool, typer.Option("--no-thread", help="Disable thread logging.")]
SessionContextOption = Annotated[
    str,
    typer.Option(
        "--session-context",
        help=(
            "User context source: a slug name (content/user_contexts/<name>.md or downloaded from remote registry), "
            "a single remote URL, a full .md file path, or a directory (all files concatenated). "
            "Omit to skip session context."
        ),
    ),
]

SessionContextRefreshOption = Annotated[
    bool,
    typer.Option(
        "--session-context-refresh",
        help="Force re-fetch of remote session context, ignoring local cache.",
    ),
]

TimezoneOption = Annotated[
    str,
    typer.Option("--timezone", help="IANA timezone (e.g. UTC, America/New_York)."),
]
StateSnapshotOption = Annotated[
    str,
    typer.Option("--state-snapshot", help="Local JSON snapshot file for compact context."),
]
StreamingOption = Annotated[
    str,
    typer.Option("--streaming", help="Streaming mode: none, stage, text, hybrid."),
]

# â”€â”€ Web-specific options â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

WebHostOption = Annotated[
    str,
    typer.Option("--host", help="Bind address for the web server. Default: 127.0.0.1"),
]
WebPortOption = Annotated[
    int,
    typer.Option("--port", help="Bind port for the web server. Default: 8765"),
]
WebTokenOption = Annotated[
    str,
    typer.Option(
        "--token",
        help="Auth token for web API requests. Default: generated at startup.",
    ),
]
NoOpenBrowserOption = Annotated[
    bool,
    typer.Option("--no-open-browser", help="Do not open a browser tab on startup."),
]


# ---------------------------------------------------------------------------
# Runtime initialization (builds HostRuntime, not _RUNTIME_STATE)
# ---------------------------------------------------------------------------


def _initialize_runtime(
    *,
    debug: bool,
    log_level: str = "",
    log_path: str = "",
    no_log: bool = False,
    data_dir: str = "",
    overlays_dir: str = "",
    thread_dir: str = "",
    no_thread: bool = False,
    user_slug: str,
    timezone: str = "",
    state_snapshot_path: str,
    streaming: str = "",
    host_log_path: str = "",
) -> HostRuntime:
    """Build and wire a HostRuntime from CLI args + env vars.

    Thin adapter over :func:`hosts.core.bootstrap.build_runtime` that
    stamps ``host_kind="web"`` and forwards every web-host argument. The
    composition sequence (config -> runtime -> state snapshot -> framework
    wiring -> MCP config compilation) lives in the shared bootstrap so
    the CLI and web hosts cannot drift.

    The MCP pool is NOT started here â€” ``runtime.start()`` runs inside
    the aiohttp server's ``on_startup`` hook (see
    :func:`hosts.web.server.run_web_server`).
    """
    from hosts.core.bootstrap import build_runtime

    return build_runtime(
        host_kind="web",
        debug=debug,
        log_level=log_level,
        log_path=log_path,
        no_log=no_log,
        data_dir=data_dir,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=user_slug,
        timezone=timezone,
        session_context=user_slug,
        state_snapshot_path=state_snapshot_path,
        streaming=streaming,
        host_log_path=host_log_path,
    )


def _initialize_runtime_or_exit(**kwargs: Any) -> HostRuntime | None:
    """Initialize runtime, rendering startup errors and exiting on failure."""

    try:
        return _initialize_runtime(**kwargs)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Startup error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


# ---------------------------------------------------------------------------
# Typer callback (top-level version flag)
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _cli_callback(
    ctx: typer.Context,
    version: bool = _VERSION_OPTION,
) -> None:
    if version:
        typer.echo(f"lisa-web v{_get_version()}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# ---------------------------------------------------------------------------
# desktop command (web server)
# ---------------------------------------------------------------------------


@app.command("desktop", help="Run the localhost web host with the desktop chat UI.")
def _desktop_command(
    web_host: WebHostOption = DEFAULT_WEB_HOST,
    web_port: WebPortOption = DEFAULT_WEB_PORT,
    web_token: WebTokenOption = "",
    no_open_browser: NoOpenBrowserOption = False,
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "text",
    host_log: HostLogOption = "",
) -> None:
    """Run the localhost web server until interrupted."""

    runtime = _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return

    token = web_token.strip() or generate_web_token()
    run_web_server(
        runtime=runtime,
        host=web_host,
        port=web_port,
        token=token,
        open_browser=not no_open_browser,
    )


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the Lisa localhost web host."""

    app(args=sys.argv[1:])


if __name__ == "__main__":
    main()
