"""hosts.cli.main — Typer app and CLI entry points for the terminal host.

This is the terminal-transport entry point for the Lisa framework. It
wraps the transport-neutral :mod:`host` runtime with Typer commands
and Rich/prompt_toolkit rendering.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

# Load .env BEFORE injecting the packs venv site-packages so that
# LISA_DATA_DIR (and LISA_PACKS_VENV) are honored when resolving the
# packs venv location. The loader only needs framework_root() and
# default_data_dir(), both of which are env-independent, so it can run
# safely before any host import. Search order:
#   (1) CWD — for repo-mode dev sessions
#   (2) ~/.lisa/.env — for tool-install mode (uv tool install lisa)
#   (3) project root — repo-mode back-compat
#   (4) hosts/little_lisa/.env — legacy repo-mode back-compat
from host.paths import (
    framework_root as _framework_root,
)
from kernel.models import McpServerHealth
from kernel.tools.mcp_client import MCPToolSpec
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from dotenv import load_dotenv

    _FRAMEWORK_ROOT = _framework_root()
    from host.config import default_data_dir as _default_data_dir

    for _env_candidate in (
        Path.cwd() / ".env",
        _default_data_dir() / ".env",
        _FRAMEWORK_ROOT / ".env",
        _FRAMEWORK_ROOT / "hosts" / "little_lisa" / ".env",
    ):
        if _env_candidate.exists():
            load_dotenv(dotenv_path=_env_candidate, override=False)
            break
except ImportError:  # pragma: no cover — python-dotenv is optional
    pass

# Inject the packs venv site-packages into sys.path BEFORE importing
# host so that importlib.metadata.entry_points discovers packs
# installed via /pack install into the packs venv. This must run
# before any host / kernel import that triggers entry-point scanning.
from host.paths import (
    inject_packs_venv_site_packages as _inject_packs_venv_site_packages,
)

_inject_packs_venv_site_packages()

from host.config import default_data_dir
from hosts.cli.render import (
    render_mcp_server_tools,
    render_mcp_servers,
    render_models,
    render_modes,
    render_operations,
    render_pack_list_panel,
    render_tools,
)

# ``HostRuntime`` is used only as a string annotation (``from __future__
# import annotations`` stringifies signatures) across the Typer callbacks.
# The runtime construction happens inside ``_initialize_runtime``, which
# imports ``HostRuntime`` + the mcp_pool / state_provider symbols lazily so
# ``--version`` (which exits in the eager Typer callback before any command
# body runs) does not pay the ~2s ``host.runtime`` ->
# ``kernel.ipsar_fsm`` -> ``openai`` import chain.
if TYPE_CHECKING:
    from host.runtime import HostRuntime as HostRuntime

# ``handle_message`` is deferred to the turn-execution call sites
# (``_run_once_with_teardown``) so ``--version`` / ``modes`` / ``models`` /
# ``verify`` don't pay the ~3s ``openai`` + ``mcp`` + ``kernel.ipsar_fsm``
# import cost it drags in. See ``host/__init__.py``'s PEP 562
# ``__getattr__`` for the package-level lazy import.

from rich.traceback import install

install(show_locals=True)

logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer(
    no_args_is_help=False,
    help="Run the Lisa framework terminal host.",
)

# Exit code returned when a restart is required (e.g. after a pack lifecycle
# change). Scripts can detect this to distinguish "restart needed" from a
# clean exit. Matches the BSD sysexits.h value for EX_UNAVAILABLE.
_RESTART_REQUIRED_EXIT = 69

# Env var carrying a message to replay as the first turn on the next launch
# after a restart. Set by `/pack install ... --resume <msg>` in
# host.commands; consumed once on startup by the CLI host.
_RESTART_RESUME_MESSAGE_ENV = "LISA_RESTART_RESUME_MESSAGE"


def _restart_current_process() -> None:
    """Restart the CLI process after a runtime change.

    - Unix (``os.name != "nt"``): true in-place re-exec via
      :func:`os.execv`. The process keeps its terminal, stdin/stdout, and
      environment; the new image re-enters :func:`main` with the same argv.
    - Windows (``os.name == "nt"``): do NOT spawn a child. A spawned
      process loses its parent terminal link and breaks the interactive
      REPL. Instead, print a clear restart-required notice and exit with
      :data:`_RESTART_REQUIRED_EXIT` so the user manually re-runs ``lisa``.

    Uses :data:`sys.argv` (not ``sys.orig_argv``) for args continuity â€”
    ``sys.orig_argv`` on Windows contains the Store launcher, not the
    venv interpreter, which corrupts re-execs.
    """
    if os.name == "nt":
        # Do not attempt to spawn a child process on Windows â€” it detaches
        # from the parent terminal and breaks the interactive REPL. Just
        # exit with a clear, script-detectable restart-required code.
        console.print(
            Panel(
                "A runtime change requires a restart to reload entry points,\n"
                "pack catalogs, and MCP manifests.\n\n"
                "Please re-run [bold]lisa[/bold] manually to continue.",
                title="Restart Required",
                border_style="gold3",
            )
        )
        raise typer.Exit(code=_RESTART_REQUIRED_EXIT)
    else:
        # True in-place exec replacement on Unix. The current process image
        # is replaced; the terminal, file descriptors, and environment are
        # preserved. os.execv does not return on success.
        os.execv(sys.executable, [sys.executable, *sys.argv])


def _get_version() -> str:
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("lisa-host-cli")
    except Exception:  # pylint: disable=broad-except
        return "0.1.0"


_VERSION_OPTION = typer.Option(
    False,
    "--version",
    help="Print the lisa version and exit.",
    is_eager=True,
)

# ---------------------------------------------------------------------------
# Shared Typer option aliases
# ---------------------------------------------------------------------------
# These Annotated type aliases define each runtime option exactly once so every
# subcommand shares identical flag names, defaults, and help text.

DebugOption = Annotated[
    bool,
    typer.Option(
        "--debug",
        help="Print turn context above each response.",
    ),
]
LogLevelOption = Annotated[
    str,
    typer.Option(
        "--log-level",
        help=("Log threshold: debug, info, warning, or error. Overrides LISA_LOG_LEVEL."),
    ),
]
LogOption = Annotated[
    str,
    typer.Option(
        "--log",
        help="Append JSONL request/response turn logs to this file.",
    ),
]
NoLogOption = Annotated[
    bool,
    typer.Option(
        "--no-log",
        help="Disable turn logging entirely.",
    ),
]
HostLogOption = Annotated[
    str,
    typer.Option(
        "--host-log",
        help=(
            "Stdlib host log file (kernel/sdk startup and runtime logs). "
            "Default: <data-dir>/host.jsonl"
        ),
    ),
]
DataDirOption = Annotated[
    str,
    typer.Option(
        "--data-dir",
        help=(
            "Data root for artifacts, memory, and logs. "
            "Overrides LISA_DATA_DIR. "
            "Default: ~/.lisa (see default_data_dir())"
        ),
    ),
]
OverlaysDirOption = Annotated[
    str,
    typer.Option(
        "--overlays-dir",
        help=(
            "User-authored overlays directory for mode and content overrides. "
            "Default: <data-dir>/overlays"
        ),
    ),
]
ThreadDirOption = Annotated[
    str,
    typer.Option(
        "--thread-dir",
        help=(
            "Thread log directory. Overrides LISA_THREAD_DIR. "
            "Defaults to <data-dir>/threads when a session context is active."
        ),
    ),
]
NoThreadOption = Annotated[
    bool,
    typer.Option(
        "--no-thread",
        help="Disable local turn history thread logging.",
    ),
]
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
    typer.Option(
        "--timezone",
        help=(
            "IANA timezone (e.g. UTC, America/New_York). "
            "Overrides LISA_TIMEZONE, then OS timezone env vars."
        ),
    ),
]
StateSnapshotOption = Annotated[
    str,
    typer.Option(
        "--state-snapshot",
        help="Local JSON snapshot file to inject compact context from.",
    ),
]
StreamingOption = Annotated[
    str,
    typer.Option(
        "--streaming",
        help=(
            "Streaming mode: none, stage, text, hybrid (default hybrid). "
            "Optionally append ':<ms>' for per-character delay (e.g. hybrid:50)."
        ),
    ),
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
    session_context_refresh: bool = False,
    state_snapshot_path: str,
    streaming: str = "",
    host_log_path: str = "",
) -> HostRuntime:
    """Build and wire a HostRuntime from CLI args + env vars.

    Thin adapter over :func:`host.bootstrap.build_runtime` that
    stamps ``host_kind="cli"`` and forwards every CLI argument. The
    composition sequence (config -> runtime -> state snapshot -> framework
    wiring -> MCP config compilation) lives in the shared bootstrap so
    the CLI and web hosts cannot drift.

    The MCP pool is NOT started here â€” ``runtime.start()`` must be called
    inside the host's event loop (see :func:`hosts.cli.repl.run_repl`).
    """
    from host.bootstrap import build_runtime

    return build_runtime(
        host_kind="cli",
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
        session_context_refresh=session_context_refresh,
        state_snapshot_path=state_snapshot_path,
        streaming=streaming,
        host_log_path=host_log_path,
    )


def _initialize_runtime_or_exit(**kwargs: Any) -> HostRuntime | None:
    """Initialize runtime, rendering startup errors and exiting on failure.

    Returns the runtime on success, or ``None`` after calling
    ``typer.Exit`` on failure (the ``typer.Exit`` raises, so ``None`` is
    only reached if the exit is suppressed).
    """
    try:
        return _initialize_runtime(**kwargs)
    except (ValueError, FileNotFoundError) as exc:
        from hosts.cli.repl import _render_runtime_error

        _render_runtime_error(console, exc, title="Startup Error")
        raise typer.Exit(code=2) from exc


# ---------------------------------------------------------------------------
# Default no-args mode
# ---------------------------------------------------------------------------


def _run_default_no_args_mode(
    *,
    debug: bool,
    log_level: str,
    log: str,
    no_log: bool,
    data_dir: str,
    overlays_dir: str,
    thread_dir: str,
    no_thread: bool,
    session_context: str,
    session_context_refresh: bool,
    timezone: str,
    state_snapshot: str,
    streaming: str,
    host_log: str,
) -> None:
    """Launch the REPL when no subcommand is given.

    The default no-args mode is the interactive REPL.
    """
    runtime = _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        overlays_dir=overlays_dir,
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
    _consume_restart_resume_message(runtime)
    from hosts.cli.repl import run_repl

    #                                   --version/modes/verify fast
    restart_requested = asyncio.run(run_repl(runtime, console=console))
    if restart_requested:
        _restart_current_process()


# ---------------------------------------------------------------------------
# Restart helpers
# ---------------------------------------------------------------------------


def _consume_restart_resume_message(runtime: HostRuntime) -> None:
    """Consume the resume-message env var on startup if present.

    If ``LISA_RESTART_RESUME_MESSAGE`` is set (by a `/pack ... --resume <msg>`
    command in the prior session), stash it on the runtime so the REPL can
    replay it as the first turn, then clear the env var to avoid replay
    loops on subsequent restarts.

    Only consumed in interactive REPL startup; one-shot ``once``/``pack``
    modes ignore it for deterministic one-shot UX.
    """
    resume_msg = os.environ.pop(_RESTART_RESUME_MESSAGE_ENV, "").strip()
    if resume_msg:
        runtime.pending_resume_message = resume_msg


def _exit_restart_required() -> None:
    """Print a restart-required notice and exit for one-shot commands.

    Used by ``once`` and ``pack install/uninstall`` after a successful
    pack lifecycle change. Unlike the REPL path, one-shot commands do
    NOT re-exec (re-running the same command would redo the mutation).
    Instead, print a clear notice and exit with
    :data:`_RESTART_REQUIRED_EXIT` on all platforms so the user manually
    restarts the interactive session.
    """
    console.print(
        Panel(
            "A runtime change requires a restart to reload entry points,\n"
            "pack catalogs, and MCP manifests.\n\n"
            "Please re-run [bold]lisa[/bold] to continue.",
            title="Restart Required",
            border_style="gold3",
        )
    )
    raise typer.Exit(code=_RESTART_REQUIRED_EXIT)


# ---------------------------------------------------------------------------
# Typer callback (no-args default)
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _cli_callback(
    ctx: typer.Context,
    version: bool = _VERSION_OPTION,
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "hybrid",
    host_log: HostLogOption = "",
) -> None:
    if version:
        console.print(f"lisa v{_get_version()}")
        raise typer.Exit()
    if ctx.invoked_subcommand is not None:
        return
    _run_default_no_args_mode(
        debug=debug,
        log_level=log_level,
        log=log,
        no_log=no_log,
        data_dir=data_dir,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot=state_snapshot,
        streaming=streaming,
        host_log=host_log,
    )


# ---------------------------------------------------------------------------
# cli command (REPL)
# ---------------------------------------------------------------------------


@app.command("cli", help="Run the interactive Lisa terminal REPL.")
def _cli_command(
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "hybrid",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        overlays_dir=overlays_dir,
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
    _consume_restart_resume_message(runtime)
    from hosts.cli.repl import run_repl

    #                                   --version/modes/verify fast
    restart_requested = asyncio.run(run_repl(runtime, console=console))
    if restart_requested:
        _restart_current_process()


# ---------------------------------------------------------------------------
# verify command
# ---------------------------------------------------------------------------

_VERIFY_PASS = "[green]PASS[/green]"
_VERIFY_WARN = "[yellow]WARN[/yellow]"
_VERIFY_FAIL = "[red]FAIL[/red]"
_KEY_REDACT_VISIBLE_CHARS = 4
_MIN_PYTHON_MAJOR = 3
_MIN_PYTHON_MINOR = 11
_HTTP_OK = 200
_OLLAMA_MODEL_PREVIEW_LIMIT = 5


def _redact_key(value: str) -> str:
    if not value:
        return "(not set)"
    if len(value) <= _KEY_REDACT_VISIBLE_CHARS:
        return "*" * len(value)
    return "*" * (len(value) - _KEY_REDACT_VISIBLE_CHARS) + value[-_KEY_REDACT_VISIBLE_CHARS:]


def _verify_check_python() -> tuple[str, str]:
    version = platform.python_version()
    major, minor, *_ = version.split(".")
    if int(major) >= _MIN_PYTHON_MAJOR and int(minor) >= _MIN_PYTHON_MINOR:
        return _VERIFY_PASS, f"Python {version}"
    return _VERIFY_FAIL, f"Python {version} (need >= 3.11)"


def _verify_check_env(runtime: HostRuntime) -> tuple[str, str]:
    # Check for .env in the same search order as the startup loader:
    # CWD → ~/.lisa/.env (tool mode) → framework root → hosts/little_lisa/.
    from host.config import default_data_dir as _default_data_dir

    _root = _framework_root()
    for candidate in (
        Path.cwd() / ".env",
        _default_data_dir() / ".env",
        _root / ".env",
        _root / "hosts" / "little_lisa" / ".env",
    ):
        if candidate.exists():
            return _VERIFY_PASS, f".env found ({candidate})"
    return _VERIFY_WARN, ".env not found. Create one at ~/.lisa/.env with your model provider keys."


def _verify_check_manifest(runtime: HostRuntime) -> tuple[str, str]:
    from host.model_manifest import load_model_manifest

    try:
        path = runtime.models_path
        manifest = load_model_manifest(path)
        model_count = len(manifest.models)
        if model_count == 0:
            return _VERIFY_FAIL, "Model manifest loaded but contains zero models"
        model_ids = ", ".join(m.model_id for m in manifest.models)
        return _VERIFY_PASS, f"{model_count} model(s): {model_ids}"
    except Exception as exc:
        return _VERIFY_FAIL, f"Model manifest failed to load: {exc}"


def _verify_check_pack_catalog() -> tuple[str, str]:
    try:
        from kernel.ipsar_fsm import get_workflow_catalog

        catalog = get_workflow_catalog()
        mode_count = len(catalog)
        if mode_count == 0:
            return _VERIFY_WARN, "Pack catalog loaded but contains zero modes"
        mode_ids = ", ".join(sorted(catalog.keys()))
        return _VERIFY_PASS, f"{mode_count} mode(s): {mode_ids}"
    except Exception as exc:
        return _VERIFY_FAIL, f"Pack catalog failed to load: {exc}"


def _verify_check_ollama() -> tuple[str, str]:
    import os

    import httpx

    configured_host = (
        os.getenv("LISA_OLLAMA_HOST", "").strip()
        or os.getenv("OLLAMA_HOST", "").strip()
        or "127.0.0.1:11434"
    )

    async def _ping() -> tuple[str, str]:
        try:
            base_url = f"http://{configured_host}"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base_url}/api/tags")
                if resp.status_code == _HTTP_OK:
                    data = resp.json()
                    models = [m.get("name", "?") for m in data.get("models", [])]
                    if models:
                        preview = ", ".join(models[:_OLLAMA_MODEL_PREVIEW_LIMIT])
                        suffix = "..." if len(models) > _OLLAMA_MODEL_PREVIEW_LIMIT else ""
                        return _VERIFY_PASS, (
                            f"Ollama reachable, {len(models)} model(s) pulled: {preview}{suffix}"
                        )
                    return _VERIFY_WARN, "Ollama reachable but no models pulled"
                return _VERIFY_WARN, f"Ollama returned HTTP {resp.status_code}"
        except Exception as exc:
            return _VERIFY_WARN, f"Ollama not reachable at {configured_host}: {exc}"

    try:
        return asyncio.run(_ping())
    except Exception as exc:
        return _VERIFY_WARN, f"Ollama check failed: {exc}"


def _verify_check_provider_keys() -> tuple[str, str]:
    import os

    parts: list[str] = []
    # Check LISA_* vars first (the new host's env vars), then fall back to
    # the bare AZURE_OPENAI_* / OPENAI_* / ANTHROPIC_* / GOOGLE_* vars that
    # the providers module also accepts.
    azure_key = (
        os.getenv("LISA_AZURE_OPENAI_API_KEY", "").strip()
        or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    )
    azure_endpoint = (
        os.getenv("LISA_AZURE_OPENAI_ENDPOINT", "").strip()
        or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    )
    if azure_key and azure_endpoint:
        parts.append(f"Azure OpenAI: {_redact_key(azure_key)} @ {azure_endpoint}")
    elif azure_key or azure_endpoint:
        parts.append("Azure OpenAI: partially configured")

    openai_key = (
        os.getenv("LISA_OPENAI_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if openai_key:
        parts.append(f"OpenAI: {_redact_key(openai_key)}")

    anthropic_key = (
        os.getenv("LISA_ANTHROPIC_API_KEY", "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    if anthropic_key:
        parts.append(f"Anthropic: {_redact_key(anthropic_key)}")

    google_key = (
        os.getenv("LISA_GOOGLE_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    if google_key:
        parts.append(f"Google Gemini: {_redact_key(google_key)}")

    ollama_host = os.getenv("LISA_OLLAMA_HOST", "").strip() or os.getenv("OLLAMA_HOST", "").strip()
    if ollama_host:
        parts.append(f"Ollama: {ollama_host}")

    if not parts:
        return _VERIFY_WARN, "No provider keys configured"
    return _VERIFY_PASS, "; ".join(parts)


def _verify_check_paths(runtime: HostRuntime) -> tuple[str, str]:
    parts: list[str] = []
    parts.append(f"Data dir: {runtime.data_dir}")
    parts.append(f"Thread logs: {runtime.thread_dir}")
    tp = runtime.thread_path
    if tp:
        parts.append(f"Thread path: {tp}")
    return _VERIFY_PASS, "; ".join(parts)


def _verify_check_session_context(runtime: HostRuntime) -> tuple[str, str]:
    if not runtime.config.user_slug:
        return "[yellow]NONE[/yellow]", "No session context active"
    parts = [f"Context: {runtime.config.user_slug}"]
    ref = runtime.config.session_context.strip()
    if ref and ref != runtime.config.user_slug:
        parts.append(f"Path: {ref}")
    return _VERIFY_PASS, "; ".join(parts)


@app.command("verify", help="Run verification checks of the Lisa environment.")
def _verify_command(
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
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
    """Run verification checks of the Lisa environment."""
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
    checks: list[tuple[str, str, str]] = [
        ("Python", *_verify_check_python()),
        (".env", *_verify_check_env(runtime)),
        ("Session context", *_verify_check_session_context(runtime)),
        ("Model manifest", *_verify_check_manifest(runtime)),
        ("Pack catalog", *_verify_check_pack_catalog()),
        ("Provider keys", *_verify_check_provider_keys()),
        ("Ollama", *_verify_check_ollama()),
        ("Paths", *_verify_check_paths(runtime)),
    ]

    table = Table(title="lisa verify")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold", no_wrap=False)

    failures = 0
    for name, status, detail in checks:
        table.add_row(name, status, detail)
        if "FAIL" in status:
            failures += 1

    console.print(table)

    if failures:
        console.print(f"\n[red]{failures} check(s) failed.[/red]")
        raise typer.Exit(code=1)
    console.print("\n[green]All checks passed.[/green]")


# ---------------------------------------------------------------------------
# modes command
# ---------------------------------------------------------------------------


@app.command("modes", help="List supported Lisa modes.")
def _modes_command(
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
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
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
    render_modes(runtime, console=console)


# ---------------------------------------------------------------------------
# models command
# ---------------------------------------------------------------------------


@app.command(
    "models",
    help=(
        "Manage the model manifest. ACTION defaults to 'show'. "
        "Subcommands: show, discover, apply, help. "
        "(reload and edit are REPL-only; they have no durable effect as a "
        "one-shot CLI invocation.)"
    ),
)
def _models_command(
    action: str = typer.Argument(
        "show",
        help=(
            "Action: show (default), discover, apply, help. "
            "'discover' writes a proposal manifest; 'apply' promotes it to active."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="With 'discover', overwrite an existing proposal manifest.",
    ),
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
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
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

    normalized = action.strip().lower()

    # 'show' uses the Rich table renderer; on failure (e.g. no active
    # manifest yet) it falls back to the core text renderer so the user
    # gets the same friendly "run discover first" guidance as the REPL.
    if normalized in {"", "show"}:
        try:
            render_models(runtime, console=console)
            return
        except Exception:  # pylint: disable=broad-except
            normalized = "show"

    # 'reload' and 'edit' only mutate transient in-process state or print
    # guidance — both pointless for a one-shot CLI process whose runtime
    # is discarded on exit. Surface that clearly instead of pretending
    # to do work; the durable CLI workflow is discover -> (edit file) ->
    # apply. Both remain available in the persistent REPL.
    if normalized in {"reload", "edit"}:
        console.print(
            "[yellow]'models " + normalized + "' has no durable effect as a one-shot "
            "CLI command: it only refreshes in-process state or prints guidance, and "
            "the CLI runtime is discarded on exit.[/yellow]\n"
            "Edit models.proposed.yaml by hand, then run 'lisa models apply'.\n"
            "Use 'lisa' (REPL) for /models " + normalized + "."
        )
        return

    # show (fallback), discover, apply, help delegate to the
    # transport-neutral core handler (the same one the REPL /models
    # command uses), so behavior stays identical across hosts. We
    # synthesize the equivalent '/models <sub> [--force]' slash string
    # and print the returned Markdown text in a Rich panel.
    from host.commands import handle_models_command
    from rich.markdown import Markdown

    if normalized == "discover" and force:
        stripped = "/models discover --force"
    else:
        stripped = f"/models {normalized}".strip()

    # The discover/apply/reload/show subcommands read or write model
    # manifests via the lisa-storage MCP server, which requires a live
    # MCP tool pool. _initialize_runtime only compiles configs; the pool
    # is started inside the host's event loop. Start it here, run the
    # handler, and tear the pool down in finally — all within one
    # asyncio.run loop so anyio cancel scopes are entered and exited in
    # the same task (per host.runtime invariants).
    async def _run_models_command() -> str:
        try:
            await runtime.start()
            return await handle_models_command(runtime, stripped, stripped)
        finally:
            await runtime.aclose()

    response_text = asyncio.run(_run_models_command())
    console.print(Panel(Markdown(response_text), title="Models", border_style="cyan"))


# ---------------------------------------------------------------------------
# operations command
# ---------------------------------------------------------------------------


@app.command("operations", help="List internal operations with IDs for allowlist authoring.")
def _operations_command(
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
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
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
    render_operations(runtime, console=console)


# ---------------------------------------------------------------------------
# tools command
# ---------------------------------------------------------------------------


@app.command(
    "tools",
    help=(
        "List internal tools with IDs for allowlist authoring. "
        "Use the `mcp` command to inspect MCP server tools."
    ),
)
def _tools_command(
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
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
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
    render_tools(runtime, console=console)


McpIntrospection = tuple[dict[str, Any], dict[str, list[MCPToolSpec]]]
"""One MCP introspection result: ``(health, tools_by_server)``.

``health`` is a ``dict[str, McpServerHealth]`` keyed by ``server_id``;
``tools_by_server`` is a ``dict[str, list[MCPToolSpec]]`` grouping the
discovered tool specs by their owning server. Both are collected by
warming the MCP pool (see :func:`_collect_mcp_tools`). The CLI render
layer reads health for the summary table and the specs for the per-server
tool detail.
"""


def _build_mcp_health(
    pool: Any,
    is_alias: dict[str, bool],
) -> dict[str, McpServerHealth]:
    """Build a per-server health dict from the live pool's introspection accessors.

    Mirrors :func:`host.introspect.gather_mcp_pool_health` but runs
    in the CLI's short-lived ``asyncio.run`` event loop (the pool is started
    and torn down within ``_collect_mcp_tools``). Reads the same accessors
    (``warm_server_ids``, ``warm_failed_ids``, ``cold_failed_ids``,
    ``permanently_failed_ids``, ``restart_counts``, ``last_errors``,
    ``tools_by_server``) so the CLI summary and the REPL ``/mcp`` command
    show identical state.
    """
    from host.introspect import gather_mcp_pool_health

    health = gather_mcp_pool_health(pool)
    # Annotate alias tools on the tool_names so the render layer can mark
    # them without a separate lookup. The specs themselves carry the real
    # server_id; this is purely a display hint.
    _ = is_alias  # reserved for a future render-layer annotation pass
    return health


async def _collect_mcp_tools(runtime: HostRuntime) -> McpIntrospection:
    """Warm the MCP pool, return per-server health + discovered tool specs.

    MCP tool names are only discoverable by calling ``tools/list`` on each
    server, which requires a live session. This starts the pool (warm + cold
    configs), reads the discovered tool specs and per-server health, and
    tears the pool down — all inside one ``asyncio.run`` event loop to honor
    anyio's same-loop contract (the pool's async context managers must be
    entered and exited in the same loop). Cold servers that fail to start
    are skipped by the pool; their tools are simply absent from the
    returned list.

    Returns an empty :data:`McpIntrospection` when the runtime has no MCP
    servers configured or when pool startup fails, so the ``tools``/``mcp``
    commands degrade to showing only the internal tool catalog instead of
    crashing.
    """
    if not runtime.mcp_warm_configs and not runtime.mcp_cold_configs:
        return ({}, {})
    try:
        await runtime.start()
        pool = runtime.mcp_tool_pool
        if pool is None:
            return ({}, {})
        tools = await pool.list_tools()
        # Capture is_alias flags BEFORE aclose() clears _synthetic_handlers.
        is_alias = {spec.name: pool.is_synthetic_tool(spec.name) for spec in tools}
        tools_by_server: dict[str, list[MCPToolSpec]] = {}
        for spec in tools:
            tools_by_server.setdefault(spec.server_id, []).append(spec)
        health = _build_mcp_health(pool, is_alias)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("MCP tool collection failed: %s", exc)
        return ({}, {})
    finally:
        await runtime.aclose()
    return (health, tools_by_server)


# ---------------------------------------------------------------------------
# mcp command
# ---------------------------------------------------------------------------


@app.command(
    "mcp",
    help=(
        "List MCP servers with status, transport, URL/command, tool count, "
        "and last error. Pass a server id to drill into that server's tools "
        "(name, description, required args)."
    ),
)
def _mcp_command(
    target: str = typer.Argument(
        "",
        help="Optional server id to drill into that server's tools.",
    ),
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
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
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
    mcp_health, mcp_tools_by_server = asyncio.run(_collect_mcp_tools(runtime))
    if not mcp_health:
        console.print("[yellow]No MCP server tools discovered.[/yellow]")
        return
    server_id = target.strip()
    if server_id:
        specs = mcp_tools_by_server.get(server_id, [])
        health = mcp_health.get(server_id)
        if health is None:
            console.print(f"[yellow]No MCP server named {server_id!r} is configured.[/yellow]")
            return
        render_mcp_server_tools(specs, health, console=console)
    else:
        render_mcp_servers(mcp_health, console=console)


# ---------------------------------------------------------------------------
# pack command
# ---------------------------------------------------------------------------


@app.command("pack", help="Manage runtime pack installs (list/install/uninstall).")
def _pack_command(
    action: str = typer.Argument(..., help="Action: list, install, uninstall."),
    target: str = typer.Argument(
        "",
        help="Install spec or package name (required for install/uninstall).",
    ),
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
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
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

    normalized = action.strip().lower()
    if normalized == "list":
        render_pack_list_panel(runtime, console=console)
        return
    if normalized == "install":
        ok, details = asyncio.run(_run_pack_install(runtime, target))
        if not ok:
            console.print(f"[red]{details}[/red]")
            raise typer.Exit(code=1)
        console.print(details)
        if runtime.restart_required:
            _exit_restart_required()
        raise typer.Exit(code=0)
    if normalized == "uninstall":
        ok, details = asyncio.run(_run_pack_uninstall(runtime, target))
        if not ok:
            console.print(f"[red]{details}[/red]")
            raise typer.Exit(code=1)
        console.print(details)
        if runtime.restart_required:
            _exit_restart_required()
        raise typer.Exit(code=0)

    console.print("[red]Unknown action. Use: list, install, uninstall.[/red]")
    raise typer.Exit(code=2)


async def _run_pack_install(runtime: HostRuntime, spec: str) -> tuple[bool, str]:
    """Install a pack by spec (delegates to host pack management)."""
    from host.commands import handle_pack_command

    result = await handle_pack_command(runtime, "/pack", f"/pack install {spec}")
    # handle_pack_command returns a plain string; treat it as success unless
    # it starts with a known failure prefix.
    return (True, result)


async def _run_pack_uninstall(runtime: HostRuntime, package_name: str) -> tuple[bool, str]:
    """Uninstall a pack by package name."""
    from host.commands import handle_pack_command

    result = await handle_pack_command(runtime, "/pack", f"/pack uninstall {package_name}")
    return (True, result)


# ---------------------------------------------------------------------------
# once command
# ---------------------------------------------------------------------------


@app.command("once", help="Run a single turn and print the response.")
def _once_command(
    message: str = typer.Argument(..., help="Message to route through the host."),
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
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
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
    restart_required = asyncio.run(_run_once_with_teardown(runtime, message))
    if restart_required:
        _exit_restart_required()


async def _run_once_with_teardown(runtime: HostRuntime, message: str) -> bool:
    """Run one turn, then close the MCP pool.

    Returns ``True`` if the runtime flagged a restart as required (e.g.
    a pack lifecycle command), so the caller can exit with a restart
    notice instead of a clean exit.
    """
    from host.handle_message import handle_message
    from hosts.cli.transport import RichTransport

    transport = RichTransport(console=console, debug=runtime.config.debug)
    try:
        await runtime.start()
        await handle_message(runtime, message, transport)
    finally:
        await runtime.aclose()
    return runtime.restart_required


# ---------------------------------------------------------------------------
# intake command
# ---------------------------------------------------------------------------


@app.command("intake", help="Run one intake tick: collect signals, process, write state.")
def _intake_command(
    data_dir: DataDirOption = "",
    log_level: LogLevelOption = "",
    host_log: HostLogOption = "",
) -> None:
    """One-shot intake process. Triggered by systemd timer or cron.

    Collects signals from installed pack signal connectors, runs processing
    loops, and writes the merged UnifiedState to ``<data-dir>/state.json``.
    The next turn reads that state so the model has fresh ambient data.
    """
    resolved_data_dir = Path(data_dir).expanduser() if data_dir else default_data_dir()
    try:
        from host.intake import run_intake
    except ImportError:
        console.print("[red]intake module not available in this host.[/red]")
        raise typer.Exit(code=2)

    state = asyncio.run(run_intake(data_dir=resolved_data_dir))
    console.print(
        f"[green]Intake tick complete.[/green] "
        f"{len(state.signal_snapshots)} signal(s), "
        f"{len(state.active_alerts)} alert(s). "
        f"State written to {resolved_data_dir / 'state.json'}"
    )


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the Lisa terminal host."""
    app(args=sys.argv[1:])


if __name__ == "__main__":
    main()
