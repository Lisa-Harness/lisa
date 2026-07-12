"""hosts.cli.main — Typer app and CLI entry points for the terminal host.

This is the terminal-transport entry point for the Lisa framework. It
wraps the transport-neutral :mod:`hosts.core` runtime with Typer commands
and Rich/prompt_toolkit rendering. It replaces the CLI-specific code in
``hosts/little_lisa/main.py``.

Key differences from Little Lisa:
- Builds a :class:`hosts.core.runtime.HostRuntime` (not a ``_RUNTIME_STATE`` dict).
- Delegates turn execution to :func:`hosts.core.handle_message`.
- The Typer app is named "lisa" (not "little-lisa"); entry point is ``lisa``.
- The default no-args mode launches the REPL (tray-first desktop is a
  future ``hosts/web`` concern).
"""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from openai import AsyncAzureOpenAI, AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load .env from the project root before importing hosts.core so that
# HostConfig.from_env sees the provider credentials and config values.
# The .env file is searched at: (1) CWD, (2) project root, (3) hosts/little_lisa/
# for backward compatibility with existing .env files.
try:
    from dotenv import load_dotenv

    _FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
    for _env_candidate in (
        Path.cwd() / ".env",
        _FRAMEWORK_ROOT / ".env",
        _FRAMEWORK_ROOT / "hosts" / "little_lisa" / ".env",
    ):
        if _env_candidate.exists():
            load_dotenv(dotenv_path=_env_candidate, override=False)
            break
except ImportError:  # pragma: no cover — python-dotenv is optional
    pass

from hosts.cli.repl import run_repl
from hosts.cli.render import render_models, render_modes, render_pack_list_panel
from hosts.core.config import HostConfig
from hosts.core.handle_message import handle_message
from hosts.core.mcp_pool import compile_mcp_configs, configure_runtime, setup_framework
from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Run the Lisa framework terminal host.",
)


def _get_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version as _pkg_version

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
            "Default: <data-dir>/host.log"
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
            "Default: <cwd>/artifacts/"
        ),
    ),
]
MemoryDbOption = Annotated[
    str,
    typer.Option(
        "--memory-db",
        help=(
            "Memory database JSON file path. "
            "Overrides LISA_MEMORY_DB. "
            "Default: <data-dir>/memory.json"
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
            "User context source: a slug name (content/user_contexts/<name>.md), "
            "a full .md file path, or a directory (all files concatenated). "
            "Omit to skip session context."
        ),
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
            "Streaming mode: none, stage, text, hybrid. "
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
    memory_db: str = "",
    thread_dir: str = "",
    no_thread: bool = False,
    user_slug: str,
    timezone: str = "",
    state_snapshot_path: str,
    streaming: str = "",
    host_log_path: str = "",
) -> HostRuntime:
    """Build and wire a HostRuntime from CLI args + env vars.

    Order matters: the config is built first (resolving paths, timezone,
    log level), then the runtime is constructed, then the framework is
    wired (content/operation/catalog MCP servers), provider clients are
    reset, and compiled MCP configs are stashed on the runtime.

    The MCP pool is NOT started here — ``runtime.start()`` must be called
    inside the host's event loop (see :func:`hosts.cli.repl.run_repl`).
    """
    _ = no_log  # honored via HostConfig.log_path resolution below
    config = HostConfig.from_env(
        debug=debug,
        log_level=log_level,
        log_path=log_path,
        host_log_path=host_log_path,
        data_dir=data_dir,
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        timezone=timezone,
        session_context=user_slug,
        user_slug=user_slug,
        state_snapshot=state_snapshot_path,
        streaming=streaming,
    )

    # Configure stdlib logging to the host-log path at the resolved level.
    _configure_root_logging(config)

    runtime = HostRuntime(config)
    setup_framework(runtime)
    configure_runtime(runtime)
    warm, cold = compile_mcp_configs(runtime)
    runtime.set_mcp_configs(warm, cold)
    return runtime


def _configure_root_logging(config: HostConfig) -> None:
    """Configure stdlib root logging to the host-log file at the resolved level."""
    import logging as _logging

    level_name = config.effective_log_level.upper()
    level = getattr(_logging, level_name, _logging.INFO)
    root = _logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicate output on re-init.
    for handler in list(root.handlers):
        root.removeHandler(handler)
    host_log = config.resolved_host_log_path
    host_log.parent.mkdir(parents=True, exist_ok=True)
    handler = _logging.FileHandler(host_log, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(
        _logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    root.addHandler(handler)


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
    memory_db: str,
    thread_dir: str,
    no_thread: bool,
    session_context: str,
    timezone: str,
    state_snapshot: str,
    streaming: str,
    host_log: str,
) -> None:
    """Launch the REPL when no subcommand is given.

    For now, the default no-args mode is the interactive REPL. A
    tray-first desktop launch is a future ``hosts/web`` concern.
    """
    runtime = _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    asyncio.run(run_repl(runtime, console=console))


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
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
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
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
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
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
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
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    asyncio.run(run_repl(runtime, console=console))


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
    # Check for .env in the project root (where the CLI loads it from).
    _root = Path(__file__).resolve().parents[2]
    for candidate in (Path.cwd() / ".env", _root / ".env", _root / "hosts" / "little_lisa" / ".env"):
        if candidate.exists():
            return _VERIFY_PASS, f".env found ({candidate})"
    return _VERIFY_WARN, ".env not found. Run `lisa setup` or create one in the project root."


def _verify_check_manifest(runtime: HostRuntime) -> tuple[str, str]:
    from hosts.core.model_manifest import load_model_manifest
    from hosts.core.model_runtime import manifest_path_from_env

    try:
        path = runtime.config.models_file or manifest_path_from_env()
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

    configured_host = os.getenv("LISA_OLLAMA_HOST", "").strip() or os.getenv(
        "OLLAMA_HOST", ""
    ).strip() or "127.0.0.1:11434"

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
    azure_key = os.getenv("LISA_AZURE_OPENAI_API_KEY", "").strip() or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    azure_endpoint = os.getenv("LISA_AZURE_OPENAI_ENDPOINT", "").strip() or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    if azure_key and azure_endpoint:
        parts.append(f"Azure OpenAI: {_redact_key(azure_key)} @ {azure_endpoint}")
    elif azure_key or azure_endpoint:
        parts.append("Azure OpenAI: partially configured")

    openai_key = os.getenv("LISA_OPENAI_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        parts.append(f"OpenAI: {_redact_key(openai_key)}")

    anthropic_key = os.getenv("LISA_ANTHROPIC_API_KEY", "").strip() or os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        parts.append(f"Anthropic: {_redact_key(anthropic_key)}")

    google_key = os.getenv("LISA_GOOGLE_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
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
    parts.append(f"Memory DB: {runtime.config.resolved_memory_db_path}")
    parts.append(f"Thread logs: {runtime.thread_dir}")
    tp = runtime.thread_path
    if tp:
        parts.append(f"Thread path: {tp}")
    return _VERIFY_PASS, "; ".join(parts)


def _verify_check_session_context(runtime: HostRuntime) -> tuple[str, str]:
    if not runtime.config.user_slug:
        return "[yellow]NONE[/yellow]", "No session context active"
    return _VERIFY_PASS, f"Context: {runtime.config.user_slug}"


@app.command("verify", help="Run verification checks of the Lisa environment.")
def _verify_command(
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
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
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
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
    table.add_column("Detail")

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
# setup wizard command
# ---------------------------------------------------------------------------

_SETUP_PROVIDERS = {
    "1": ("ollama", "Ollama (local, no API key needed)"),
    "2": ("openai", "OpenAI (requires API key)"),
    "3": ("azure_openai", "Azure OpenAI (requires endpoint + API key)"),
    "4": ("anthropic", "Anthropic (requires API key)"),
    "5": ("google", "Google Gemini (requires API key)"),
}


def _setup_write_env(env_path: Path, updates: dict[str, str]) -> None:
    existing: dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            existing[key.strip()] = value.strip()
    existing.update(updates)
    lines = [
        "# Lisa environment configuration",
        "# Generated by: lisa setup",
        "",
    ]
    for key, value in sorted(existing.items()):
        lines.append(f"{key}={value}")
    lines.append("")
    env_path.write_text("\n".join(lines), encoding="utf-8")


async def _setup_validate_ollama() -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags")
            if resp.status_code == _HTTP_OK:
                return True, "Ollama is reachable"
            return False, f"Ollama returned HTTP {resp.status_code}"
    except Exception as exc:
        return False, f"Cannot reach Ollama: {exc}"


async def _setup_validate_openai(api_key: str) -> tuple[bool, str]:
    try:
        client = AsyncOpenAI(api_key=api_key)
        await client.models.list()
        return True, "OpenAI API key is valid"
    except Exception as exc:
        return False, f"OpenAI validation failed: {exc}"


async def _setup_validate_google(api_key: str) -> tuple[bool, str]:
    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        await client.models.list()
        return True, "Google Gemini API key is valid"
    except Exception as exc:
        return False, f"Google Gemini validation failed: {exc}"


async def _setup_validate_azure(endpoint: str, api_key: str) -> tuple[bool, str]:
    try:
        client = AsyncAzureOpenAI(
            api_version="2024-10-21",
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        await client.models.list()
        return True, "Azure OpenAI endpoint is reachable"
    except Exception as exc:
        return False, f"Azure OpenAI validation failed: {exc}"


async def _setup_validate_anthropic(api_key: str) -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            if resp.status_code == _HTTP_OK:
                return True, "Anthropic API key is valid"
            return False, f"Anthropic returned HTTP {resp.status_code}"
    except Exception as exc:
        return False, f"Anthropic validation failed: {exc}"


def _setup_step_ollama() -> dict[str, str]:
    console.print("\n[bold]Ollama[/bold] no API key required.")
    console.print("Make sure Ollama is installed and running.")
    console.print("Install: https://ollama.com")
    console.print("\nPull the default models:")
    console.print("  ollama pull llama3.2:latest")
    console.print("  ollama pull qwen2.5:7b")
    console.print()
    ok, msg = asyncio.run(_setup_validate_ollama())
    if ok:
        console.print(f"[green]{msg}[/green]")
    else:
        console.print(f"[yellow]{msg}[/yellow]")
        console.print("You can still proceed just start Ollama before running Lisa.")
    return {}


def _setup_step_openai() -> dict[str, str]:
    console.print("\n[bold]OpenAI[/bold]")
    api_key = console.input("Enter your OpenAI API key: ").strip()
    if not api_key:
        console.print("[red]API key is required.[/red]")
        raise typer.Exit(code=1)
    console.print("Validating...")
    ok, msg = asyncio.run(_setup_validate_openai(api_key))
    if ok:
        console.print(f"[green]{msg}[/green]")
    else:
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(code=1)
    return {"OPENAI_API_KEY": api_key}


def _setup_step_azure() -> dict[str, str]:
    console.print("\n[bold]Azure OpenAI[/bold]")
    endpoint = console.input("Enter your Azure OpenAI endpoint URL: ").strip()
    if not endpoint:
        console.print("[red]Endpoint URL is required.[/red]")
        raise typer.Exit(code=1)
    api_key = console.input("Enter your Azure OpenAI API key: ").strip()
    if not api_key:
        console.print("[red]API key is required.[/red]")
        raise typer.Exit(code=1)
    console.print("Validating...")
    ok, msg = asyncio.run(_setup_validate_azure(endpoint, api_key))
    if ok:
        console.print(f"[green]{msg}[/green]")
    else:
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(code=1)
    return {
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_API_KEY": api_key,
    }


def _setup_step_anthropic() -> dict[str, str]:
    console.print("\n[bold]Anthropic[/bold]")
    api_key = console.input("Enter your Anthropic API key: ").strip()
    if not api_key:
        console.print("[red]API key is required.[/red]")
        raise typer.Exit(code=1)
    console.print("Validating...")
    ok, msg = asyncio.run(_setup_validate_anthropic(api_key))
    if ok:
        console.print(f"[green]{msg}[/green]")
    else:
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(code=1)
    return {"ANTHROPIC_API_KEY": api_key}


def _setup_step_google() -> dict[str, str]:
    console.print("\n[bold]Google Gemini[/bold]")
    api_key = console.input("Enter your Google API key: ").strip()
    if not api_key:
        console.print("[red]API key is required.[/red]")
        raise typer.Exit(code=1)
    console.print("Validating...")
    ok, msg = asyncio.run(_setup_validate_google(api_key))
    if ok:
        console.print(f"[green]{msg}[/green]")
    else:
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(code=1)
    return {"GOOGLE_API_KEY": api_key}


_SETUP_PROVIDER_STEPS: dict[str, Callable[[], dict[str, str]]] = {
    "ollama": _setup_step_ollama,
    "openai": _setup_step_openai,
    "azure_openai": _setup_step_azure,
    "anthropic": _setup_step_anthropic,
    "google": _setup_step_google,
}


@app.command("setup", help="Interactive wizard to configure Lisa providers.")
def _setup_command() -> None:
    """Interactive wizard to configure Lisa providers."""
    env_path = Path.cwd() / ".env"

    console.print(
        Panel.fit(
            "Lisa Setup Wizard\n\n"
            "This wizard will help you configure a model provider.\n"
            "Ollama is the easiest path no API key needed, fully local.",
            title="Setup",
            border_style="steel_blue3",
        )
    )

    console.print("\nChoose a provider:")
    for key, (_, label) in _SETUP_PROVIDERS.items():
        console.print(f"  [{key}] {label}")

    choice = console.input("\nEnter choice [1-5] (default: 1): ").strip() or "1"
    if choice not in _SETUP_PROVIDERS:
        console.print(f"[red]Invalid choice: {choice}[/red]")
        raise typer.Exit(code=1)

    provider, _ = _SETUP_PROVIDERS[choice]
    updates = _SETUP_PROVIDER_STEPS[provider]()

    if updates:
        _setup_write_env(env_path, updates)
        console.print(f"\n[green]Configuration written to {env_path}[/green]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Run [bold]lisa verify[/bold] to verify everything is working")
    console.print('  2. Run [bold]lisa once "/markdown Hello, Lisa!"[/bold] for your first turn')
    console.print("  3. Run [bold]lisa cli[/bold] for the interactive REPL")
    console.print("  4. Run [bold]lisa[/bold] for the default REPL launch")


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
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
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
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
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


@app.command("models", help="List configured models with designations and aliases.")
def _models_command(
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
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
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    render_models(runtime, console=console)


# ---------------------------------------------------------------------------
# packs command
# ---------------------------------------------------------------------------


@app.command("packs", help="Manage runtime pack installs (list/install/uninstall).")
def _packs_command(
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
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
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
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
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
        raise typer.Exit(code=0)
    if normalized == "uninstall":
        ok, details = asyncio.run(_run_pack_uninstall(runtime, target))
        if not ok:
            console.print(f"[red]{details}[/red]")
            raise typer.Exit(code=1)
        console.print(details)
        raise typer.Exit(code=0)

    console.print("[red]Unknown action. Use: list, install, uninstall.[/red]")
    raise typer.Exit(code=2)


async def _run_pack_install(runtime: HostRuntime, spec: str) -> tuple[bool, str]:
    """Install a pack by spec (delegates to hosts.core pack management)."""
    from hosts.core.commands import handle_pack_command

    result = await handle_pack_command(runtime, "/pack", f"/pack install {spec}")
    # handle_pack_command returns a plain string; treat it as success unless
    # it starts with a known failure prefix.
    return (True, result)


async def _run_pack_uninstall(runtime: HostRuntime, package_name: str) -> tuple[bool, str]:
    """Uninstall a pack by package name."""
    from hosts.core.commands import handle_pack_command

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
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
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
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    asyncio.run(_run_once_with_teardown(runtime, message))


async def _run_once_with_teardown(runtime: HostRuntime, message: str) -> None:
    """Run one turn, then close the MCP pool."""
    from hosts.cli.transport import RichTransport

    transport = RichTransport(console=console, debug=runtime.config.debug)
    try:
        await runtime.start()
        await handle_message(runtime, message, transport)
    finally:
        await runtime.aclose()


# ---------------------------------------------------------------------------
# bottom command
# ---------------------------------------------------------------------------


@app.command("bottom", help="Run one bottom-half tick: collect signals, process, write state.")
def _bottom_command(
    data_dir: DataDirOption = "",
    log_level: LogLevelOption = "",
    host_log: HostLogOption = "",
) -> None:
    """One-shot bottom-half process. Triggered by systemd timer or cron."""
    resolved_data_dir = Path(data_dir).expanduser() if data_dir else Path.cwd() / "artifacts"
    try:
        from hosts.core.bottom import run_tick  # noqa: PLC0415
    except ImportError:
        console.print("[red]bottom-half module not available in this host.[/red]")
        raise typer.Exit(code=2)

    state = asyncio.run(run_tick(data_dir=resolved_data_dir))
    console.print(
        f"[green]Bottom-half tick complete.[/green] "
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