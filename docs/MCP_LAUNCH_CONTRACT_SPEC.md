# MCP Launch Contract & Tier 1 Server Implementation Spec

Status: Proposed 2026-07-06
Owner: Architecture

This document specifies the controlled launch contract for all MCP servers
(first-party and third-party), the standard CLI argument surface, the
environment-variable philosophy, the runtime-agnostic subprocess launcher, and
the implementation plan for the three Tier 1 MCP servers (`lisa-guard`,
`lisa-storage`, `lisa-content`).

It is self-contained: a new implementation thread can execute this spec without
prior conversation context.

## Design Verdicts

| Question | Answer |
| --- | --- |
| How do servers discover their data directory? | **CLI args only.** The host passes `--data-dir` explicitly. Servers do not hardcode fallback paths. |
| What are env vars for? | **Behavioral config only** — feature flags, API keys, backend selection, log level defaults. Not path resolution. |
| Can servers read `LISA_DATA_DIR` from env? | **No.** Path-like arguments come exclusively from CLI args. Env is supplemental for behaviors. |
| Does the host have a default data-dir? | **Yes, the host may default** (e.g. Little Lisa defaults to `examples/little_lisa/artifacts/`). But the host must pass it as a CLI arg to every server. The server never assumes a default. |
| How are third-party MCP servers controlled? | **Approval registry.** Hosts register approved server specs with explicit env/cwd/runtime/tool policy. Unregistered specs are rejected. |
| How do we handle multiple runtimes (uv, node, npx, binary)? | **Runtime adapter layer.** Abstract runtime specs compile to concrete `MCPClientConfig`. The host declares the runtime kind; the adapter builds the command/args. |
| Do existing servers need migration? | **Yes.** All first-party servers adopt the shared argparse contract. Legacy env-var path resolution is removed (hard cutover, no compat). |

## Part 1 — CLI Argument Contract

Every first-party MCP server **must** parse these CLI args via `argparse` in
`main()`. Servers that need path-like configuration **require** the relevant
args; they do not fall back to hardcoded paths.

### Required args (all servers)

| Arg | Type | Required | Purpose |
| --- | --- | --- | --- |
| `--data-dir` | string | **Yes** | Canonical data root. Server resolves all storage paths relative to this. |
| `--transport` | string | No (default `stdio`) | MCP transport. |
| `--log-level` | string | No (default from env or `INFO`) | Logging level. |

### Optional args (server-specific)

| Arg | Type | Servers | Purpose |
| --- | --- | --- | --- |
| `--workspace-root` | string | all | Logical workspace root for relative path resolution. |
| `--config` | string | all | Path to optional JSON/YAML server config file. |
| `--server-id` | string | all | Explicit identity override for diagnostics. |
| `--readonly` | flag | storage, context | Disable write-side operations. |
| `--backend` | string | guard | Backend selection (`passthrough`, `azure_content_safety`, etc.). |

### What goes in env vars (not CLI args)

Env vars are for **behavioral configuration** that doesn't change the process
launch shape:

| Category | Examples |
| --- | --- |
| API keys | `WEB_SEARCH_API_KEY`, `AZURE_AI_CONTENT_SAFETY_KEY` |
| Feature flags | `LISA_MEMORY_EMBEDDING_ENABLED`, `LISA_GUARD_LOG_BLOCKED` |
| Backend endpoints | `WEB_SEARCH_ENDPOINT`, `AZURE_AI_CONTENT_SAFETY_ENDPOINT` |
| External service config | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| Log formatting | `LOG_FORMAT` (`json`, `text`) |

**Env vars must not be used for:** data directory, database path, storage
root, content root path, or any path that determines where files are read from
or written to. Those are CLI args.

### Resolution rules

1. **Path-like config** (`--data-dir`, `--workspace-root`, `--config`):
   CLI args only. If a required arg is missing, the server prints an error and
   exits with code 2. No env-var fallback, no hardcoded default.
2. **Behavioral config** (API keys, feature flags, endpoints):
   Env vars are the primary source. CLI args may override env for debugging
   (e.g. `--log-level DEBUG` overrides `LOG_LEVEL=INFO`).
3. **Derived paths**: Servers derive subpaths from `--data-dir` using
   server-specific conventions (e.g. `lisa-storage` uses
   `{data_dir}/storage/`, `lisa-context` uses `{data_dir}/memory.db`).
   These derived paths are never configurable via env — only the root
   `--data-dir` is passed, and the server owns the subpath layout.

### Shared CLI parser

A shared argparse helper lives in `mcp_server/src/lisa_mcp_config/cli.py`:

```python
"""Shared CLI argument parser for all first-party MCP servers."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def build_parser(server_name: str, *, supports_readonly: bool = False) -> argparse.ArgumentParser:
    """Build the standard argument parser for one MCP server.

    Every first-party server calls this in ``main()`` and adds server-specific
    args as needed.
    """
    parser = argparse.ArgumentParser(
        prog=server_name,
        description=f"{server_name} — Lisa Framework MCP server",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help="Canonical data root. All storage paths are resolved relative to this.",
    )
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=None,
        help="Logical workspace root for relative path resolution.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON/YAML server config file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Defaults to LOG_LEVEL env var or INFO.",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio"],
        help="MCP transport (only stdio is supported in v1).",
    )
    parser.add_argument(
        "--server-id",
        type=str,
        default=None,
        help="Explicit server identity override for diagnostics.",
    )
    if supports_readonly:
        parser.add_argument(
            "--readonly",
            action="store_true",
            default=False,
            help="Disable write-side operations.",
        )
    return parser


def resolve_data_dir(cli_value: str) -> Path:
    """Resolve the data directory from a CLI arg.

    No env-var fallback, no hardcoded default. The host is responsible
    for providing a sensible default; the server just uses what it gets.
    """
    path = Path(cli_value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(level: str, *, server_name: str) -> logging.Logger:
    """Configure standard logging for one MCP server."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    return logging.getLogger(server_name)
```

### Server `main()` pattern

Every first-party server's `main()` follows this shape:

```python
def main() -> None:
    from lisa_mcp_config.cli import build_parser, resolve_data_dir, configure_logging

    parser = build_parser("lisa-context", supports_readonly=True)
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-context")
    data_dir = resolve_data_dir(args.data_dir)

    # Server-specific config resolution — all derived from data_dir
    db_path = data_dir / "memory.db"

    # Initialize server state
    ...

    mcp.run(transport=args.transport)
```

## Part 2 — Environment Variable Philosophy

### Rule

**CLI args own path-like configuration. Env vars own behavioral configuration.**

| Concern | Mechanism | Examples |
| --- | --- | --- |
| Where to store data | CLI arg | `--data-dir` |
| Where to read content | CLI arg | `--workspace-root` |
| Which backend to use | Env var or CLI arg | `LISA_GUARD_BACKEND` env, `--backend` CLI |
| API keys | Env var | `WEB_SEARCH_API_KEY`, `OPENAI_API_KEY` |
| Feature toggles | Env var | `LISA_MEMORY_EMBEDDING_ENABLED=1` |
| Log formatting | Env var | `LOG_FORMAT=json` |
| External endpoints | Env var | `WEB_SEARCH_ENDPOINT` |

### What is removed

The following env-var-based path resolution is **removed** (hard cutover, no
compat):

| Removed env var | Replaced by |
| --- | --- |
| `LISA_MCP_DB_PATH` | `--data-dir` (server derives `memory.db`) |
| `LISA_MEMORY_DB_PATH` | `--data-dir` (server derives `memory.db`) |
| `LISA_STORAGE_ROOT` | `--data-dir` (server derives `storage/`) |
| `LISA_DATA_DIR` | `--data-dir` (CLI arg, not env) |

Servers that previously read these env vars must be updated to derive paths
from `--data-dir` exclusively.

### What stays in env

| Env var | Used by | Purpose |
| --- | --- | --- |
| `WEB_SEARCH_API_KEY` | `lisa-search` (via `kernel.config.settings`) | Brave Search API key |
| `WEB_SEARCH_ENDPOINT` | `lisa-search` | Brave Search endpoint override |
| `WEB_SEARCH_MARKET` | `lisa-search` | Search market |
| `OPENAI_API_KEY` | host completion callbacks | Model provider key |
| `OPENAI_BASE_URL` | host completion callbacks | Model provider endpoint |
| `SDK_CONTENT_ROOTS` | `lisa-content` (via `kernel.content_roots`) | Content root discovery (behavioral: which roots to scan) |
| `LOG_LEVEL` | all servers | Default log level (overridable by `--log-level`) |
| `LOG_FORMAT` | all servers | Log output format |

## Part 3 — Controlled Third-Party MCP Server Model

### Launch policy registry

Hosts maintain a registry of approved MCP server launch specs. Only registered
specs can be compiled into `MCPClientConfig` and spawned.

```python
# kernel/mcp_launch/policy.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RuntimeKind(str, Enum):
    PYTHON = "python"       # python -m package.server
    UV = "uv"                # uv run --package mcp-server python -m package.server
    VENV_BIN = "venv_bin"    # .venv/Scripts/lisa-context.exe (console_scripts entry point)
    NODE = "node"            # node server.js
    NPX = "npx"              # npx -y @scope/server
    BINARY = "binary"        # /usr/local/bin/server


@dataclass(frozen=True, slots=True)
class EnvPolicy:
    """Controls what environment variables the subprocess receives."""

    inherit_parent: bool = True
    """When True, the subprocess inherits the parent process environment."""

    forced_overrides: dict[str, str] = field(default_factory=dict)
    """Env vars the host forces into the subprocess (e.g. API keys)."""

    allowed_keys: list[str] | None = None
    """When set, only these keys (plus forced_overrides) are passed. None = pass all inherited."""

    denied_keys: list[str] = field(default_factory=list)
    """Env vars to strip from the inherited environment."""


@dataclass(frozen=True, slots=True)
class CwdPolicy:
    """Controls the working directory of the subprocess."""

    kind: str = "inherit"
    """One of: 'inherit' (parent cwd), 'fixed' (explicit path), 'workspace' (workspace-relative)."""

    path: str | None = None
    """Path for kind='fixed' or kind='workspace'."""


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Controls which MCP tools from this server are exposed to the model."""

    allowed_tools: list[str] | None = None
    """Allowlist of tool names. None = allow all."""

    denied_tools: list[str] = field(default_factory=list)
    """Denylist of tool names to strip from the catalog."""


@dataclass(frozen=True, slots=True)
class LaunchTimeouts:
    """Timeouts for subprocess lifecycle."""

    startup_seconds: float = 30.0
    """Max time to wait for server initialization."""

    call_seconds: float = 60.0
    """Default per-tool-call timeout (enforced by host, not MCP protocol)."""


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    """One approved MCP server launch specification.

    The host registers these in a policy registry. Only registered specs
    can be compiled into ``MCPClientConfig`` and spawned.
    """

    server_id: str
    """Stable identifier used for routing, logging, and tool-name prefixing."""

    runtime_kind: RuntimeKind
    """How to launch the server (python, uv, node, npx, binary)."""

    module: str | None = None
    """Python module path for runtime_kind=PYTHON (e.g. 'lisa_context.server')."""

    package: str | None = None
    """uv package name for runtime_kind=UV (e.g. 'mcp-server')."""

    entry_point: str | None = None
    """Console_scripts entry point name for runtime_kind=VENV_BIN (e.g. 'lisa-context').
    Defaults to server_id when not set."""

    venv_dir: str | None = None
    """Path to the virtual environment root for runtime_kind=VENV_BIN.
    The entry point is resolved as ``<venv_dir>/Scripts/<entry_point>`` on Windows
    or ``<venv_dir>/bin/<entry_point>`` on POSIX."""

    command_path: str | None = None
    """Explicit command path for runtime_kind=BINARY or NODE."""

    script_path: str | None = None
    """Script path for runtime_kind=NODE."""

    npx_package: str | None = None
    """npx package spec for runtime_kind=NPX (e.g. '@modelcontextprotocol/server-memory')."""

    cli_args: list[str] = field(default_factory=list)
    """Additional CLI args to pass after the standard contract args."""

    env_policy: EnvPolicy = field(default_factory=EnvPolicy)
    cwd_policy: CwdPolicy = field(default_factory=CwdPolicy)
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    timeouts: LaunchTimeouts = field(default_factory=LaunchTimeouts)

    # Host-provided args injected at compile time
    data_dir: str | None = None
    """Resolved data directory. The host sets this; the compiler passes it as --data-dir."""

    workspace_root: str | None = None
    """Resolved workspace root. Passed as --workspace-root when set."""

    readonly: bool = False
    """When True, passes --readonly to servers that support it."""


@dataclass(frozen=True, slots=True)
class MCPServerRegistry:
    """Registry of approved MCP server specs."""

    _specs: dict[str, MCPServerSpec] = field(default_factory=dict)

    def register(self, spec: MCPServerSpec) -> None:
        if spec.server_id in self._specs:
            raise ValueError(f"MCP server '{spec.server_id}' is already registered")
        self._specs[spec.server_id] = spec

    def get(self, server_id: str) -> MCPServerSpec | None:
        return self._specs.get(server_id)

    def is_approved(self, server_id: str) -> bool:
        return server_id in self._specs

    def all_specs(self) -> list[MCPServerSpec]:
        return list(self._specs.values())
```

### Runtime adapter (compiler)

The compiler takes an `MCPServerSpec` and produces a concrete
`MCPClientConfig` that `MCPToolPool` can spawn.

```python
# kernel/mcp_launch/compiler.py

from __future__ import annotations

import os
from pathlib import Path

from kernel.mcp_launch.policy import (
    CwdPolicy,
    EnvPolicy,
    MCPServerSpec,
    RuntimeKind,
)
from kernel.tools.mcp_client import MCPClientConfig


def compile_env(policy: EnvPolicy) -> dict[str, str]:
    """Build the subprocess environment from an EnvPolicy."""
    if not policy.inherit_parent:
        env: dict[str, str] = {}
    else:
        env = dict(os.environ)
    for key in policy.denied_keys:
        env.pop(key, None)
    if policy.allowed_keys is not None:
        env = {k: v for k, v in env.items() if k in policy.allowed_keys}
    env.update(policy.forced_overrides)
    return env


def compile_cwd(policy: CwdPolicy, *, workspace_root: str | None = None) -> str | Path | None:
    """Resolve the working directory from a CwdPolicy."""
    if policy.kind == "inherit":
        return None
    if policy.kind == "fixed":
        return Path(policy.path or ".")
    if policy.kind == "workspace":
        if workspace_root is None:
            return None
        return Path(workspace_root) / (policy.path or ".")
    return None


def compile_command(spec: MCPServerSpec) -> tuple[str, list[str]]:
    """Build (command, args) for one runtime kind.

    Returns the base command and args before standard CLI contract args
    are appended.
    """
    if spec.runtime_kind == RuntimeKind.PYTHON:
        if not spec.module:
            raise ValueError(f"PYTHON runtime requires 'module' for server '{spec.server_id}'")
        return "python", ["-m", spec.module]

    if spec.runtime_kind == RuntimeKind.UV:
        if not spec.package or not spec.module:
            raise ValueError(f"UV runtime requires 'package' and 'module' for server '{spec.server_id}'")
        return "uv", ["run", "--package", spec.package, "python", "-m", spec.module]

    if spec.runtime_kind == RuntimeKind.VENV_BIN:
        if not spec.venv_dir:
            raise ValueError(f"VENV_BIN runtime requires 'venv_dir' for server '{spec.server_id}'")
        entry_point = spec.entry_point or spec.server_id
        if sys.platform == "win32":
            command_path = str(Path(spec.venv_dir) / "Scripts" / f"{entry_point}.exe")
        else:
            command_path = str(Path(spec.venv_dir) / "bin" / entry_point)
        return command_path, []

    if spec.runtime_kind == RuntimeKind.NODE:
        if not spec.command_path or not spec.script_path:
            raise ValueError(f"NODE runtime requires 'command_path' and 'script_path' for server '{spec.server_id}'")
        return spec.command_path, [spec.script_path]

    if spec.runtime_kind == RuntimeKind.NPX:
        if not spec.npx_package:
            raise ValueError(f"NPX runtime requires 'npx_package' for server '{spec.server_id}'")
        return "npx", ["-y", spec.npx_package]

    if spec.runtime_kind == RuntimeKind.BINARY:
        if not spec.command_path:
            raise ValueError(f"BINARY runtime requires 'command_path' for server '{spec.server_id}'")
        return spec.command_path, []

    raise ValueError(f"Unknown runtime kind '{spec.runtime_kind}' for server '{spec.server_id}'")


def compile_mcp_config(spec: MCPServerSpec) -> MCPClientConfig:
    """Compile one approved MCPServerSpec into a concrete MCPClientConfig.

    This is the only function that produces MCPClientConfig entries for
    MCPToolPool. The host calls this after registry approval.
    """
    if not spec.data_dir:
        raise ValueError(
            f"Cannot compile MCP server '{spec.server_id}': data_dir is required. "
            "The host must resolve --data-dir before compiling."
        )

    command, base_args = compile_command(spec)

    # Build standard CLI contract args
    cli_args: list[str] = ["--data-dir", spec.data_dir]
    if spec.workspace_root:
        cli_args.extend(["--workspace-root", spec.workspace_root])
    if spec.readonly:
        cli_args.append("--readonly")
    cli_args.extend(spec.cli_args)

    all_args = [*base_args, *cli_args]
    env = compile_env(spec.env_policy)
    cwd = compile_cwd(spec.cwd_policy, workspace_root=spec.workspace_root)

    return MCPClientConfig(
        server_id=spec.server_id,
        command=command,
        args=all_args,
        env=env,
        cwd=cwd,
    )


def compile_registry(
    registry: "MCPServerRegistry",
    *,
    data_dir: str,
    workspace_root: str | None = None,
) -> list[MCPClientConfig]:
    """Compile all approved server specs from a registry into MCPClientConfig entries."""
    configs: list[MCPClientConfig] = []
    for spec in registry.all_specs():
        # Inject host-provided args
        resolved_spec = MCPServerSpec(
            **{**spec.__dict__, "data_dir": data_dir, "workspace_root": workspace_root}
        )
        configs.append(compile_mcp_config(resolved_spec))
    return configs
```

### Tool policy enforcement

Tool allow/deny is enforced at two levels:

1. **Registration time**: The host validates that the spec's `tool_policy`
   doesn't conflict with other registered servers.
2. **Runtime**: After `MCPToolPool.list_tools()` returns, the host filters
   the tool catalog against the `tool_policy` before passing schemas to the
   model. This is done in the turn runner or a wrapper around `MCPToolPool`.

```python
# kernel/mcp_launch/tool_filter.py

from __future__ import annotations

from kernel.mcp_launch.policy import ToolPolicy
from kernel.tools.mcp_client import MCPToolSpec


def filter_tools(tools: list[MCPToolSpec], policy: ToolPolicy) -> list[MCPToolSpec]:
    """Filter a tool list by allow/deny policy."""
    result = tools
    if policy.allowed_tools is not None:
        allowed = set(policy.allowed_tools)
        result = [t for t in result if t.name in allowed]
    if policy.denied_tools:
        denied = set(policy.denied_tools)
        result = [t for t in result if t.name not in denied]
    return result
```

## Part 4 — Tier 1 MCP Servers

Three new MCP servers, each following the shared CLI contract.

### `lisa-guard` — content safety server

**Package**: `mcp_server/src/lisa_guard/`
**Entry point**: `lisa_guard.server:main`
**Script**: `lisa-guard`

Tools:

| Tool | Args | Returns |
| --- | --- | --- |
| `validate_input` | `query: str`, `surface_type: str`, `surface_id: str`, `memory_scope: str`, `thread_owner_id: str` | JSON: `{"allowed": bool, "blocked_reason": str\|null, "sanitized_text": str\|null, "annotations": {}}` |
| `validate_output` | `text: str`, `surface_type: str`, `surface_id: str`, `memory_scope: str`, `thread_owner_id: str` | Same shape |
| `guard_configured` | (none) | `{"configured": true, "backend": "passthrough"}` |

Behavior:

- Default backend: `passthrough` (allows everything, sanitizes nothing).
- Backend selection via `--backend` CLI arg (default `passthrough`).
- When a guard blocks, the server logs a structured audit record to stderr:
  `{"event": "guard.blocked", "direction": "input"\|"output", "surface_id": "...", "blocked_reason": "...", "text_len": N}`
- The server owns audit logging. Hosts using `MCPInputGuard`/`MCPOutputGuard`
  adapters do not need an event bus for guardrail audits.
- `--readonly` is not supported (guards are stateless).

Env vars (behavioral):

| Env var | Default | Purpose |
| --- | --- | --- |
| `LISA_GUARD_BACKEND` | `passthrough` | Backend selection (overridden by `--backend`). |
| `LISA_GUARD_LOG_BLOCKED` | `1` | Whether to log blocked attempts. |
| `AZURE_AI_CONTENT_SAFETY_KEY` | (none) | API key for Azure backend (future). |
| `AZURE_AI_CONTENT_SAFETY_ENDPOINT` | (none) | Endpoint for Azure backend (future). |

### `lisa-storage` — blob I/O server

**Package**: `mcp_server/src/lisa_storage/`
**Entry point**: `lisa_storage.server:main`
**Script**: `lisa-storage`

Tools (22, one per `BlobIO` method):

| Tool | Args | Returns |
| --- | --- | --- |
| `read_unified_state` | (none) | JSON `UnifiedState` or `null` |
| `write_unified_state` | `state_json: str` | `{"ok": bool}` |
| `read_blob_bytes` | `container: str`, `blob_path: str` | Base64 string or `null` |
| `write_blob_bytes` | `container: str`, `blob_path: str`, `payload_b64: str`, `content_type: str` | `{"ok": bool}` |
| `read_blob_text` | `container: str`, `blob_path: str` | Text or `null` |
| `write_blob_text` | `container: str`, `blob_path: str`, `payload: str`, `content_type: str` | `{"ok": bool}` |
| `read_surf_workflow_session` | `workspace_id: str`, `surface_id: str` | JSON `SurfSessionState` or `null` |
| `write_surf_workflow_session` | `session_json: str` | `{"ok": bool}` |
| `delete_surf_workflow_session` | `workspace_id: str`, `surface_id: str` | `{"ok": bool}` |
| `read_surf_screenshot` | `blob_path: str` | Base64 string or `null` |
| `write_surf_trace` | `trace_json: str`, `screenshots_json: str`, `artifact_prefix: str` | `{"ok": bool}` |
| `read_user_context_overlays` | (none) | JSON or `null` |
| `write_user_context_overlays` | `overlays_json: str` | `{"ok": bool}` |
| `read_surface_context_overlays` | (none) | JSON or `null` |
| `write_surface_context_overlays` | `overlays_json: str` | `{"ok": bool}` |
| `read_proactive_recipients` | (none) | JSON or `null` |
| `write_proactive_recipients` | `registry_json: str` | `{"ok": bool}` |
| `read_user_reminders` | (none) | JSON or `null` |
| `write_user_reminders` | `registry_json: str` | `{"ok": bool}` |
| `read_thread_state` | (none) | JSON or `null` |
| `write_thread_state` | `state_json: str` | `{"ok": bool}` |
| `read_memory_snapshot` | (none) | JSON or `null` |
| `write_memory_snapshot` | `snapshot_json: str` | `{"ok": bool}` |

Behavior:

- Filesystem-backed. Root: `{data_dir}/storage/`.
- Container subdirectories under root. Path-safe resolution (no `..` traversal).
- Bytes use base64 encoding over the wire.
- Structured types (`UnifiedState`, `SurfSessionState`, overlays, etc.) serialize
  as JSON strings, deserialized via `model_validate` on the adapter side.
- `--readonly` disables all write tools (they return `{"ok": false, "error": "readonly"}`).

Env vars: none (all config is via CLI args).

### `lisa-content` — content registry server

**Package**: `mcp_server/src/lisa_content/`
**Entry point**: `lisa_content.server:main`
**Script**: `lisa-content`

Tools:

| Tool | Args | Returns |
| --- | --- | --- |
| `read_authored_text` | `relative_path: str` | Text content or empty string |
| `list_content_roots` | (none) | JSON array of root paths |
| `invalidate_cache` | (none) | `{"ok": true}` |

Behavior:

- Uses `kernel.content_roots.discover_installed_content_roots()` +
  `discover_configured_content_roots()` to find content roots at startup.
- `read_authored_text` searches all discovered roots, returns first match
  or empty string (same logic as `DiskContentRegistry`).
- `--readonly` is not applicable (content server is read-only by design).

Env vars (behavioral):

| Env var | Default | Purpose |
| --- | --- | --- |
| `SDK_CONTENT_ROOTS` | (none) | Additional content roots to scan (path-separated). |

## Part 5 — Kernel MCP-Client Adapters

Each adapter implements the corresponding Protocol by calling MCP tools through
a persistent or per-call MCP session.

### Warm vs cold servers (pool lifecycle)

`MCPClientConfig` carries two runtime fields that govern pool lifecycle:

| Field | Default | Meaning |
| --- | --- | --- |
| `warm` | `False` | When `True`, this server is a candidate for the host-owned long-lived pool: spawned once at runtime init and kept alive across turns (no per-turn spawn/teardown). Cold servers (`False`) keep the original per-turn spawn/teardown lifecycle. |
| `required` | `False` | Meaningful only with `warm=True`. When `True`, startup failure after bounded retry-with-backoff raises `MCPConnectionError` (propagates, fails the turn or startup loudly). When `False`, startup failure is logged and the pool proceeds without that server's tools. Ignored when `warm=False`. |

These are **runtime** fields, applied by the host post-`compile_registry` via
`dataclasses.replace`. The launch policy layer (`MCPServerSpec` /
`MCPServerRegistry`) stays pure and runtime-agnostic — it does not know about
warm-ness.

The host owns the warm pool lifecycle:

1. At runtime init: build warm configs, create `MCPToolPool(warm_configs)`,
   `await pool.start()`, stash on host state.
2. Pass the started pool through `TurnHooks.mcp_tool_pool`; pass cold configs
   through `TurnHooks.mcp_tool_servers` (the per-turn fallback).
3. `run_turn` reuses the host pool directly (no `start()`/`aclose()` on it).
4. At process exit (CLI REPL finally / desktop `on_cleanup` / tray child exit):
   `await pool.aclose()`.

Mid-session crashes of warm servers are recovered event-drivenly via
`MCPToolPool.restart_warm(server_id)` from `call_tool` / `list_tools` when a
session is missing or errors — not via a background scheduler. This respects
the "no cron in lightweight hosts" rule: eager-start is a one-time startup
cost; crash-restart is event-driven.

Non-warm designation: `lisa-operations` is intentionally NOT warmed. It
exposes every operation as a tool, so its tool surface changes when packs are
installed/uninstalled and a warm pool's cached `_tool_index` would go stale.
`lisa-content` and `lisa-context` have stable tool surfaces and are warmed.

`MCPToolPool.refresh_tool_index()` is the public entry point for hosts that
change a server's tool surface at runtime (e.g. after installing/uninstalling
packs) to rebuild the merged catalog without restarting the pool.

### `kernel/context/mcp_guard.py`

```python
class MCPInputGuard:
    """InputGuard backed by an MCP guard server."""
    def __init__(self, config: MCPClientConfig) -> None: ...
    async def validate(self, query: str, surface: SurfaceContext) -> GuardResult: ...

class MCPOutputGuard:
    """OutputGuard backed by an MCP guard server."""
    def __init__(self, config: MCPClientConfig) -> None: ...
    async def validate(self, text: str, surface: SurfaceContext) -> GuardResult: ...
```

The adapter opens a stdio session, calls `validate_input`/`validate_output`,
parses the JSON response, and returns a `GuardResult`. The server handles
audit logging internally — the adapter does not need an event bus.

### `kernel/context/mcp_storage.py`

```python
class MCPBlobIO:
    """BlobIO backed by an MCP storage server."""
    def __init__(self, config: MCPClientConfig) -> None: ...
    # All 22 BlobIO methods, each calling the corresponding MCP tool.
```

### `kernel/context/mcp_content_registry.py`

```python
class MCPContentRegistry:
    """ContentRegistry backed by an MCP content server."""
    def __init__(self, config: MCPClientConfig) -> None: ...
    def read_authored_text(self, relative_path: str) -> str: ...
    def invalidate(self) -> None: ...
```

### Session strategy

v1 uses **per-call stdio sessions** (open → call → close), matching the
existing `MCPMemoryClient` pattern in `kernel/mcp_client.py`. This is simple
but has subprocess-spawn overhead per call. A follow-up can add session
pooling if profiling shows it's needed.

### Exports

Add to `kernel/context/__init__.py`:

```python
from kernel.context.mcp_guard import MCPInputGuard, MCPOutputGuard
from kernel.context.mcp_storage import MCPBlobIO
from kernel.context.mcp_content_registry import MCPContentRegistry
```

### ContextSeams

`ContextSeams` defaults remain the in-process implementations
(`PassThroughInputGuard`, `PassThroughOutputGuard`, `DiskContentRegistry`).
Hosts that want MCP-backed seams override the fields with the MCP adapters.
This is additive — no cutover, no dual path, just two first-class options.

## Part 6 — Host Wiring (Little Lisa + General Hosts)

### Little Lisa changes

1. **Resolve `--data-dir` once** at startup (already exists via Typer `--data-dir`).
2. **Build an `MCPServerRegistry`** with approved server specs for all first-party
   servers (context, storage, content, guard, search, surf, operations).
3. **Compile the registry** into `list[MCPClientConfig]` using
   `compile_registry(registry, data_dir=str(data_dir))`.
4. **Populate `TurnHooks.mcp_tool_servers`** with the compiled configs.
5. **Replace ad-hoc `MCPMemoryClient` spawn** with the new compiler where
   appropriate. The memory seam can either stay on `MCPMemoryClient` (legacy)
   or migrate to `MCPToolPool` (new). v1 keeps `MCPMemoryClient` for memory;
   the new servers go through `MCPToolPool`.

### General host pattern

```python
# Any host
from kernel.mcp_launch.policy import MCPServerRegistry, MCPServerSpec, RuntimeKind, EnvPolicy
from kernel.mcp_launch.compiler import compile_registry

registry = MCPServerRegistry()
registry.register(MCPServerSpec(
    server_id="lisa-context",
    runtime_kind=RuntimeKind.VENV_BIN,
    venv_dir="/path/to/.venv",
    env_policy=EnvPolicy(inherit_parent=True, forced_overrides={"WEB_SEARCH_API_KEY": "..."}),
))
registry.register(MCPServerSpec(
    server_id="lisa-guard",
    runtime_kind=RuntimeKind.VENV_BIN,
    venv_dir="/path/to/.venv",
))
# ... more servers

configs = compile_registry(registry, data_dir="/path/to/data")
hooks = TurnHooks(mcp_tool_servers=configs, ...)
```

### Third-party server registration

```python
registry.register(MCPServerSpec(
    server_id="mem0",
    runtime_kind=RuntimeKind.NPX,
    npx_package="@mem0/mcp-server",
    env_policy=EnvPolicy(
        inherit_parent=True,
        forced_overrides={"MEM0_API_KEY": "..."},
        denied_keys=["OPENAI_API_KEY"],  # don't leak host keys
    ),
    tool_policy=ToolPolicy(
        allowed_tools=["add_memory", "search_memory", "get_all_memories"],
    ),
))
```

Unregistered specs are rejected — the host cannot spawn a server that isn't
in the registry.

## Part 7 — Migration of Existing Servers

All existing first-party servers must adopt the shared CLI contract.

### `lisa-context`

- Add `argparse` with `--data-dir` (required), `--transport`, `--log-level`, `--readonly`.
- Remove `LISA_MCP_DB_PATH` / `LISA_MEMORY_DB_PATH` env-var reads.
- Derive `memory.db` path from `--data-dir`.
- `main()` calls `build_parser("lisa-context", supports_readonly=True)`.

### `lisa-operations`

- Add `argparse` with `--data-dir` (required), `--transport`, `--log-level`.
- No path config needed today, but `--data-dir` is required for consistency.

### `lisa-search`

- Add `argparse` with `--data-dir` (required), `--transport`, `--log-level`.
- `WEB_SEARCH_API_KEY` stays in env (behavioral config).

### `lisa-surf`

- Add `argparse` with `--data-dir` (required), `--transport`, `--log-level`.
- No path config needed today, but `--data-dir` is required for consistency.

## Part 8 — Implementation Steps

### Phase 1: Launch infrastructure (no server changes yet)

1. Create `mcp_server/src/lisa_mcp_config/__init__.py` — shared config package.
2. Create `mcp_server/src/lisa_mcp_config/cli.py` — shared argparse helper
   (`build_parser`, `resolve_data_dir`, `configure_logging`).
3. Create `kernel/mcp_launch/__init__.py` — launch policy package.
4. Create `kernel/mcp_launch/policy.py` — `MCPServerSpec`, `MCPServerRegistry`,
   `EnvPolicy`, `CwdPolicy`, `ToolPolicy`, `LaunchTimeouts`, `RuntimeKind`.
5. Create `kernel/mcp_launch/compiler.py` — `compile_env`, `compile_cwd`,
   `compile_command`, `compile_mcp_config`, `compile_registry`.
6. Create `kernel/mcp_launch/tool_filter.py` — `filter_tools`.

### Phase 2: Migrate existing servers to CLI contract

1. Update `mcp_server/src/lisa_context/server.py` — add argparse, remove
   `LISA_MCP_DB_PATH`/`LISA_MEMORY_DB_PATH` reads, derive from `--data-dir`.
2. Update `mcp_server/src/lisa_operations/server.py` — add argparse.
3. Update `mcp_server/src/lisa_search/server.py` — add argparse.
4. Update `mcp_server/src/lisa_surf/server.py` — add argparse.
5. Update `mcp_server/pyproject.toml` — add new script entries for Tier 1 servers.

### Phase 3: Tier 1 servers

1. Create `mcp_server/src/lisa_guard/server.py` — guard server (3 tools).
2. Create `mcp_server/src/lisa_storage/server.py` — storage server (22 tools).
3. Create `mcp_server/src/lisa_content/server.py` — content server (3 tools).
4. Create `mcp_server/src/lisa_guard/py.typed`, `lisa_storage/py.typed`,
    `lisa_content/py.typed`.

### Phase 4: Kernel adapters

1. Create `kernel/context/mcp_guard.py` — `MCPInputGuard`, `MCPOutputGuard`.
2. Create `kernel/context/mcp_storage.py` — `MCPBlobIO`.
3. Create `kernel/context/mcp_content_registry.py` — `MCPContentRegistry`.
4. Update `kernel/context/__init__.py` — add 4 new exports.

### Phase 5: Host wiring

1. Update `examples/little_lisa/main.py` — build `MCPServerRegistry`, compile
    to `MCPClientConfig` list, populate `TurnHooks.mcp_tool_servers`.
2. Replace ad-hoc `_mcp_server_env()` with the new compiler-based approach.

### Phase 6: Tests

1. Create `mcp_server/tests/test_cli_contract.py` — test argparse parsing
    for all servers (required `--data-dir`, precedence, error on missing).
2. Create `mcp_server/tests/test_tier1_servers.py` — test tool registration
    and dispatch for guard, storage, content.
3. Create `kernel/tests/test_mcp_launch.py` — test policy validation,
    runtime adapter compilation, env/cwd projection, tool filtering.
4. Create `kernel/tests/test_mcp_adapters.py` — test `MCPInputGuard`,
    `MCPOutputGuard`, `MCPBlobIO`, `MCPContentRegistry` via in-process MCP.
5. Update `mcp_server/tests/test_mcp_servers.py` — update existing tests
    for new CLI contract (pass `--data-dir` in test fixtures).

### Phase 7: Documentation

1. Update `mcp_server/README.md` — add Tier 1 servers, document CLI contract.
2. Update `docs/MCP_NATIVE_DESIGN.md` — add Tier 1 servers to built-in servers
    table, note launch contract, note `guardrail.audit` event bus elimination.
3. Update `docs/KERNEL_ARCHITECTURE.md` — add `kernel/mcp_launch/` to module map.

## Part 9 — Verification

1. **CLI contract**: Every first-party server exits with code 2 when
   `--data-dir` is missing. Every server accepts `--data-dir`, `--transport`,
   `--log-level`.
2. **No env-var path fallback**: `grep -r "LISA_MCP_DB_PATH\|LISA_MEMORY_DB_PATH\|LISA_STORAGE_ROOT\|LISA_DATA_DIR"`
   in `mcp_server/src/` returns zero results after migration.
3. **Policy gate**: `MCPServerRegistry.is_approved("unknown")` returns `False`.
   `compile_registry` only compiles approved specs.
4. **Runtime adapter**: `compile_command` produces correct `(command, args)` for
   each `RuntimeKind`.
5. **Env projection**: `compile_env` with `inherit_parent=True` includes `PATH`.
   With `inherit_parent=False` it doesn't. `denied_keys` strips entries.
   `allowed_keys` filters to allowlist. `forced_overrides` wins.
6. **Tool filtering**: `filter_tools` respects allow/deny lists.
7. **Data-dir propagation**: Host `--data-dir /tmp/test` causes
   `lisa-storage` to write to `/tmp/test/storage/` and `lisa-context` to write
   to `/tmp/test/memory.db`.
8. **Existing tests**: `kernel/tests/test_guardrails.py` passes unchanged
   (in-process defaults are not modified).
9. **Import boundaries**: `kernel/tests/test_context_import_boundaries.py`
   passes (no new `app.*` imports).
10. **Mypy**: `mypy kernel/mcp_launch/ kernel/context/mcp_*.py
    mcp_server/src/lisa_mcp_config/ mcp_server/src/lisa_guard/
    mcp_server/src/lisa_storage/ mcp_server/src/lisa_content/` is clean.

## Part 10 — What This Does Not Change

- The `EventBus` Protocol and `InProcessEventBus` default stay. Hosts using
  in-process guards still use the event bus for `guardrail.audit`. Hosts using
  `MCPInputGuard`/`MCPOutputGuard` don't need it (the server handles audit).
- The `ContextSeams` defaults stay in-process. MCP adapters are opt-in.
- The `MCPToolPool` class is unchanged — it receives `MCPClientConfig` entries
  from the compiler.
- The `MCPMemoryClient` in `kernel/mcp_client.py` stays for backward
  compatibility with the memory seam. A future change can migrate it to the
  new compiler.
- The slice system, front-door router, IPSAR FSM, and operation registry are
  unchanged.

## Part 11 — File Inventory

### New files

| File | Purpose |
| --- | --- |
| `mcp_server/src/lisa_mcp_config/__init__.py` | Shared config package |
| `mcp_server/src/lisa_mcp_config/cli.py` | Shared argparse helper |
| `mcp_server/src/lisa_guard/__init__.py` | Guard server package |
| `mcp_server/src/lisa_guard/server.py` | Guard server (3 tools) |
| `mcp_server/src/lisa_guard/py.typed` | Type marker |
| `mcp_server/src/lisa_storage/__init__.py` | Storage server package |
| `mcp_server/src/lisa_storage/server.py` | Storage server (22 tools) |
| `mcp_server/src/lisa_storage/py.typed` | Type marker |
| `mcp_server/src/lisa_content/__init__.py` | Content server package |
| `mcp_server/src/lisa_content/server.py` | Content server (3 tools) |
| `mcp_server/src/lisa_content/py.typed` | Type marker |
| `kernel/mcp_launch/__init__.py` | Launch policy package |
| `kernel/mcp_launch/policy.py` | Policy types + registry |
| `kernel/mcp_launch/compiler.py` | Runtime adapter + config compiler |
| `kernel/mcp_launch/tool_filter.py` | Tool allow/deny filter |
| `kernel/context/mcp_guard.py` | `MCPInputGuard` + `MCPOutputGuard` |
| `kernel/context/mcp_storage.py` | `MCPBlobIO` |
| `kernel/context/mcp_content_registry.py` | `MCPContentRegistry` |
| `kernel/tests/test_mcp_launch.py` | Policy + compiler tests |
| `kernel/tests/test_mcp_adapters.py` | Adapter tests |
| `mcp_server/tests/test_cli_contract.py` | CLI contract tests |
| `mcp_server/tests/test_tier1_servers.py` | Tier 1 server tests |

### Modified files

| File | Changes |
| --- | --- |
| `mcp_server/src/lisa_context/server.py` | Add argparse, remove env-var path reads |
| `mcp_server/src/lisa_operations/server.py` | Add argparse |
| `mcp_server/src/lisa_search/server.py` | Add argparse |
| `mcp_server/src/lisa_surf/server.py` | Add argparse |
| `mcp_server/pyproject.toml` | Add 3 script entries + `lisa_mcp_config` package |
| `kernel/context/__init__.py` | Add 4 new exports |
| `examples/little_lisa/main.py` | Build registry, compile configs, populate `mcp_tool_servers` |
| `mcp_server/tests/test_mcp_servers.py` | Update for CLI contract |
| `mcp_server/README.md` | Document new servers + CLI contract |
| `docs/MCP_NATIVE_DESIGN.md` | Update built-in servers table |
| `docs/KERNEL_ARCHITECTURE.md` | Add `kernel/mcp_launch/` to module map |
