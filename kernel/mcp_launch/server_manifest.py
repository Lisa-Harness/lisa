"""Declarative external MCP server manifest loader.

Mirrors :mod:`kernel.model_manifest`: a YAML file declares external MCP
servers (e.g. Mem0, Letta) plus a memory-backend designation, and the
host merges the resulting :class:`MCPServerSpec` entries into its
:class:`~kernel.mcp_launch.policy.MCPServerRegistry` alongside the
first-party specs.

This is to MCP servers what ``models.yaml`` is to models: declare once,
the host resolves paths and env at compile time. Adding a new external
server is a YAML edit, not a code change.

Supported runtime kinds per spec: ``python``, ``uv``, ``venv_bin``,
``node``, ``npx``, ``binary`` (see :class:`~kernel.mcp_launch.policy.RuntimeKind`).

Env values support ``${VAR}`` expansion against the parent environment at
load time — behavioral config (API keys, endpoints) is the only thing that
belongs in env; path-like config stays in CLI args per the launch contract.

A server entry may be marked ``disabled: true`` to keep it declared but
excluded from compilation (useful for opt-in backends).

Data-dir adaptation: each server may specify ``data_dir_arg`` (the flag name
to inject, or ``null`` for servers that manage their own storage),
``data_dir_subdir`` (a namespaced subfolder of the host data-dir the server
owns), and ``ensure_data_dir`` (pre-create the dir before spawning).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kernel.mcp_launch.policy import (
    CwdPolicy,
    EnvPolicy,
    LaunchTimeouts,
    MCPServerSpec,
    RuntimeKind,
    ToolPolicy,
)

__all__ = [
    "MemoryBackendConfig",
    "MCPServerManifestError",
    "load_mcp_server_manifest",
]


class MCPServerManifestError(ValueError):
    """Raised when an mcp_servers manifest is structurally invalid."""


@dataclass(frozen=True, slots=True)
class MemoryBackendConfig:
    """Which server_id backs the generic memory aliases, plus per-backend tool-name mapping.

    The host uses ``default_backend`` to decide which client to inject as
    the default :class:`~kernel.memory_protocol.MemoryStore`, and
    ``tool_names`` to map generic alias calls (``add`` / ``search`` /
    ``list``) to a backend's actual MCP tool names — so the generic
    ``memory.search`` synthetic tool can route to either
    ``search_memory`` (lisa-context) or ``search_memories`` (Mem0) based
    on the wired default.
    """

    default_backend: str | None = None
    tool_names: dict[str, dict[str, str]] = field(default_factory=dict)
    # e.g. {"mem0": {"add": "add_memories", "search": "search_memories",
    #                "list": "get_all_memories"}}


def _expand_env_value(raw: Any) -> str:
    """Expand ${VAR} references against the parent environment at load time.

    A value like ``"${MEM0_API_KEY}"`` becomes the value of ``MEM0_API_KEY``
    (or empty string when unset). Literal values pass through unchanged.
    """
    if not isinstance(raw, str):
        return str(raw)
    if raw.startswith("${") and raw.endswith("}") and len(raw) > 3:
        var_name = raw[2:-1]
        return os.getenv(var_name, "")
    return raw


def load_mcp_server_manifest(
    path: Path,
) -> tuple[list[MCPServerSpec], MemoryBackendConfig]:
    """Load external MCP server specs + memory backend config from YAML.

    Returns ``([], MemoryBackendConfig())`` when the file does not exist,
    so hosts can call this unconditionally and treat a missing manifest as
    "no external servers".

    Raises :class:`MCPServerManifestError` on structural problems (not a
    mapping, server entries missing required fields, bad runtime kind, …).
    """
    if not path.exists():
        return [], MemoryBackendConfig()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise MCPServerManifestError(f"Failed to parse {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise MCPServerManifestError(f"mcp_servers manifest must be a mapping: {path}")

    memory_config = _parse_memory_section(raw.get("memory") or {})
    specs = _parse_server_entries(raw.get("servers") or [], path)
    return specs, memory_config


def _parse_memory_section(raw: Any) -> MemoryBackendConfig:
    if not isinstance(raw, dict):
        return MemoryBackendConfig()
    default_backend = raw.get("default_backend")
    if default_backend is not None:
        default_backend = str(default_backend).strip() or None
    backends_raw = raw.get("backends") or {}
    if not isinstance(backends_raw, dict):
        raise MCPServerManifestError("'memory.backends' must be a mapping of backend_id -> tool map")
    tool_names: dict[str, dict[str, str]] = {}
    for backend_id, mapping in backends_raw.items():
        if not isinstance(mapping, dict):
            raise MCPServerManifestError(
                f"'memory.backends.{backend_id}' must be a mapping of alias -> tool name"
            )
        tool_names[str(backend_id)] = {
            str(alias): str(tool_name)
            for alias, tool_name in mapping.items()
        }
    return MemoryBackendConfig(default_backend=default_backend, tool_names=tool_names)


def _parse_server_entries(raw_servers: Any, path: Path) -> list[MCPServerSpec]:
    if not isinstance(raw_servers, list):
        raise MCPServerManifestError("'servers' must be a list of server entries")
    specs: list[MCPServerSpec] = []
    for index, entry in enumerate(raw_servers):
        if not isinstance(entry, dict):
            raise MCPServerManifestError(
                f"servers[{index}] in {path} must be a mapping"
            )
        if entry.get("disabled"):
            continue
        specs.append(_parse_server_entry(entry, index, path))
    return specs


def _parse_server_entry(entry: dict[str, Any], index: int, path: Path) -> MCPServerSpec:
    server_id = entry.get("server_id")
    if not isinstance(server_id, str) or not server_id.strip():
        raise MCPServerManifestError(f"servers[{index}] in {path}: 'server_id' is required")
    runtime_raw = entry.get("runtime_kind")
    if not isinstance(runtime_raw, str):
        raise MCPServerManifestError(f"servers[{index}] '{server_id}': 'runtime_kind' is required")
    try:
        runtime_kind = RuntimeKind(runtime_raw)
    except ValueError as exc:
        valid = ", ".join(kind.value for kind in RuntimeKind)
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': invalid runtime_kind {runtime_raw!r} "
            f"(valid: {valid})"
        ) from exc

    env_raw = entry.get("env") or {}
    if not isinstance(env_raw, dict):
        raise MCPServerManifestError(f"servers[{index}] '{server_id}': 'env' must be a mapping")
    forced = {str(k): _expand_env_value(v) for k, v in env_raw.items()}

    tool_policy = _parse_tool_policy(entry.get("tool_policy"), server_id, index, path)
    timeouts = _parse_timeouts(entry.get("timeouts"))
    cwd_policy = _parse_cwd_policy(entry.get("cwd"), server_id, index, path)

    # data-dir adaptation. data_dir_arg may be null (servers managing their
    # own storage); default "--data-dir" preserves the first-party contract.
    data_dir_arg_raw = entry.get("data_dir_arg", "--data-dir")
    data_dir_arg = None if data_dir_arg_raw is None else str(data_dir_arg_raw)
    data_dir_subdir = entry.get("data_dir_subdir")
    if data_dir_subdir is not None:
        data_dir_subdir = str(data_dir_subdir)
    ensure_data_dir = bool(entry.get("ensure_data_dir", False))

    cli_args_raw = entry.get("cli_args") or []
    if not isinstance(cli_args_raw, list):
        raise MCPServerManifestError(f"servers[{index}] '{server_id}': 'cli_args' must be a list")

    return MCPServerSpec(
        server_id=server_id,
        runtime_kind=runtime_kind,
        module=entry.get("module"),
        package=entry.get("package"),
        entry_point=entry.get("entry_point"),
        venv_dir=entry.get("venv_dir"),
        command_path=entry.get("command_path"),
        script_path=entry.get("script_path"),
        npx_package=entry.get("npx_package"),
        uv_directory=entry.get("uv_directory"),
        cli_args=[str(a) for a in cli_args_raw],
        data_dir_arg=data_dir_arg,
        data_dir_subdir=data_dir_subdir,
        ensure_data_dir=ensure_data_dir,
        env_policy=EnvPolicy(inherit_parent=True, forced_overrides=forced),
        cwd_policy=cwd_policy,
        tool_policy=tool_policy,
        timeouts=timeouts,
    )


def _parse_tool_policy(
    raw: Any, server_id: str, index: int, path: Path
) -> ToolPolicy:
    if raw is None:
        return ToolPolicy()
    if not isinstance(raw, dict):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: 'tool_policy' must be a mapping"
        )
    allowed = raw.get("allowed_tools")
    denied = raw.get("denied_tools") or []
    if allowed is not None and not isinstance(allowed, list):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': 'tool_policy.allowed_tools' must be a list"
        )
    if not isinstance(denied, list):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': 'tool_policy.denied_tools' must be a list"
        )
    return ToolPolicy(
        allowed_tools=[str(t) for t in allowed] if allowed is not None else None,
        denied_tools=[str(t) for t in denied],
    )


def _parse_timeouts(raw: Any) -> LaunchTimeouts:
    if raw is None:
        return LaunchTimeouts()
    if not isinstance(raw, dict):
        return LaunchTimeouts()
    try:
        return LaunchTimeouts(
            startup_seconds=float(raw.get("startup_seconds", 30.0)),
            call_seconds=float(raw.get("call_seconds", 60.0)),
        )
    except (TypeError, ValueError):
        return LaunchTimeouts()


def _parse_cwd_policy(
    raw: Any, server_id: str, index: int, path: Path
) -> CwdPolicy:
    if raw is None:
        return CwdPolicy()
    if not isinstance(raw, dict):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: 'cwd' must be a mapping"
        )
    kind = str(raw.get("kind", "inherit"))
    return CwdPolicy(kind=kind, path=raw.get("path"))