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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kernel.mcp_launch.policy import (
    AliasSpec,
    AmbientSource,
    CwdPolicy,
    EnvPolicy,
    LaunchTimeouts,
    MCPServerSpec,
    ProxyPolicy,
    RuntimeKind,
    ToolPolicy,
)

__all__ = [
    "MCPServerManifestError",
    "MemoryBackendConfig",
    "MemoryBackendSpec",
    "load_mcp_server_manifest",
]

#: Matches ``${VAR}`` references for in-place env expansion. Supports both
#: whole-string refs (``"${MEM0_API_KEY}"``) and embedded refs
#: (``"Bearer ${MEM0_API_KEY}"``).
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class MCPServerManifestError(ValueError):
    """Raised when an mcp_servers manifest is structurally invalid."""


@dataclass(frozen=True, slots=True)
class MemoryBackendSpec:
    """One declared memory backend — a peer in the memory.backends list.

    Each backend is provider-agnostic: it declares its connection ``kind``
    (http, stdio) plus the tool names the generic aliases route to. The host
    builds the right client per ``kind`` and wires it behind the generic
    ``memory.add`` / ``memory.search`` / ``memory.list`` synthetic aliases.
    No backend-specific knowledge lives in the schema — adding a new backend
    is a YAML entry, not a code change.
    """

    id: str
    kind: str  # "http" | "stdio"
    tool_names: dict[str, str] = field(default_factory=dict)
    result_maps: dict[str, str] = field(default_factory=dict)
    """Dotted import paths to output-normalization adapters, keyed by
    generic operation name (``add``, ``search``, ``list``). Each value is a
    ``module.path:function_name`` string resolved lazily by the host when
    building the alias. When empty (the default for first-party backends like
    lisa_memory that already return the canonical shape), the alias passes
    the raw ``output_text`` through unchanged.

    This is the symmetric counterpart of ``tool_names``: ``tool_names`` maps
    generic input names to native tool names (input dispatch), and
    ``result_maps`` maps generic operation names to output adapters (output
    normalization). Together they make the alias boundary the single
    chokepoint for both directions of backend-specific translation."""
    # http connection details (kind="http")
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    # stdio connection details (kind="stdio")
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    agent_id: str = "lisa"


@dataclass(frozen=True, slots=True)
class MemoryBackendConfig:
    """Memory backend configuration: which backend is default + the backends list.

    The host uses ``default_backend`` to decide which
    :class:`~kernel.memory_protocol.MemoryStore` to inject as the default,
    and ``backends`` to build each one. Each backend in the list is a peer
    (lisa-context, mem0, a future backend, …); the default picks one by id.
    Chaining (fan-out writes / merged reads across backends) is a future
    extension — the router already holds a ``backends`` dict keyed by id.
    """

    default_backend: str | None = None
    backends: tuple[MemoryBackendSpec, ...] = ()


def _expand_env_value(raw: Any) -> str:
    """Expand ${VAR} references against the parent environment at load time.

    A value like ``"${MEM0_API_KEY}"`` becomes the value of ``MEM0_API_KEY``
    (or empty string when unset). Embedded refs like
    ``"Bearer ${MEM0_API_KEY}"`` are expanded in place. Literal values pass
    through unchanged.
    """
    if not isinstance(raw, str):
        return str(raw)

    def _replace(match: re.Match[str]) -> str:
        return os.getenv(match.group(1), "")

    return _ENV_REF_RE.sub(_replace, raw)


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
    _validate_default_backend_server(memory_config, specs, path)
    return specs, memory_config


def _validate_default_backend_server(
    memory_config: MemoryBackendConfig,
    specs: list[MCPServerSpec],
    path: Path,
) -> None:
    """Ensure the default memory backend's id matches a registered server.

    The host pins memory-alias dispatch to ``default_backend`` by id (see
    ``_register_memory_synthetic_tools``); that id must equal a
    ``server_id`` in the pool so the scoped call resolves. HTTP backends are
    also registered as tool servers, so this is required for any default
    backend kind. A mismatch is a manifest error, not a silent fallback.
    """
    if memory_config.default_backend is None:
        return
    declared_ids = {s.id for s in memory_config.backends}
    if memory_config.default_backend not in declared_ids:
        raise MCPServerManifestError(
            f"memory.default_backend {memory_config.default_backend!r} is not in "
            f"memory.backends {sorted(declared_ids)} in {path}"
        )
    server_ids = {s.server_id for s in specs}
    if memory_config.default_backend not in server_ids:
        # The first-party backends (lisa_memory in the manifest, lisa-memory in
        # the pool) are registered in code, not in the manifest's servers
        # list, so allow them through. Any other id must have a matching
        # servers entry to be routable as the default.
        if memory_config.default_backend in {"lisa-memory", "lisa_memory"}:
            return
        raise MCPServerManifestError(
            f"memory.default_backend {memory_config.default_backend!r} has no matching "
            f"server_id in the servers list {sorted(server_ids)} in {path} — the host "
            "pins alias dispatch to this id, so it must match a registered server."
        )


def _parse_memory_section(raw: Any) -> MemoryBackendConfig:
    if not isinstance(raw, dict):
        return MemoryBackendConfig()
    default_backend = raw.get("default_backend")
    if default_backend is not None:
        default_backend = str(default_backend).strip() or None
    backends_raw = raw.get("backends") or []
    backends = _parse_memory_backends(backends_raw)
    return MemoryBackendConfig(default_backend=default_backend, backends=backends)


def _parse_memory_backends(raw: Any) -> tuple[MemoryBackendSpec, ...]:
    """Parse the memory.backends list into MemoryBackendSpec entries.

    Each entry is a peer: an id, a kind (http | stdio), connection details,
    and tool_names mapping the generic aliases to that backend's tools.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise MCPServerManifestError("'memory.backends' must be a list of backend entries")
    specs: list[MemoryBackendSpec] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise MCPServerManifestError(f"memory.backends[{index}] must be a mapping")
        backend_id = str(entry.get("id") or "").strip()
        if not backend_id:
            raise MCPServerManifestError(f"memory.backends[{index}]: 'id' is required")
        if backend_id in seen_ids:
            raise MCPServerManifestError(f"memory.backends[{index}]: duplicate backend id {backend_id!r}")
        seen_ids.add(backend_id)
        kind = str(entry.get("kind") or "").strip()
        if kind not in {"http", "stdio"}:
            raise MCPServerManifestError(
                f"memory.backends[{index}] {backend_id!r}: 'kind' must be 'http' or 'stdio'"
            )
        tool_names_raw = entry.get("tool_names") or {}
        if not isinstance(tool_names_raw, dict):
            raise MCPServerManifestError(
                f"memory.backends[{index}] {backend_id!r}: 'tool_names' must be a mapping"
            )
        tool_names = {str(alias): str(tool) for alias, tool in tool_names_raw.items()}
        agent_id = str(entry.get("agent_id") or "lisa")
        result_maps_raw = entry.get("result_maps") or {}
        if not isinstance(result_maps_raw, dict):
            raise MCPServerManifestError(
                f"memory.backends[{index}] {backend_id!r}: 'result_maps' must be a mapping"
            )
        result_maps = {str(op): str(path) for op, path in result_maps_raw.items()}
        if kind == "http":
            url = str(entry.get("url") or "").strip()
            if not url:
                raise MCPServerManifestError(
                    f"memory.backends[{index}] {backend_id!r}: 'url' is required for kind=http"
                )
            headers_raw = entry.get("headers") or {}
            if not isinstance(headers_raw, dict):
                raise MCPServerManifestError(
                    f"memory.backends[{index}] {backend_id!r}: 'headers' must be a mapping"
                )
            headers = {str(k): _expand_env_value(v) for k, v in headers_raw.items()}
            specs.append(
                MemoryBackendSpec(
                    id=backend_id,
                    kind=kind,
                    tool_names=tool_names,
                    result_maps=result_maps,
                    url=url,
                    headers=headers,
                    agent_id=agent_id,
                )
            )
        else:  # stdio
            command = str(entry.get("command") or "").strip()
            if not command:
                raise MCPServerManifestError(
                    f"memory.backends[{index}] {backend_id!r}: 'command' is required for kind=stdio"
                )
            args_raw = entry.get("args") or []
            if not isinstance(args_raw, list):
                raise MCPServerManifestError(
                    f"memory.backends[{index}] {backend_id!r}: 'args' must be a list"
                )
            args = [_expand_env_value(a) for a in args_raw]
            env_raw = entry.get("env") or {}
            if not isinstance(env_raw, dict):
                raise MCPServerManifestError(
                    f"memory.backends[{index}] {backend_id!r}: 'env' must be a mapping"
                )
            env = {str(k): _expand_env_value(v) for k, v in env_raw.items()}
            cwd = entry.get("cwd")
            if cwd is not None:
                cwd = str(cwd) or None
            specs.append(
                MemoryBackendSpec(
                    id=backend_id,
                    kind=kind,
                    tool_names=tool_names,
                    result_maps=result_maps,
                    command=command,
                    args=args,
                    env=env,
                    cwd=cwd,
                    agent_id=agent_id,
                )
            )
    return tuple(specs)


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
    proxy_policy = _parse_proxy_policy(entry.get("proxy_policy"), server_id, index, path)
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

    url: str | None = None
    headers: dict[str, str] = {}
    if runtime_kind is RuntimeKind.HTTP:
        url = str(entry.get("url") or "").strip()
        if not url:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': 'url' is required for runtime_kind=http"
            )
        headers_raw = entry.get("headers") or {}
        if not isinstance(headers_raw, dict):
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': 'headers' must be a mapping"
            )
        headers = {str(k): _expand_env_value(v) for k, v in headers_raw.items()}

    aliases = _parse_aliases(entry.get("aliases"), server_id, index, path)

    return MCPServerSpec(
        server_id=server_id,
        runtime_kind=runtime_kind,
        module=entry.get("module"),
        package=entry.get("package"),
        entry_point=entry.get("entry_point"),
        venv_dir=_expand_env_value(entry.get("venv_dir")) if entry.get("venv_dir") else None,
        command_path=_expand_env_value(entry.get("command_path")) if entry.get("command_path") else None,
        script_path=_expand_env_value(entry.get("script_path")) if entry.get("script_path") else None,
        npx_package=entry.get("npx_package"),
        uv_directory=_expand_env_value(entry.get("uv_directory")) if entry.get("uv_directory") else None,
        cli_args=[str(a) for a in cli_args_raw],
        data_dir_arg=data_dir_arg,
        data_dir_subdir=data_dir_subdir,
        ensure_data_dir=ensure_data_dir,
        env_policy=EnvPolicy(inherit_parent=True, forced_overrides=forced),
        cwd_policy=cwd_policy,
        tool_policy=tool_policy,
        proxy_policy=proxy_policy,
        timeouts=timeouts,
        url=url,
        headers=headers,
        warm=bool(entry.get("warm", False)),
        required=bool(entry.get("required", False)),
        disabled=False,
        aliases=aliases,
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


def _parse_aliases(
    raw: Any, server_id: str, index: int, path: Path
) -> tuple[AliasSpec, ...]:
    """Parse the optional ``aliases`` block on a server entry.

    Each entry maps a stable model-facing name to a native tool on the
    server. Format::

        aliases:
          - alias_name: cloudflare.search
            native_tool: search
            description: Search the Cloudflare API spec
          - alias_name: cloudflare.execute
            native_tool: execute

    Returns an empty tuple when absent.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: 'aliases' must be a list"
        )
    specs: list[AliasSpec] = []
    seen_names: set[str] = set()
    for alias_index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': aliases[{alias_index}] must be a mapping"
            )
        alias_name = str(entry.get("alias_name") or "").strip()
        if not alias_name:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': aliases[{alias_index}] requires 'alias_name'"
            )
        if alias_name in seen_names:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': duplicate alias_name {alias_name!r}"
            )
        seen_names.add(alias_name)
        native_tool = str(entry.get("native_tool") or "").strip()
        if not native_tool:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': aliases[{alias_index}] requires 'native_tool'"
            )
        description = str(entry.get("description") or "").strip()
        specs.append(
            AliasSpec(
                alias_name=alias_name,
                native_tool=native_tool,
                description=description,
            )
        )
    return tuple(specs)


def _parse_proxy_policy(
    raw: Any, server_id: str, index: int, path: Path
) -> ProxyPolicy:
    """Parse the optional ``proxy_policy`` block on a server entry.

    Returns a default (empty) :class:`ProxyPolicy` when absent — the proxy
    passes schemas and arguments through unchanged. When present, the
    ``ambient`` mapping tells the proxy which native argument fields are
    host-injected (stripped from the model's schema, injected from runtime
    at dispatch time).
    """
    if raw is None:
        return ProxyPolicy()
    if not isinstance(raw, dict):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: 'proxy_policy' must be a mapping"
        )

    ambient_raw = raw.get("ambient") or {}
    if not isinstance(ambient_raw, dict):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': 'proxy_policy.ambient' must be a mapping"
        )
    ambient: dict[str, AmbientSource] = {}
    for native_field, source_str in ambient_raw.items():
        source_str = str(source_str or "").strip()
        try:
            ambient[str(native_field)] = AmbientSource(source_str)
        except ValueError as exc:
            valid = ", ".join(s.value for s in AmbientSource)
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': 'proxy_policy.ambient.{native_field}' "
                f"must be one of [{valid}], got {source_str!r}"
            ) from exc

    structured_filters = bool(raw.get("structured_filters", False))

    return ProxyPolicy(
        ambient=ambient,
        structured_filters=structured_filters,
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