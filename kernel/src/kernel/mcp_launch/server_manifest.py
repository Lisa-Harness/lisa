"""Declarative external MCP server manifest loader.

Mirrors :mod:`kernel.model_manifest`: a YAML file declares external MCP
servers (e.g. Mem0, Letta), and the host merges the resulting
:class:`MCPServerSpec` entries into its
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

Binary servers may declare a pinned GitHub ``binary_download`` source;
the runtime resolves and verifies it before starting the MCP pool.

Data-dir adaptation: each server may specify ``data_dir_arg`` (the flag name
to inject, or ``null`` for servers that manage their own storage),
``data_dir_subdir`` (a namespaced subfolder of the host data-dir the server
owns), and ``ensure_data_dir`` (pre-create the dir before spawning).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from kernel.mcp_binary import MCPBinaryDownload
from kernel.mcp_launch.policy import (
    AliasSpec,
    AmbientSource,
    AuthKind,
    AuthSpec,
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
    "load_mcp_server_manifest",
]

#: Matches ``${VAR}`` references for in-place env expansion. Supports both
#: whole-string refs (``"${MEM0_API_KEY}"``) and embedded refs
#: (``"Bearer ${MEM0_API_KEY}"``).
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Capability ids double as prefixes of model-facing tool ids
#: (``<capability>.<tool>``), so they must be bare lowercase slugs.
_CAPABILITY_NAME_RE = re.compile(r"[a-z][a-z0-9_-]*")


class MCPServerManifestError(ValueError):
    """Raised when an mcp_servers manifest is structurally invalid."""


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
) -> list[MCPServerSpec]:
    """Load MCP server specs from YAML.

    Returns an empty list when the file does not exist, so hosts can call
    this unconditionally and treat a missing manifest as "no external
    servers".

    Raises :class:`MCPServerManifestError` on structural problems (not a
    mapping, server entries missing required fields, bad runtime kind, …).
    """
    if not path.exists():
        return []

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise MCPServerManifestError(f"Failed to parse {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise MCPServerManifestError(f"mcp_servers manifest must be a mapping: {path}")

    if "memory" in raw:
        raise MCPServerManifestError(
            f"{path}: top-level 'memory' is no longer supported; declare a server "
            "with capability: memory"
        )
    return _parse_server_entries(raw.get("servers") or [], path)


def _parse_server_entries(raw_servers: Any, path: Path) -> list[MCPServerSpec]:
    if not isinstance(raw_servers, list):
        raise MCPServerManifestError("'servers' must be a list of server entries")
    specs: list[MCPServerSpec] = []
    for index, entry in enumerate(raw_servers):
        if not isinstance(entry, dict):
            raise MCPServerManifestError(f"servers[{index}] in {path} must be a mapping")
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
            f"servers[{index}] '{server_id}': invalid runtime_kind {runtime_raw!r} (valid: {valid})"
        ) from exc

    binary_download = _parse_binary_download(entry, server_id, index, runtime_kind)

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
    workspace_root_arg_raw = entry.get("workspace_root_arg", "--workspace-root")
    workspace_root_arg = None if workspace_root_arg_raw is None else str(workspace_root_arg_raw)
    data_dir_subdir = entry.get("data_dir_subdir")
    if data_dir_subdir is not None:
        data_dir_subdir = str(data_dir_subdir)
    ensure_data_dir = bool(entry.get("ensure_data_dir", False))

    cli_args_raw = entry.get("cli_args") or []
    if not isinstance(cli_args_raw, list):
        raise MCPServerManifestError(f"servers[{index}] '{server_id}': 'cli_args' must be a list")
    # Expand ${VAR} references against the parent environment, mirroring
    # every other path-like field (venv_dir, command_path, url, …). The
    # lisa-thread server's ``--thread-dir ${LISA_THREAD_DIR}`` /
    # ``--user-slug ${LISA_USER_SLUG}`` rely on this; storing the literal
    # ``${...}`` string (the pre-2026-08-25 ``str(a)`` passthrough) sent
    # the subprocess a literal ``${LISA_USER_SLUG}`` path that resolved
    # to a nonexistent store, so ``thread_read_range`` returned ``[]``.
    cli_args = [_expand_env_value(a) for a in cli_args_raw]

    url: str | None = None
    headers: dict[str, str] = {}
    auth = AuthSpec()
    if runtime_kind is RuntimeKind.HTTP:
        url = _expand_env_value(entry.get("url") or "").strip()
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
        auth = _parse_auth(entry.get("auth"), server_id, index, path)

    aliases = _parse_aliases(entry.get("aliases"), server_id, index, path)
    default_available_tools = _parse_default_available_tools(
        entry.get("default_available_tools"), server_id, index, path
    )
    capability, tool_map, capability_tools, capability_optional_tools = _parse_capability_contract(
        entry,
        server_id,
        index,
        path,
        default_available_tools,
    )

    return MCPServerSpec(
        server_id=server_id,
        capability=capability,
        tool_map=tool_map,
        capability_tools=capability_tools,
        capability_optional_tools=capability_optional_tools,
        runtime_kind=runtime_kind,
        module=entry.get("module"),
        package=entry.get("package"),
        entry_point=entry.get("entry_point"),
        venv_dir=_expand_env_value(entry.get("venv_dir")) if entry.get("venv_dir") else None,
        command_path=_expand_env_value(entry.get("command_path"))
        if entry.get("command_path")
        else None,
        binary_download=binary_download,
        script_path=_expand_env_value(entry.get("script_path"))
        if entry.get("script_path")
        else None,
        npx_package=entry.get("npx_package"),
        uv_directory=_expand_env_value(entry.get("uv_directory"))
        if entry.get("uv_directory")
        else None,
        cli_args=cli_args,
        data_dir_arg=data_dir_arg,
        data_dir_subdir=data_dir_subdir,
        workspace_root_arg=workspace_root_arg,
        ensure_data_dir=ensure_data_dir,
        env_policy=EnvPolicy(inherit_parent=True, forced_overrides=forced),
        cwd_policy=cwd_policy,
        tool_policy=tool_policy,
        proxy_policy=proxy_policy,
        timeouts=timeouts,
        url=url,
        headers=headers,
        auth=auth,
        warm=bool(entry.get("warm", False)),
        required=bool(entry.get("required", False)),
        disabled=False,
        aliases=aliases,
        default_available_tools=default_available_tools,
    )


def _parse_binary_download(
    entry: dict[str, Any],
    server_id: str,
    index: int,
    runtime_kind: RuntimeKind,
) -> MCPBinaryDownload | None:
    raw = entry.get("binary_download")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': 'binary_download' must be a mapping"
        )
    if runtime_kind is not RuntimeKind.BINARY:
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': 'binary_download' requires runtime_kind=binary"
        )
    if entry.get("command_path"):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': 'command_path' and 'binary_download' "
            "cannot both be set"
        )

    repository = raw.get("repository")
    version = raw.get("version")
    binary_name = raw.get("binary_name")
    if (
        not isinstance(repository, str)
        or not isinstance(version, str)
        or not isinstance(binary_name, str)
    ):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': binary_download requires string "
            "repository, version, and binary_name fields"
        )
    try:
        return MCPBinaryDownload(
            repository=repository,
            version=version,
            binary_name=binary_name,
        )
    except ValueError as exc:
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': invalid binary_download: {exc}"
        ) from exc


def _parse_bearer_auth(raw: dict[str, Any], server_id: str, index: int) -> AuthSpec:
    """Build a bearer-token AuthSpec, requiring an expanded ``token`` value."""
    token = _expand_env_value(raw.get("token") or "")
    if not token:
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': auth type 'bearer' requires 'token'"
        )
    return AuthSpec(kind=AuthKind.BEARER, token=token)


def _parse_header_auth(raw: dict[str, Any], server_id: str, index: int) -> AuthSpec:
    """Build a named-header AuthSpec, requiring ``header_name`` and ``token``."""
    header_name = str(raw.get("header_name") or "").strip()
    if not header_name:
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': auth type 'header' requires 'header_name'"
        )
    token = _expand_env_value(raw.get("token") or "")
    if not token:
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': auth type 'header' requires 'token'"
        )
    return AuthSpec(kind=AuthKind.HEADER, header_name=header_name, token=token)


def _parse_oauth2_pkce_auth(raw: dict[str, Any], server_id: str, index: int) -> AuthSpec:
    """Build an OAuth2 PKCE AuthSpec, requiring an expanded ``client_id``."""
    client_id = _expand_env_value(raw.get("client_id") or "")
    if not client_id:
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': auth type 'oauth2-pkce' requires 'client_id'"
        )
    scopes_raw = raw.get("scopes") or []
    if not isinstance(scopes_raw, list):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': 'auth.scopes' must be a list"
        )
    auth_server = raw.get("auth_server")
    if auth_server is not None:
        auth_server = _expand_env_value(auth_server) or None
    return AuthSpec(
        kind=AuthKind.OAUTH2_PKCE,
        client_id=client_id,
        scopes=tuple(str(s) for s in scopes_raw),
        auth_server=auth_server,
    )


def _parse_auth(raw: Any, server_id: str, index: int, path: Path) -> AuthSpec:
    """Parse the ``auth:`` block for a network MCP server entry.

    Returns the default :class:`AuthSpec` (kind=``none``) when the block is
    absent. Raises :class:`MCPServerManifestError` on invalid kinds or
    missing required fields. ``${VAR}`` references in ``token`` /
    ``client_id`` / ``auth_server`` are expanded in place.
    """
    if raw is None:
        return AuthSpec()
    if not isinstance(raw, dict):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: 'auth' must be a mapping"
        )
    kind_raw = str(raw.get("type") or raw.get("kind") or "none").strip()
    try:
        kind = AuthKind(kind_raw)
    except ValueError as exc:
        valid = ", ".join(k.value for k in AuthKind)
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}': invalid auth type {kind_raw!r} (valid: {valid})"
        ) from exc

    if kind is AuthKind.BEARER:
        return _parse_bearer_auth(raw, server_id, index)
    if kind is AuthKind.HEADER:
        return _parse_header_auth(raw, server_id, index)
    if kind is AuthKind.OAUTH2_PKCE:
        return _parse_oauth2_pkce_auth(raw, server_id, index)
    return AuthSpec(kind=AuthKind.NONE)


def _parse_tool_policy(raw: Any, server_id: str, index: int, path: Path) -> ToolPolicy:
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


def _parse_aliases(raw: Any, server_id: str, index: int, path: Path) -> tuple[AliasSpec, ...]:
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


def _parse_default_available_tools(
    raw: Any,
    server_id: str,
    index: int,
    path: Path,
) -> tuple[str, ...]:
    """Parse the optional ``default_available_tools`` list on a server entry.

    Each entry is a bare published tool name (e.g. ``read_file``). The host
    namespaces it to ``<capability>.tool`` for capability servers and
    ``server_id.tool`` otherwise, so authors never repeat a prefix. Format::

        default_available_tools:
          - read_file
          - stat_file

    Returns an empty tuple when absent. Duplicates are rejected.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: 'default_available_tools' must be a list"
        )
    seen: set[str] = set()
    names: list[str] = []
    for pos, item in enumerate(raw):
        name = str(item or "").strip()
        if not name:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': "
                f"default_available_tools[{pos}] must be a non-empty string"
            )
        if "." in name:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': "
                f"default_available_tools[{pos}] {name!r} must be a bare tool name, "
                "not a namespaced id (omit the server_id prefix)"
            )
        if name in seen:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': duplicate default_available_tools entry {name!r}"
            )
        seen.add(name)
        names.append(name)
    return tuple(names)


def _parse_capability(raw: Any, server_id: str, index: int, path: Path) -> str | None:
    """Parse the optional ``capability`` claim on a server entry.

    A capability claim (``storage``, ``thread``, ``memory``, …) declares
    that this server is an implementation of that capability. The pool
    resolves one owner per capability at compile time — selected via
    ``providers.capabilities.<name>`` in the session plan or defaulted to
    the built-in host-manifest candidate — and only the winner registers.
    Losing candidates are skipped entirely; the winner publishes tools
    under ``<capability>.<tool>``.

    The name must be a bare slug (``[a-z][a-z0-9_-]*``): capability ids
    are prefixes of model-facing tool ids, so dots and uppercase are
    rejected to keep the published id space unambiguous.
    """
    if raw is None:
        return None
    name = str(raw or "").strip()
    if not name:
        return None
    if not _CAPABILITY_NAME_RE.fullmatch(name):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: invalid capability {name!r} "
            "(must be a bare slug matching [a-z][a-z0-9_-]*)"
        )
    return name


def _parse_tool_map(raw: Any, server_id: str, index: int, path: Path) -> dict[str, str]:
    """Parse the optional ``tool_map`` published→native mapping on a server entry.

    Keys are canonical published tool names (bare, without the capability
    prefix — e.g. ``read_file`` for the published id ``storage.read_file``);
    values are the native MCP wire tool names on this server. Published
    names absent from the map pass through unchanged. Format::

        tool_map:
          read_file: read_file
          append_file: append_file

    Only valid on capability-claiming servers (the map describes how this
    implementation satisfies the capability's published contract).
    Empty keys and values, as well as multiple published tools mapping to
    one native tool, are rejected.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: 'tool_map' must be a mapping"
        )
    tool_map: dict[str, str] = {}
    native_names: set[str] = set()
    for published_raw, native_raw in raw.items():
        published = str(published_raw or "").strip()
        native = str(native_raw or "").strip()
        if not published or not native:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}' in {path}: "
                "tool_map keys and values must be non-empty strings"
            )
        if "." in published:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}' in {path}: "
                f"tool_map key {published!r} must be a bare published tool name, "
                "not a namespaced id (omit the capability prefix)"
            )
        if native in native_names:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}' in {path}: "
                f"tool_map maps multiple published tools to native tool {native!r}"
            )
        native_names.add(native)
        tool_map[published] = native
    return tool_map


def _parse_capability_tools(raw: Any, server_id: str, index: int, path: Path) -> tuple[str, ...]:
    """Parse the canonical bare tool names in a capability's contract."""
    return _parse_capability_tool_names(raw, "capability_tools", server_id, index, path)


def _parse_capability_optional_tools(
    raw: Any, server_id: str, index: int, path: Path
) -> tuple[str, ...]:
    """Parse provider-specific published tools beyond the required contract."""
    return _parse_capability_tool_names(raw, "capability_optional_tools", server_id, index, path)


def _parse_capability_tool_names(
    raw: Any, key: str, server_id: str, index: int, path: Path
) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: '{key}' must be a list"
        )
    names: list[str] = []
    seen: set[str] = set()
    for pos, item in enumerate(raw):
        name = str(item or "").strip()
        if not name:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': {key}[{pos}] must be a non-empty string"
            )
        if "." in name:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': {key} entries must be bare names"
            )
        if name in seen:
            raise MCPServerManifestError(
                f"servers[{index}] '{server_id}': duplicate capability tool {name!r}"
            )
        seen.add(name)
        names.append(name)
    return tuple(names)


def _parse_capability_contract(
    entry: dict[str, Any],
    server_id: str,
    index: int,
    path: Path,
    default_available_tools: tuple[str, ...],
) -> tuple[str | None, dict[str, str], tuple[str, ...], tuple[str, ...]]:
    capability = _parse_capability(entry.get("capability"), server_id, index, path)
    tool_map = _parse_tool_map(entry.get("tool_map"), server_id, index, path)
    required_tools = _parse_capability_tools(entry.get("capability_tools"), server_id, index, path)
    optional_tools = _parse_capability_optional_tools(
        entry.get("capability_optional_tools"), server_id, index, path
    )
    if tool_map and capability is None:
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: 'tool_map' requires 'capability' "
            "(the map describes how this server satisfies the capability's "
            "published tool contract)"
        )
    if capability and not required_tools:
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: 'capability_tools' is required "
            "when 'capability' is set"
        )
    if (required_tools or optional_tools) and capability is None:
        key = "capability_tools" if required_tools else "capability_optional_tools"
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: '{key}' requires 'capability'"
        )

    overlap = set(required_tools) & set(optional_tools)
    declared_tools = set(required_tools) | set(optional_tools)
    undeclared = set(tool_map) - declared_tools
    undeclared.update(set(default_available_tools) - set(required_tools))
    if capability and overlap:
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: tools cannot be both required "
            f"and optional: {sorted(overlap)}"
        )
    if capability and undeclared:
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: capability tool names "
            f"{sorted(undeclared)} are not in capability_tools"
        )
    return capability, tool_map, required_tools, optional_tools


def _parse_proxy_policy(raw: Any, server_id: str, index: int, path: Path) -> ProxyPolicy:
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
    for native_field, source_raw in ambient_raw.items():
        source_str = str(source_raw or "").strip()
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


def _parse_cwd_policy(raw: Any, server_id: str, index: int, path: Path) -> CwdPolicy:
    if raw is None:
        return CwdPolicy()
    if not isinstance(raw, dict):
        raise MCPServerManifestError(
            f"servers[{index}] '{server_id}' in {path}: 'cwd' must be a mapping"
        )
    kind = str(raw.get("kind", "inherit"))
    return CwdPolicy(kind=kind, path=raw.get("path"))
