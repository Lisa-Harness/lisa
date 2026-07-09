"""Launch policy types and the approved-server registry.

The registry gates which MCP servers a host may spawn. Only registered
:class:`MCPServerSpec` entries can be compiled into
:class:`~kernel.tools.mcp_client.MCPClientConfig` by
:mod:`kernel.mcp_launch.compiler`.

Path-like configuration (``data_dir``, ``workspace_root``) is injected at
compile time by the host — the spec itself is runtime-agnostic and can be
registered once and reused across environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "CwdPolicy",
    "EnvPolicy",
    "LaunchTimeouts",
    "MCPServerRegistry",
    "MCPServerSpec",
    "RuntimeKind",
    "ToolPolicy",
]


class RuntimeKind(str, Enum):
    """How to launch an MCP server subprocess."""

    PYTHON = "python"  # python -m package.server
    UV = "uv"  # uv run --package mcp-server python -m package.server
    VENV_BIN = "venv_bin"  # .venv/Scripts/lisa-context.exe (console_scripts entry point)
    NODE = "node"  # node server.js
    NPX = "npx"  # npx -y @scope/server
    BINARY = "binary"  # /usr/local/bin/server
    HTTP = "http"  # streamable-HTTP MCP endpoint (no subprocess)


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
    can be compiled into :class:`~kernel.tools.mcp_client.MCPClientConfig`
    and spawned.
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

    uv_directory: str | None = None
    """Working directory for runtime_kind=UV (passed as ``uv run --directory <path>``)."""

    cli_args: list[str] = field(default_factory=list)
    """Additional CLI args to pass after the standard contract args."""

    data_dir_arg: str | None = "--data-dir"
    """How to inject the host data-dir into the server's CLI.

    - ``"--data-dir"`` (default): pass as ``--data-dir <path>`` (first-party contract).
    - any other flag string (e.g. ``"--store-dir"``): pass as that flag + the resolved path.
      Used for third-party servers that accept a data-directory flag under a
      different name.
    - ``None``: do not inject a data-dir flag at all. Used for servers that
      manage their own storage (e.g. a hosted MCP server like Mem0 via npx).
    """

    data_dir_subdir: str | None = None
    """Optional namespaced subfolder under the host data-dir this server owns.

    When set, the actual directory passed to the server is
    ``Path(data_dir) / data_dir_subdir``. Lets a server own a dedicated
    subfolder of the host data-dir (possibly used destructively) without
    colliding with other servers' storage.
    """

    ensure_data_dir: bool = False
    """When True, the runtime creates the (sub)dir before spawning the server.

    Useful when a server expects its storage directory to pre-exist, or when
    the host wants to guarantee a managed subfolder is present before the
    server owns it. The directory is created even when ``data_dir_arg`` is
    ``None`` — a server may want a managed folder it discovers its own way.
    """

    env_policy: EnvPolicy = field(default_factory=EnvPolicy)
    cwd_policy: CwdPolicy = field(default_factory=CwdPolicy)
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    timeouts: LaunchTimeouts = field(default_factory=LaunchTimeouts)

    # Host-provided args injected at compile time
    data_dir: str | None = None
    """Resolved data directory. The host sets this; the compiler passes it as
    ``--data-dir`` (or whatever ``data_dir_arg`` names)."""

    workspace_root: str | None = None
    """Resolved workspace root. Passed as --workspace-root when set."""

    readonly: bool = False
    """When True, passes --readonly to servers that support it."""

    url: str | None = None
    """Endpoint URL for runtime_kind=HTTP (streamable-HTTP MCP).
    None for all subprocess-based runtimes."""

    headers: dict[str, str] = field(default_factory=dict)
    """HTTP headers for runtime_kind=HTTP (e.g. Authorization bearer token).
    ${VAR} expansion happens in the manifest loader before the spec is built."""


@dataclass(frozen=True, slots=True)
class MCPServerRegistry:
    """Registry of approved MCP server specs.

    Only specs added via :meth:`register` can be compiled into
    :class:`~kernel.tools.mcp_client.MCPClientConfig` entries by
    :func:`kernel.mcp_launch.compiler.compile_registry`.
    """

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
