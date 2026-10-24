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
from enum import StrEnum

from kernel.mcp_binary import MCPBinaryDownload

__all__ = [
    "AliasSpec",
    "AmbientSource",
    "AuthSpec",
    "CwdPolicy",
    "EnvPolicy",
    "LaunchTimeouts",
    "MCPServerRegistry",
    "MCPServerSpec",
    "ProxyPolicy",
    "RuntimeKind",
    "ToolPolicy",
]


class RuntimeKind(StrEnum):
    """How to launch an MCP server subprocess."""

    PYTHON = "python"  # python -m package.server
    UV = "uv"  # uv run --package mcp-server python -m package.server
    VENV_BIN = "venv_bin"  # .venv/Scripts/lisa-context.exe (console_scripts entry point)
    NODE = "node"  # node server.js
    NPX = "npx"  # npx -y @scope/server
    BINARY = "binary"  # /usr/local/bin/server
    HTTP = "http"  # streamable-HTTP MCP endpoint (no subprocess)
    IN_PROCESS = "in_process"  # first-party server run in-process via memory transport


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


class AuthKind(StrEnum):
    """How a network (HTTP) MCP server authenticates the client."""

    NONE = "none"  # no auth — only allowed for loopback/allowlisted hosts
    BEARER = "bearer"  # static bearer token, resolved at connect time
    OAUTH2_PKCE = "oauth2-pkce"  # RFC 9728 OAuth 2.1 authorization-code + PKCE
    HEADER = "header"  # custom header name + token (e.g. X-Scoped-Budget-Token)


@dataclass(frozen=True, slots=True)
class AuthSpec:
    """Authentication for a network (runtime_kind=HTTP) MCP server.

    Declared per-server in ``mcp_servers.yaml`` under ``auth:``. The manifest
    loader expands ``${VAR}`` references (e.g. ``"${GITHUB_TOKEN}"``) before
    building the spec. The compiler carries it onto
    :class:`~kernel.tools.mcp_client.MCPClientConfig.auth`; the pool resolves
    it at connect time.

    Two first-class kinds:

    - ``bearer``: a static ``Authorization: Bearer <token>`` header, resolved
      at connect time (so a cold server re-reads the token on every connect).
    - ``oauth2-pkce``: RFC 9728 OAuth 2.1 authorization-code + PKCE via the MCP
      SDK's auth provider. **Warm-only** — the loopback redirect doesn't fit
      per-turn cold spawns. ``auth_server`` may be omitted and is discovered
      from the server URL's ``/.well-known/oauth-protection-metadata``.
    - ``header``: a custom header name + token value (e.g.
      ``X-Scoped-Budget-Token: <token>``). Used by API-gateway-scoped tokens
      that don't follow the ``Authorization: Bearer`` convention. The header
      name and token are ``${VAR}``-expanded at load time.
    """

    kind: AuthKind = AuthKind.NONE
    """How to authenticate. Default ``none`` (no auth)."""

    token: str | None = None
    """Bearer token (kind=BEARER only). ``${VAR}``-expanded at load time."""

    header_name: str | None = None
    """Custom header name (kind=HEADER only, e.g. ``X-Scoped-Budget-Token``)."""

    client_id: str | None = None
    """OAuth client id (kind=OAUTH2_PKCE only)."""

    scopes: tuple[str, ...] = ()
    """OAuth scopes to request (kind=OAUTH2_PKCE only)."""

    auth_server: str | None = None
    """Authorization server URL (kind=OAUTH2_PKCE only). When None, it is
    discovered from the server URL's well-known metadata."""

    def is_authenticated(self) -> bool:
        """True when this spec provides a usable credential."""
        if self.kind is AuthKind.BEARER:
            return bool(self.token)
        if self.kind is AuthKind.OAUTH2_PKCE:
            return bool(self.client_id)
        if self.kind is AuthKind.HEADER:
            return bool(self.header_name) and bool(self.token)
        return False


class AmbientSource(StrEnum):
    """Runtime sources the proxy can resolve ambient field values from.

    The host registers a resolver callable for each source; the proxy
    calls it at dispatch time to inject the current value. The model
    never sees or controls these fields.
    """

    USER_SLUG = "user_slug"
    """The current user's slug (user isolation)."""

    MODE_ID = "mode_id"
    """The current mode's id (agent identity)."""

    APP_ID = "app_id"
    """The host's application id (surface boundary)."""


@dataclass(frozen=True, slots=True)
class ProxyPolicy:
    """Controls how the MCP proxy transforms tool schemas and arguments.

    The :class:`~kernel.tools.mcp_client.MCPToolPool` is a proxy between
    the LLM and N external MCP servers. A ``ProxyPolicy`` on a server's
    config tells the proxy to enforce ambient scope boundaries: fields
    the model must not see or set (user identity, agent identity, app
    identity) are **stripped from the tool schema** before it reaches the
    model and **injected into the arguments** before they reach the
    server. The model cannot escape the boundary because it never knows
    the fields exist.

    Declared per-server in ``mcp_servers.yaml`` under ``proxy_policy:``.
    The manifest compiler carries it onto :class:`~kernel.tools.mcp_client.MCPClientConfig`;
    the pool applies it at two chokepoints:

    1. **Schema projection** (catalog → model): ambient fields are removed
       from ``input_schema.properties`` and ``required`` so the model's tool
       schema shows only the fields it controls.
    2. **Arg injection** (model → server): ambient fields are injected
       from the host runtime resolver, overriding any value the model
       might have sent (defense in depth — the schema already hides them).
    """

    ambient: dict[str, AmbientSource] = field(default_factory=dict)
    """Maps native argument names to runtime sources. The proxy resolves
    each source at dispatch time and injects the value into the outgoing
    arguments. Keys are the server's native parameter names (e.g.
    ``user_id``, ``agent_id``, ``app_id``); values are
    :class:`AmbientSource` members."""

    structured_filters: bool = False
    """When True, ambient fields for search/list tools are injected as
    ``{"key": value}`` entries inside a ``filters.AND`` list (Mem0's
    structured filter syntax). When False, they are top-level arguments.
    The proxy determines which tools are "search-like" by checking for a
    ``filters`` parameter in the tool's input schema."""

    def ambient_field_names(self) -> set[str]:
        """Return the set of native field names that are ambient-injected."""
        return set(self.ambient.keys())


@dataclass(frozen=True, slots=True)
class LaunchTimeouts:
    """Timeouts for subprocess lifecycle."""

    startup_seconds: float = 30.0
    """Max time to wait for server initialization."""

    call_seconds: float = 60.0
    """Default per-tool-call timeout (enforced by host, not MCP protocol)."""


@dataclass(frozen=True, slots=True)
class AliasSpec:
    """One generic alias mapping a stable model-facing name to a native tool.

    Declared per-server in ``mcp_servers.yaml`` under ``aliases:``. The host
    registers each alias as a synthetic tool after ``pool.start()`` so the
    model sees a stable name (e.g. ``cloudflare.search``) instead of the
    filesystem-derived namespaced id (e.g.
    ``lisa_packs.cloudflare.cloudflare-api.search``).
    """

    alias_name: str
    """The stable, model-facing tool name (e.g. ``cloudflare.search``)."""

    native_tool: str
    """The native MCP tool name on the server (e.g. ``search``)."""

    description: str = ""
    """Optional description override for the synthetic tool."""


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

    binary_download: MCPBinaryDownload | None = None
    """Version-pinned GitHub release source for a runtime_kind=BINARY server."""

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
    proxy_policy: ProxyPolicy = field(default_factory=ProxyPolicy)
    timeouts: LaunchTimeouts = field(default_factory=LaunchTimeouts)

    # Host-provided args injected at compile time
    data_dir: str | None = None
    """Resolved data directory. The host sets this; the compiler passes it as
    ``--data-dir`` (or whatever ``data_dir_arg`` names)."""

    workspace_root: str | None = None
    """Resolved workspace root. Passed as --workspace-root when set."""

    workspace_root_arg: str | None = "--workspace-root"
    """How to inject the workspace root into the server's CLI.

    - ``"--workspace-root"`` (default): pass as ``--workspace-root <path>``
      (first-party contract; the shared parser in ``mcp_server.config.cli``
      accepts it).
    - any other flag string: pass as that flag + the resolved path.
    - ``None``: do not inject a workspace-root flag at all. Used for
      third-party servers whose own CLI does not know this flag (the flag is
      injected before the server's own subcommand, so an unknown option is
      a hard parse error that kills the subprocess).
    """

    readonly: bool = False
    """When True, passes --readonly to servers that support it."""

    url: str | None = None
    """Endpoint URL for runtime_kind=HTTP (streamable-HTTP MCP).
    None for all subprocess-based runtimes. ${VAR} expansion happens in
    the manifest loader before the spec is built."""

    headers: dict[str, str] = field(default_factory=dict)
    """HTTP headers for runtime_kind=HTTP (e.g. Authorization bearer token).
    ${VAR} expansion happens in the manifest loader before the spec is built."""

    auth: AuthSpec = field(default_factory=AuthSpec)
    """Authentication for runtime_kind=HTTP (static bearer via
    :class:`AuthSpec`). Rides through to ``MCPClientConfig.auth`` and is
    resolved at connect time. For non-loopback hosts, ``auth`` must be
    authenticated (or explicitly ``none`` with an allowlisted host) or
    compilation is rejected."""

    warm: bool = False
    """When True, this server is a candidate for the host-owned long-lived
    pool: it is spawned eagerly at runtime init and kept alive across turns
    (with bounded retry and event-driven restart on crash). When False
    (default), the server is cold: it is **not** connected at pool
    ``start()`` — it is lazily connected on first ``list_tools`` or
    ``call_tool`` that needs it, and its tools are absent from the initial
    catalog."""

    required: bool = False
    """Meaningful only when ``warm=True``. When True, startup failure (after
    bounded retry-with-backoff) raises :class:`MCPConnectionError`,
    propagating to the caller so a load-bearing warm server fails the turn
    or startup loudly. When False, startup failure is logged and the pool
    proceeds without that server's tools. Ignored when ``warm=False``."""

    disabled: bool = False
    """When True, the manifest loader skips this spec entirely — it is not
    registered in the :class:`MCPServerRegistry` and cannot be compiled.
    This is the config-only escape hatch for disabling a server without a
    code change. The manifest loader also honors a YAML-level ``disabled:
    true`` key on each entry, which short-circuits before spec
    construction; this field is for programmatic registration paths."""

    aliases: tuple[AliasSpec, ...] = field(default_factory=tuple)
    """Generic aliases mapping stable model-facing names to native tools.
    The host registers each alias as a synthetic tool after pool start so
    the model sees a stable name (e.g. ``cloudflare.search``) instead of
    the namespaced server id."""

    default_available_tools: tuple[str, ...] = field(default_factory=tuple)
    """Bare native tool names on this server that should be ambient-available
    to every mode, mirroring the ``default_available: true`` flag pack
    ``.tool.yml`` tools declare. The SDK catalog loader merges their
    published ids (``<capability>.<tool>`` for capability servers,
    ``server_id.tool`` otherwise) into each mode's
    ``allowed_tool_ids`` unless the mode opts out via
    ``accept_default_capabilities: false`` or lists the published id in
    ``excluded_tool_ids``. Use this to ship ambient MCP-native capabilities
    (e.g. ``storage.read_file``) without a delegate pack tool and
    without forcing every mode author to re-list the id."""

    capability: str | None = None
    """Capability this server is an implementation of (e.g. ``storage``,
    ``thread``, ``memory``, ``fetch``).

    When set, the server competes for ownership of that capability with
    every other declared implementation (built-in host-manifest servers and
    pack-contributed servers alike). The selected winner publishes its
    tools under the capability prefix (``<capability>.<tool>``) instead of
    ``<server_id>.<tool>``, so the model-facing tool ids — and every
    ambient/default id downstream — are identical regardless of which
    implementation (Python, Go, third-party) is selected. Unselected
    candidates are not registered at all; there is exactly one live
    implementation per capability per pool."""

    tool_map: dict[str, str] = field(default_factory=dict)
    """Published→native tool name mapping for capability servers.

    Keys are the canonical published names (``<capability>.<tool>`` minus
    the prefix — e.g. ``read_file`` for ``storage.read_file``); values are
    the native MCP wire tool names on this server. Entries absent from the
    map pass through under their published name. This only maps names; a
    server claiming a capability must itself implement that capability's
    argument, result, safety, and semantic contract. Provider-specific
    adaptation belongs in the pack's MCP server, not in the host."""

    capability_tools: tuple[str, ...] = field(default_factory=tuple)
    """Canonical bare tool names in this capability's required contract.

    Every implementation of one capability must declare the same required
    tool set. The pool publishes these names (mapping through ``tool_map``
    where declared) and fails explicitly if a server does not expose them.
    This allows the host to validate Python/Go/pack swappability without
    knowing implementation-specific tools.
    """

    capability_optional_tools: tuple[str, ...] = field(default_factory=tuple)
    """Additional published tools this implementation supports beyond the
    required capability contract. These are opt-in extensions and may vary
    across providers; a declared extension must be present on the server."""


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
