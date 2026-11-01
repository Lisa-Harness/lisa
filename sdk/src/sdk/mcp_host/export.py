"""Export compiled MCP server configs as another agent's client config.

Lisa's first-party servers are ordinary stdio MCP servers. An external agent
host (Claude Desktop, VS Code, Cursor, another harness) can attach to them
directly, without Lisa's turn loop. This module turns the host's
:class:`~kernel.tools.mcp_client.MCPClientConfig` entries — already resolved
for paths, flags, and launch kind — into the ``mcpServers`` shape those
clients read.

Two properties matter more than convenience:

**Environment discipline.** A compiled config's ``env`` is normally the whole
parent process environment plus the host's overrides (``EnvPolicy`` defaults
to ``inherit_parent=True``). Serializing that verbatim into a file a user
copies, pastes, and commits would leak every secret in their shell. The
exporter therefore emits an explicit allowlist of behavioral variables the
first-party servers actually read, and renders credential-bearing values as
``${NAME}`` placeholders instead of the resolved secret.

**Honest reporting.** Anything the export cannot faithfully represent is
reported rather than silently dropped or silently widened: in-process and
download-only binaries with no spawnable command, HTTP servers (whose headers
carry credentials), and servers whose host-side tool policy hides tools that
the exported entry would expose.

The functions here are pure: they take resolved configs and an optional
command resolver, never read ``os.environ``, and perform no I/O.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from kernel.tools.mcp_client import MCPClientConfig

__all__ = [
    "EXPORT_FORMAT_MCPSERVERS",
    "MCPExport",
    "MCPServerExport",
    "export_mcp_servers",
    "render_mcp_export",
    "render_mcp_export_markdown",
]

#: The only export format implemented. Kept as a named constant so callers
#: validate rather than pass through arbitrary strings.
EXPORT_FORMAT_MCPSERVERS = "mcpServers"

#: Behavioral environment variables the first-party servers read.
#:
#: Everything else in a compiled config's ``env`` is inherited parent
#: environment, which must never reach an exported file. Adding a server-side
#: environment read means adding it here — an unlisted variable is a
#: mis-exported server, so this table is the review point for that change.
EXPORTABLE_ENV_KEYS: frozenset[str] = frozenset(
    {
        "LOG_LEVEL",
        "LISA_DEFAULT_MCP_TOOLS",
        "LISA_DISABLED_PACKS",
        "LISA_EMBEDDING_DIMENSION",
        "LISA_EMBEDDING_ENDPOINT",
        "LISA_EMBEDDING_MODEL",
        "LISA_EMBEDDING_TIMEOUT",
        "LISA_GUARD_BACKEND",
        "LISA_MCP_LOG_PATH",
        "LISA_OVERLAY_ROOT",
        "LISA_PACKS_VENV_SITE_PACKAGES",
        "LISA_SESSION_PROVIDER_SELECTIONS",
        "LISA_STORAGE_DENY_GLOBS",
        "LISA_THREAD_REDACTION_ENABLED",
        "LISA_THREAD_REDACTION_PATTERNS",
        "LISA_TIMEZONE",
    }
)

#: Exportable variables whose value is a credential. Rendered as a
#: ``${NAME}`` placeholder so the export never contains the secret; the
#: operator substitutes it in the target client's own secret mechanism.
PLACEHOLDER_ENV_KEYS: frozenset[str] = frozenset(
    {
        "LISA_EMBEDDING_API_KEY",
    }
)


@dataclass(frozen=True, slots=True)
class MCPServerExport:
    """One server rendered into a client config entry.

    Attributes:
        server_id: The pool's server id; used as the ``mcpServers`` key.
        transport: ``"stdio"`` or ``"http"``.
        command: Executable to spawn (stdio only).
        args: Arguments after ``command``, including the resolved
            ``--data-dir``, ``--readonly``, and manifest ``cli_args``.
        env: Behavioral environment only, with credentials placeholdered.
        cwd: Working directory when the compiled config pins one.
        url: Endpoint for HTTP servers.
        header_names: Header names the HTTP server needs. Values are never
            exported — they are credentials.
        capability: The capability this server implements, when it is the
            selected implementation (``memory``, ``storage``, …).
        published_tools: Model-facing tool ids the Lisa pool would publish
            (``<capability>.<name>``) for a capability server. Informational:
            an external client sees the server's native wire names instead.
        notes: Per-entry caveats the operator must read before use.
    """

    server_id: str
    transport: str
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str = ""
    header_names: tuple[str, ...] = ()
    capability: str | None = None
    published_tools: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_entry(self) -> dict[str, object]:
        """Return this export as one ``mcpServers`` entry.

        Keys that would be empty are omitted: an ``env: {}`` or a ``cwd: null``
        in the output suggests configuration that is not there. No comment or
        annotation key is emitted — several clients reject unknown fields, so
        caveats live on the :class:`MCPExport` and in the Markdown rendering
        instead of inside the document a client parses.
        """
        if self.transport == "http":
            return {"type": "http", "url": self.url}

        entry: dict[str, object] = {"command": self.command, "args": list(self.args)}
        if self.env:
            entry["env"] = dict(self.env)
        if self.cwd:
            entry["cwd"] = self.cwd
        return entry

    def to_shell_command(self) -> str:
        """Return a single-line shell invocation for this stdio server.

        Environment assignments are prefixed so the command is directly
        runnable. Only meaningful for ``transport == "stdio"``.
        """
        parts = [f"{key}={shlex.quote(value)}" for key, value in sorted(self.env.items())]
        parts.extend(shlex.quote(token) for token in (self.command, *self.args))
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class MCPExport:
    """The complete export result.

    Attributes:
        format: The client format this result renders as.
        data_dir: The resolved data directory the configs were compiled
            against, when known. Informational and used in renderings.
        servers: Exportable servers, sorted by ``server_id``.
        skipped: ``(server_id, reason)`` pairs for configured servers that
            this format cannot faithfully represent.
        warnings: Export-wide caveats (unsubstituted credentials, hidden
            tools, and so on).
    """

    format: str
    servers: tuple[MCPServerExport, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    data_dir: str | None = None

    @property
    def is_empty(self) -> bool:
        """True when nothing could be exported."""
        return not self.servers

    def to_payload(self) -> dict[str, object]:
        """Return the ``mcpServers`` document as a plain mapping."""
        return {"mcpServers": {server.server_id: server.to_entry() for server in self.servers}}


def _filtered_env(
    config: MCPClientConfig,
    *,
    warnings: list[str],
) -> dict[str, str]:
    """Return only exportable behavioral env vars, placeholdering credentials.

    A credential key whose resolved value differs from its placeholder is
    recorded in ``warnings`` so the operator knows the export needs
    substitution — a placeholder that silently became an empty string would
    look like working configuration.
    """
    env = config.env or {}
    exported: dict[str, str] = {}
    for key in sorted(EXPORTABLE_ENV_KEYS | PLACEHOLDER_ENV_KEYS):
        if key not in env:
            continue
        if key in PLACEHOLDER_ENV_KEYS:
            exported[key] = f"${{{key}}}"
            if env[key]:
                warnings.append(
                    f"{config.server_id}: '{key}' is a credential and was exported as "
                    f"${{{key}}}; set it in the target client's own secret store."
                )
            continue
        exported[key] = env[key]
    return exported


def _tool_policy_note(config: MCPClientConfig) -> str | None:
    """Describe a host-side tool policy the exported entry cannot carry.

    ``tool_policy`` is enforced by Lisa's pool when it builds the model-facing
    catalog. A client that spawns the server directly sees the server's full
    native surface, so a restrictive policy is *widened* by exporting. Say so
    instead of pretending the entry is equivalent.
    """
    policy = config.tool_policy
    allowed = getattr(policy, "allowed_tools", None)
    denied = tuple(getattr(policy, "denied_tools", ()) or ())
    if allowed is not None and not allowed and not denied:
        return (
            "host tool policy exposes none of this server's tools to Lisa's model; "
            "this entry exposes the server's full native surface"
        )
    if allowed is not None:
        return (
            "host tool policy is restricted to "
            f"{', '.join(sorted(allowed))}; this entry exposes the server's full "
            "native surface"
        )
    if denied:
        return (
            f"host tool policy hides {', '.join(sorted(denied))}; this entry exposes "
            "the server's full native surface"
        )
    return None


def _is_internal(config: MCPClientConfig) -> bool:
    """True when the host hides every tool of this server from its model.

    ``lisa-content`` is the canonical case (``allowed_tools: []``): the kernel
    calls it through its own sessions, and no model ever sees those tools.
    Such a server is Lisa's internal plumbing, not part of an interop surface,
    so the default export omits it — exporting it would hand an external agent
    a file-reading surface the host deliberately never exposed.
    """
    policy = config.tool_policy
    if policy is None:
        return False
    allowed = getattr(policy, "allowed_tools", None)
    return allowed is not None and not tuple(allowed)


def _published_tools(config: MCPClientConfig) -> tuple[str, ...]:
    """Return the model-facing tool ids the Lisa pool would publish.

    Capability servers publish ``<capability>.<published>``; the declared
    ``capability_tools`` / ``capability_optional_tools`` entries are already
    published names (``tool_map`` maps them *to* native wire names, so it is
    not applied here). Ordinary servers publish ``<server_id>.<native>``, and
    their native names are not known before discovery, so the result is empty
    for them.
    """
    capability = config.capability
    if not capability:
        return ()
    names = [*(config.capability_tools or ()), *(config.capability_optional_tools or ())]
    return tuple(sorted(f"{capability}.{name}" for name in names))


def _resolve_command(
    config: MCPClientConfig,
    command_resolver: Callable[[MCPClientConfig], str | None] | None,
) -> str:
    """Return a spawnable command for ``config``.

    Prefers the compiled command. Falls back to the resolver (used for
    in-process configs, which carry args but no subprocess command). The
    resolver receives the whole config so it can distinguish launch kinds
    instead of guessing a script name for every server.
    """
    if config.command:
        return config.command
    if command_resolver is not None:
        return command_resolver(config) or ""
    return ""


@dataclass(frozen=True, slots=True)
class _Skip:
    """One server that this format cannot represent, with the operator reason."""

    reason: str


@dataclass(frozen=True, slots=True)
class _Exported:
    """One exported server plus the export-wide warnings it produced."""

    server: MCPServerExport
    warnings: tuple[str, ...] = ()


def _export_http(config: MCPClientConfig) -> _Skip | _Exported:
    """Export one HTTP endpoint, or explain why it cannot be exported.

    The entry carries the URL and the header *names* only. Header values are
    credentials, so every entry produced here is accompanied by a warning that
    the operator must supply them in the target client.
    """
    if not config.url:
        return _Skip("HTTP server with no resolved url")
    header_names = tuple(sorted((config.headers or {}).keys()))
    notes = ["supply credential header values in the target client"]
    if header_names:
        notes.append(f"required headers: {', '.join(header_names)}")
    return _Exported(
        server=MCPServerExport(
            server_id=config.server_id,
            transport="http",
            url=config.url,
            header_names=header_names,
            notes=tuple(notes),
        ),
        warnings=(
            f"{config.server_id}: HTTP entry carries no header values; set "
            f"{', '.join(header_names) or 'credentials'} in the target client.",
        ),
    )


def _stdio_notes(config: MCPClientConfig, env: Mapping[str, str]) -> list[str]:
    """Collect the per-entry caveats a stdio export must carry.

    Each note is a way the exported entry differs from the server as Lisa runs
    it: a host tool policy that the export widens, a capability whose
    published tool ids an external client will not see, a pinned working
    directory some clients reject, and a log destination that defaults to the
    target project's cwd.
    """
    notes: list[str] = []
    policy_note = _tool_policy_note(config)
    if policy_note:
        notes.append(policy_note)
    if config.capability:
        notes.append(
            f"Lisa publishes this implementation as capability {config.capability!r}; "
            "an external client sees the native tool names"
        )
    if config.cwd is not None:
        notes.append(
            "this entry pins a working directory; clients that do not support "
            "'cwd' need it removed and the paths made absolute"
        )
    if "LISA_MCP_LOG_PATH" not in env:
        notes.append(
            f"no LISA_MCP_LOG_PATH is set, so the server writes "
            f"`<cwd>/{config.server_id}.jsonl`; set it to avoid logging into the "
            "target project"
        )
    return notes


def _export_stdio(
    config: MCPClientConfig,
    *,
    command_resolver: Callable[[MCPClientConfig], str | None] | None,
) -> _Skip | _Exported:
    """Export one subprocess server, or explain why it cannot be exported."""
    if config.binary_download is not None and not config.command:
        return _Skip(
            "binary download is resolved by Lisa's async startup; no command path "
            "is known at export time"
        )

    command = _resolve_command(config, command_resolver)
    if not command:
        if (config.runtime_kind or "") == "in_process":
            return _Skip(
                "in-process launch has no spawnable command; pass a command_resolver "
                "or disable in-process mode"
            )
        return _Skip("no spawnable command could be resolved")

    warnings: list[str] = []
    policy_note = _tool_policy_note(config)
    if policy_note:
        warnings.append(f"{config.server_id}: {policy_note}.")
    env = _filtered_env(config, warnings=warnings)

    return _Exported(
        server=MCPServerExport(
            server_id=config.server_id,
            transport="stdio",
            command=command,
            args=tuple(config.args),
            env=env,
            cwd=str(config.cwd) if config.cwd is not None else None,
            capability=config.capability,
            published_tools=_published_tools(config),
            notes=tuple(_stdio_notes(config, env)),
        ),
        warnings=tuple(warnings),
    )


def _export_one(
    config: MCPClientConfig,
    *,
    include_http: bool,
    include_internal: bool,
    command_resolver: Callable[[MCPClientConfig], str | None] | None,
) -> _Skip | _Exported:
    """Export one config, applying the selection policies in a fixed order."""
    if not include_internal and _is_internal(config):
        return _Skip(
            "the host exposes none of this server's tools to its own model "
            "(it is internal plumbing); pass include_internal to export it anyway"
        )

    is_http = bool(config.url) or (config.runtime_kind or "") == "http"
    if not is_http:
        return _export_stdio(config, command_resolver=command_resolver)
    if not include_http:
        return _Skip(
            "HTTP server: its credential headers are not exportable; re-run with "
            "include_http to emit the URL"
        )
    return _export_http(config)


def export_mcp_servers(
    configs: Iterable[MCPClientConfig],
    *,
    format: str = EXPORT_FORMAT_MCPSERVERS,
    server_ids: Sequence[str] | None = None,
    include_http: bool = False,
    include_internal: bool = False,
    command_resolver: Callable[[MCPClientConfig], str | None] | None = None,
    data_dir: str | None = None,
) -> MCPExport:
    """Project compiled MCP configs into an external client's server map.

    Args:
        configs: The host's compiled configs (warm and cold together).
        format: Output format. Only :data:`EXPORT_FORMAT_MCPSERVERS` is
            implemented; anything else raises ``ValueError`` rather than
            producing a shape the caller did not ask for.
        server_ids: Restrict the export to these ids. ``None`` exports every
            configured server. Unknown ids raise ``ValueError``.
        include_http: Include HTTP-transport servers. Default ``False``
            because their credentials are not exportable: the entry carries
            the URL and the header *names*, and the operator supplies values.
        include_internal: Include servers the host hides from its own model
            (empty ``tool_policy.allowed_tools``). Default ``False``: those are
            Lisa's internal plumbing, and exporting one exposes a surface the
            host chose not to expose.
        command_resolver: Optional ``config -> command`` fallback, used for
            configs that carry no subprocess command (in-process launch). It
            receives the whole config so it can key off the launch kind. The
            runtime passes one so an in-process host still exports working
            entries; a pure caller may omit it and get an explicit skip.
        data_dir: The resolved data directory, recorded on the result.

    Raises:
        ValueError: On an unsupported format or a ``server_ids`` entry that
            matches no configured server.

    Returns:
        An :class:`MCPExport` with entries sorted by ``server_id`` and every
        omission reported in ``skipped``.
    """
    if format != EXPORT_FORMAT_MCPSERVERS:
        raise ValueError(
            f"unsupported export format {format!r}; supported: {EXPORT_FORMAT_MCPSERVERS}"
        )

    ordered = sorted(configs, key=lambda config: config.server_id)
    if server_ids is not None:
        requested = set(server_ids)
        known = {config.server_id for config in ordered}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(
                f"unknown server id(s) {', '.join(unknown)}; configured: "
                f"{', '.join(sorted(known)) or 'none'}"
            )
        ordered = [config for config in ordered if config.server_id in requested]

    servers: list[MCPServerExport] = []
    skipped: list[tuple[str, str]] = []
    warnings: list[str] = []

    for config in ordered:
        entry = _export_one(
            config,
            include_http=include_http,
            include_internal=include_internal,
            command_resolver=command_resolver,
        )
        if isinstance(entry, _Skip):
            skipped.append((config.server_id, entry.reason))
            continue
        servers.append(entry.server)
        warnings.extend(entry.warnings)

    return MCPExport(
        format=format,
        servers=tuple(servers),
        skipped=tuple(skipped),
        warnings=tuple(warnings),
        data_dir=data_dir,
    )


def render_mcp_export(export: MCPExport) -> str:
    """Render the export as the JSON document a client config expects."""
    return json.dumps(export.to_payload(), indent=2, sort_keys=True)


def render_mcp_export_markdown(export: MCPExport) -> str:
    """Render the export as Markdown: the JSON block plus its caveats.

    Warnings and skips are part of the artifact. An operator who copies only
    the fenced JSON would otherwise ship an entry whose credential is a
    literal ``${...}`` string or whose tools were widened by the export.
    """
    lines: list[str] = ["## MCP export", ""]
    if export.data_dir:
        lines.extend([f"Data directory: `{export.data_dir}`", ""])

    if export.is_empty:
        lines.append("_No configured server can be exported in this format._")
    else:
        lines.extend(
            [
                "Paste this into the target client's MCP configuration file:",
                "",
                "```json",
                render_mcp_export(export),
                "```",
            ]
        )

    if export.warnings:
        lines.extend(["", "### Before you use this", ""])
        lines.extend(f"- {warning}" for warning in export.warnings)

    noted = [server for server in export.servers if server.notes]
    if noted:
        lines.extend(["", "### Per-server notes", ""])
        for server in noted:
            lines.append(f"- `{server.server_id}`")
            lines.extend(f"  - {note}" for note in server.notes)

    if export.skipped:
        lines.extend(["", "### Not exported", ""])
        lines.extend(f"- `{server_id}` — {reason}" for server_id, reason in export.skipped)

    stdio_entries = [server for server in export.servers if server.transport == "stdio"]
    if stdio_entries:
        lines.extend(["", "### Direct invocation", ""])
        lines.extend(
            [
                "Each stdio entry without a client in the middle:",
                "",
                "```sh",
                *(server.to_shell_command() for server in stdio_entries),
                "```",
            ]
        )

    return "\n".join(lines)
