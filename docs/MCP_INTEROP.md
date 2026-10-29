# MCP interoperability

Lisa's first-party MCP servers are ordinary stdio MCP servers. A host that
does not run Lisa's turn loop can attach an external agent (Claude Desktop, VS
Code, Cursor, another harness) directly to them, and a Lisa host can hand its
own configuration to such an agent. This page describes what that deployment
gets, what it does not, and how to produce the config.

The servers own *actions and storage*: memory, a file sandbox, fetch, guard,
and thread reads. They do not own modes, stage transitions, approvals, effect
receipts, or context fitting. Those live in the kernel and runtime turn loop
and are absent here.

## Scenario: Lisa as the data plane for another agent

Install `lisa-harness-mcp-servers` in the environment where the agent runs,
then point the agent's MCP configuration at the servers it should use. The
compiled form of that configuration is produced by the export described below;
the minimal hand-written version is:

```json
{
  "mcpServers": {
    "lisa-memory": {
      "command": "lisa-memory",
      "args": ["--data-dir", "/home/u/.lisa"]
    },
    "lisa-storage": {
      "command": "lisa-storage",
      "args": ["--data-dir", "/home/u/.lisa", "--readonly"]
    },
    "lisa-fetch": {
      "command": "lisa-fetch",
      "args": ["--data-dir", "/home/u/.lisa"]
    }
  }
}
```

Each entry is spawned per the client's own lifecycle. Everything the servers
persist lands under the resolved `--data-dir` using the
[canonical categories](CONFIGURATION.md#data-directory-and-profiles).

## Which servers to expose

| Server | Expose | Notes |
| --- | --- | --- |
| `lisa-memory` | Yes, when the agent should share learned context | `--readonly` serves recall without allowing writes. Recall skips its access-tracking update in that mode rather than failing. |
| `lisa-storage` | Yes | `--readonly` is enforced. Add `--deny-glob` for host configuration, `.env`, and credentials. Traversal outside the sandbox is rejected. |
| `lisa-fetch` | Yes | Private-IP/link-local SSRF blocks and an absolute `max_bytes` ceiling apply by default. |
| `lisa-thread` | Only when a Lisa host writes the threads | This server is read-only. Against a store no Lisa runtime writes, every read returns an empty result. |
| `lisa-catalog` | Rarely | It exposes Lisa's own pack catalog. Useful to an agent that intends to run Lisa packs; requires `SDK_CONTENT_ROOTS`. |
| `lisa-content` | No | The runtime declares `tool_policy.allowed_tools: []` for it; its file tools are internal to the kernel's own sessions. |
| `lisa-guard` | No | Only the `passthrough` backend exists, and an unknown backend name also falls through to `passthrough`. Exposing it advertises a control that is not there. |

`/mcp export` applies these defaults: it omits servers whose host tool policy
hides all their tools (`lisa-content`) and omits HTTP endpoints unless asked.
Both omissions are reported with the reason.

## Identity is a host responsibility

Tool arguments such as `user_slug`, `workspace_id`, `surface_id`, and
`memory_scope` arrive from the caller. Inside Lisa, `proxy_policy` strips them
from the model-facing schema and injects them from a host resolver, so the
model can neither see nor set them. An external client has no such proxy: the
scope fields are plain arguments the agent supplies.

The row filters are enforced — a request scoped to one user cannot read
another user's records — but *which* identity is presented is not authenticated
by the servers. Treat these arguments as labels, not as a security boundary:

| Topology | Isolation | Requires |
| --- | --- | --- |
| One data directory per principal, agent running as that principal | Filesystem/OS | Nothing beyond correct deployment |
| Shared data directory behind your own injecting proxy | As strong as the injection | Implementing strip-and-inject outside Lisa |
| Shared data directory with no proxy | None | Do not use for more than one principal |

`lisa-storage`'s sandbox is a path boundary, not a user boundary: two clients
sharing one data directory share one `files/` tree.

## Producing the configuration

Two equivalent surfaces:

- **Inside a Lisa host:** `/mcp export` renders the compiled configuration as
  Markdown containing an `mcpServers` document, its warnings, the servers it
  omitted, and one shell invocation per stdio entry.
- **Programmatically:** `sdk.mcp_host.export.export_mcp_servers(configs)` for a
  pure projection, or `runtime.mcp_export.export_runtime_mcp_servers(runtime)`
  to read a runtime's compiled configs. Neither starts the MCP pool, so an
  export works before startup and after shutdown.

`/mcp export` flags:

| Flag | Effect |
| --- | --- |
| `--server <id>` | Restrict the export (repeatable). Unknown ids are an error, not an empty result. |
| `--include-http` | Also emit HTTP endpoints. The entry carries the URL and the required header *names*; values are never exported. |
| `--include-internal` | Also emit servers the host hides from its own model. |

Only the `mcpServers` shape is implemented; any other requested format raises
rather than producing a shape the caller did not ask for.

### What the export will not do

The export is a projection of resolved configuration, so it is careful about
three things:

- **Environment.** A compiled config's `env` is normally the whole parent
  process environment plus host overrides. Only an allowlist of behavioral
  variables the servers actually read is exported; credential-valued
  variables become `${NAME}` placeholders and are called out as needing
  substitution. `PATH`, `HOME`, cloud credentials, and tokens are never
  included.
- **Width.** `tool_policy` is enforced by Lisa's pool when it builds the
  model-facing catalog. A client that spawns a server directly sees the
  server's full native surface, so a restrictive policy is widened by
  exporting. The export says so per server instead of implying equivalence.
- **Fidelity.** A server it cannot represent is reported with a reason: an
  in-process launch whose console script is not installed, a binary whose
  release path Lisa resolves asynchronously, an HTTP endpoint without a
  resolved URL.

### Naming

Inside Lisa, tools are advertised as `<server_id>.<tool>`, or as
`<capability>.<tool>` (`memory.add`, `storage.read_file`, `fetch.fetch_url`,
`thread.thread_read_range`, …) for the selected implementation of a capability.
An external client sees the server's native wire names (`add_memory`,
`read_file`, …). Prompts and instructions written against Lisa's published ids
do not transfer verbatim; the export notes which capability each server
implements so the mapping is explicit.

## Operational notes

- **Transport.** `--transport` accepts only `stdio`; every server runs as a
  child process of whichever client spawns it.
- **Logging.** Set `LISA_MCP_LOG_PATH` per server. When it is unset the server
  writes `<cwd>/<server_id>.jsonl`, which in an external client means the
  target project's directory. The export emits the host's resolved log paths
  and warns when a server has none.
- **Read-only coverage.** `lisa-storage` and `lisa-memory` enforce
  `--readonly`. The other servers accept no such flag because they are
  read-only by design (`catalog`, `content`, `fetch` reads cached content,
  `thread`) or stateless (`guard`).
- **Memory backups.** The framework does not dump memory databases. Point-in-
  time copies belong to the deployment's own SQLite-aware backup process; see
  [storage and data handling](STORAGE.md).
