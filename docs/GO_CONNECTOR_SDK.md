# Go Connector SDK — contract

This document defines the **language-neutral MCP server contract** that the Go
Connector SDK (`github.com/Lisa-Harness/lisa-go`) implements against. It is the
single source of truth both the Python Pack SDK / framework and the Go
Connector SDK bind to. It deliberately specifies *behavior and wire shape*, not
Python internals, so the contract stays stable as implementations move.

Audience: anyone writing or embedding a first-party MCP server that the Lisa
runtime spawns. This contract is what makes a server "first-party."

## Roles and the two SDKs

Lisa separates two audiences, each with its own SDK, both standing on this contract:

| SDK | Module | Audience | Role |
| --- | --- | --- | --- |
| **Python Pack SDK** | `lisa-harness-sdk` | pack authors | Author Lisa's extensibility (operations, modes, tools). Must be Python — the extension mechanism is Python entry points. |
| **Go Connector SDK** | `github.com/Lisa-Harness/lisa-go` | Go devs writing an MCP server or embedding Lisa as a host | Connect to / embed Lisa. Consumes this contract only; never reimplements the Python framework, never hosts packs. |

Both implement against the contract below. Neither owns it; this document owns it.

## Launch model

The runtime spawns first-party servers as **subprocesses over stdio** (the only
transport in v1). A server is declared in `mcp_servers.yaml` and compiled into an
`MCPClientConfig` by [kernel/mcp_launch/](../kernel/src/kernel/mcp_launch). Launch
kinds are enumerated in
[RuntimeKind](../kernel/src/kernel/mcp_launch/policy.py); the kind relevant to Go is:

```text
BINARY  — command_path points at a compiled server binary
```

The compiler resolves the binary path (with `${VAR}` env expansion), injects the
standard CLI args below, builds the subprocess env, and spawns it. **A Go server
needs no kernel change** — it is registered as a `binary`-runtime spec and spawned
exactly like any other first-party server. An empty `command_path` (e.g. an
optional binary that's not installed) degrades to "server absent" rather than
aborting startup.

### Compiled spawn shape

From [compiler.py](../kernel/src/kernel/mcp_launch/compiler.py)
(`compile_mcp_config`), the effective invocation is:

```text
<command_path> [data_dir_arg <resolved_data_dir>] \
               [workspace_root_arg <workspace_root>] \
               [--readonly] \
               [spec.cli_args...]
```

- `data_dir_arg` defaults to `--data-dir` for first-party servers; a server that
  manages its own storage sets it to `null`.
- `data_dir_subdir` namespaces the server's storage under the host data dir;
  `ensure_data_dir` pre-creates it before spawn.
- `workspace_root_arg` is injected only when the spec sets it.
- `--readonly` is appended only when the server declares support and it is set.
- `spec.cli_args` is appended last verbatim.

An optional `binary_download:` block (GitHub `repository`, `binary_name`,
`version`) pins a server to a release asset instead of a `${VAR}`-resolved
`command_path`. The runtime downloads the platform asset from the release,
verifies it against the release's `checksums.txt` (SHA-256), and caches it
under `<data_dir>/cache/mcp-binaries` before spawn; a pack omitting
`binary_download.version` reuses its own installed version's release.

## Server CLI contract

Every first-party server **must** accept the standard flag surface defined by
[mcp_server/config/cli.py](../mcp_server/src/mcp_server/config/cli.py):

| Flag | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--data-dir` | yes | — | Canonical data root; all storage paths resolve relative to it. No env-var fallback, no hardcoded default. |
| `--transport` | no | `stdio` | MCP transport; only `stdio` in v1. |
| `--log-level` | no | `$LOG_LEVEL` or `INFO` | One of `DEBUG, INFO, WARNING, ERROR, CRITICAL`. |
| `--workspace-root` | no | — | Logical workspace root for relative path resolution. |
| `--config` | no | — | Path to an optional JSON/YAML server config file. |
| `--server-id` | no | — | Explicit server identity override for diagnostics. |
| `--readonly` | no | off | Disable write-side operations (only when the server supports it). |

Path-like config (`--data-dir`, `--workspace-root`) travels in **CLI args**, never
env vars. Env vars are for behavioral configuration only (API keys, feature flags,
endpoints, log formatting).

## Logging contract

Server logs route through the shared stdlib infrastructure in
[sdk/logging_setup.py](../sdk/src/sdk/logging_setup.py) so every first-party
subprocess emits the same JSON format. The contract:

- **`LISA_MCP_LOG_PATH`** — injected by the host at launch; the per-server
  JSONL log path under the resolved data dir. This is the primary destination.
- **Fallback** — when `LISA_MCP_LOG_PATH` is unset (server launched outside the
  host pool), write JSONL to `<cwd>/<server_id>.jsonl`. **Never log to bare
  stderr by default**: the stdio SDK inherits the host terminal, which corrupts
  the TUI.
- **`LISA_LOG_STDERR=1`** — opt-in stderr mirror for debugging.

Go servers must honor all three: write JSONL to `LISA_MCP_LOG_PATH` when set,
fall back to `<cwd>/<server_id>.jsonl`, and keep stderr clean unless
`LISA_LOG_STDERR=1`.

## Tool surface

Servers expose MCP tools via the stdio JSON-RPC protocol. First-party
conventions:

- one `@tool` per operation; thin pass-through over the domain logic;
- tool annotations set `read_only_hint` / `idempotent_hint` appropriately;
- the **safety invariants of the underlying primitive are non-bypassable** and
  must be preserved verbatim by any port — for `lisa-fetch` that is the SSRF
  gate (scheme/host/private-IP) and the response-size ceiling
  (see [sdk/content/_fetcher.py](../sdk/src/sdk/content/_fetcher.py)); for
  `lisa-storage` the deny-glob files sandbox and `edit_file`'s exact-match and
  atomic-replace rules (see [storage](STORAGE.md#exact-edits)).

A Go port that weakens a safety invariant to simplify the port is a defect, not
a shortcut.

## What is *not* part of the contract

The following are Python implementation details and must **not** be depended on
by the Go SDK (they would calcify internals and fork the source of truth):

- the pydantic shape of `MCPServerSpec` / `MCPClientConfig` (bind to the YAML
  manifest schema and the compiled spawn shape above, not the Python classes);
- Python entry-point discovery (`lisa.extensions.*`) — that is the Pack SDK's
  mechanism, invisible to servers;
- any SDK-internal helper the server would have to import.

The Go SDK binds to: this YAML manifest schema, the CLI flag surface, the
logging env contract, the stdio MCP protocol, and the declared tool surface.

## Versioning

The Go Connector SDK declares the contract version it implements in its
`PROTOCOL.md`. Bumping this document's contract in a way that breaks an
existing Go server requires a coordinated note in
[CHANGELOG.md](../CHANGELOG.md).
