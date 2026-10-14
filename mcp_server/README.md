# MCP Servers

`lisa-harness-mcp-servers` (`mcp_server`) contains Lisa's first-party MCP (Model Context Protocol) servers — the external tool-facing integrations and transport concerns of the framework.

See [configuration](../docs/CONFIGURATION.md), [storage](../docs/STORAGE.md),
and the [documentation index](../docs/README.md).

## Design Philosophy

MCP servers own **actions and storage**: client-invoked tools, file access, search, and memory. Packs own workflow and personality; the kernel owns contracts; hosts own runtime wiring.

Key principles:

- **Thin protocol adapters**: `MCPServer` instances, tool decorators, `main()`, and server-local state live here. Domain logic stays in the SDK — most servers are thin facades over SDK backends.
- **Explicit CLI contract**: every server entry point uses `mcp_server.config.cli.build_parser(server_name)`, requires `--data-dir`, and is registered under `[project.scripts]` in this package's `pyproject.toml`. Data paths are never resolved from environment variables in server code.
- **No cross-imports**: servers may import from `kernel` and `sdk`, never from hosts. Server subpackages do not import each other; shared helpers live in `mcp_server.config`.

## The Servers

| Script | Module | Purpose |
| --- | --- | --- |
| `lisa-memory` | `mcp_server.memory.server` | Optional built-in memory adapter; domain storage and embeddings primitives live in the SDK. |
| `lisa-guard` | `mcp_server.guard.server` | Thin facade over `sdk.guard.GuardBackend` — input/output guard tools. |
| `lisa-content` | `mcp_server.content.server` | Authored-text file reading for the kernel. |
| `lisa-catalog` | `mcp_server.catalog.server` | Mode/operation/tool catalog and diagnostics. |
| `lisa-storage` | `mcp_server.storage.server` | Thin facade over `sdk.storage.StorageBackend` (includes model manifest read/write tools). |
| `lisa-thread` | `mcp_server.thread.server` | Thin facade over `sdk.thread.ThreadStore`. |
| — | `mcp_server.fetch.server` | Fetch-and-cache of remote http(s) content (not registered as a console script). |

## The CLI Contract

Every first-party server shares the same argument surface, built by `build_parser(server_name)`:

- `--data-dir` is **required** — the canonical data root; all storage paths derive relative to it. No env-var fallback, no hardcoded default.
- `--transport` defaults to `stdio` (only stdio is supported in v1).
- `--log-level` defaults to the `LOG_LEVEL` env var or `INFO`.
- `--workspace-root`, `--config`, and `--server-id` are optional.
- `--readonly` is added only when the server supports it.

Environment variables are for behavioral configuration only (API keys, feature flags, endpoints, log formatting) — never for path resolution.

## Shared Internals

| Module | Responsibility |
| --- | --- |
| `config/cli.py` | The shared argument parser, logging setup, data-dir resolution, and `run()` for all first-party servers. |
| `registry.py` | In-process MCP server registry. Maps first-party `server_id` strings to their `MCPServer` instance and `init` function so the kernel's `MCPToolPool` can connect in-process via the MCP SDK's in-memory transport — no subprocesses. Lazily imports server modules to keep `--version`/`verify` fast paths cheap. |
| `_entrypoint.py` | Shared stdio entrypoint shim: suppresses the benign `runpy` `RuntimeWarning` from console-script launchers and activates the packs venv site-packages. Imported side-effect-first by the servers that need it (`content`, `catalog`, `fetch`). |
| `_packs_venv_site.py` | Activates the packs venv site-packages in MCP server child processes. Reads `LISA_PACKS_VENV_SITE_PACKAGES` (set by the host) and calls `site.addsitedir()` so editable-install `.pth` meta-path finders map `lisa_packs.<name>` to source directories. |

## In-Process vs. Subprocess

First-party servers run in two modes:

- **In-process**: the host bridges `mcp_server.registry` into the kernel's `MCPToolPool` via `InProcessServerEntry`, giving stdio-semantics without subprocesses.
- **Subprocess**: the host spawns the console scripts (`lisa-content`, `lisa-catalog`, …) as child Python processes. Since `sys.path` modifications do not cross process boundaries, the host sets `LISA_PACKS_VENV_SITE_PACKAGES` and `_packs_venv_site` re-activates the packs venv inside the child.

## Testing

```bash
uv run pytest mcp_server/tests mcp_server/src/mcp_server/fetch/tests mcp_server/src/mcp_server/memory/tests mcp_server/src/mcp_server/thread/tests -m "not live" -q
```

Run from the workspace root; see [testing](../docs/TESTING.md) for full gates.

Architecture boundaries for this package are defined in [.github/instructions/architecture/mcp.instructions.md](../.github/instructions/architecture/mcp.instructions.md).
