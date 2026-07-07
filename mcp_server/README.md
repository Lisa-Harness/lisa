# Lisa MCP Servers

A family of [Model Context Protocol](https://modelcontextprotocol.io/) servers that expose Lisa kernel capabilities over stdio. Any MCP-compatible host (Claude Desktop, VS Code, custom agents) can use these servers to access memory, operations, search, surf, guard, storage, and content tools.

## Built-in Servers

| Server | Package | Tools | Description |
| --- | --- | --- | --- |
| `lisa-context` | `lisa_context` | `add_memory`, `search_memory`, `list_memories`, `get_surface_context` | Surface-scoped memory store (SQLite) |
| `lisa-operations` | `lisa_operations` | one tool per registered operation | Operation registry dispatch |
| `lisa-search` | `lisa_search` | `web_search` | Brave Web Search wrapper |
| `lisa-surf` | `lisa_surf` | `surf_url`, `flush_session` | Browser automation |
| `lisa-guard` | `lisa_guard` | `validate_input`, `validate_output`, `guard_configured` | Input/output guardrails (passthrough or configurable backend) |
| `lisa-storage` | `lisa_storage` | 23 BlobIO tools | Filesystem-backed blob storage with path-safe resolution |
| `lisa-content` | `lisa_content` | `read_authored_text`, `list_content_roots`, `invalidate_cache` | Authored content registry with LRU cache |

## CLI Contract

Every first-party server shares the same argparse CLI:

```plaintext
usage: server.py [-h] --data-dir DATA_DIR [--transport {stdio}] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--readonly]
```

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--data-dir` | yes | — | Root data directory for server state (SQLite DB, storage root, etc.) |
| `--transport` | no | `stdio` | MCP transport (only `stdio` supported) |
| `--log-level` | no | `INFO` (or `LOG_LEVEL` env) | Logging level |
| `--readonly` | no | off | Disable write operations (supported by `lisa-context`, `lisa-storage`) |
| `--backend` | no | `passthrough` | Guard backend (`lisa-guard` only) |

Servers exit with code 2 if `--data-dir` is missing. There is no env-var fallback for the data directory — the host must always pass it explicitly.

## Quick Start

From `lisa_framework/`:

```powershell
uv run lisa-context --data-dir ./.lisa_data
```

Or install the package and run directly:

```powershell
uv pip install -e .\mcp_server
python -m lisa_context.server --data-dir ./.lisa_data
```

## Tools

### `add_memory`

Store a new memory snippet scoped to a user identity and surface.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `user_slug` | `str` | required | User identity slug |
| `content` | `str` | required | Memory content to store |
| `memory_scope` | `str` | `"personal"` | Scope: `personal`, `team`, or `workspace` |
| `workspace_id` | `str` | `"default"` | Workspace identifier |
| `surface_id` | `str` | `""` | Surface identifier (Teams channel, chat thread, etc.) |

### `search_memory`

Search for surface-scoped memories by keyword recall.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `user_slug` | `str` | required | User identity slug |
| `query` | `str` | required | Search query |
| `memory_scope` | `str` | `"personal"` | Scope filter |
| `workspace_id` | `str` | `"default"` | Workspace filter |
| `surface_id` | `str` | `""` | Surface filter |
| `limit` | `int` | `5` | Max results |

### `get_surface_context`

Retrieve the most recent memories for a user and surface context.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `user_slug` | `str` | required | User identity slug |
| `memory_scope` | `str` | `"personal"` | Scope filter |
| `workspace_id` | `str` | `"default"` | Workspace filter |
| `surface_id` | `str` | `""` | Surface filter |
| `limit` | `int` | `20` | Max records |

### `list_memories`

Return a raw JSON array of memory records for a user and surface.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `user_slug` | `str` | required | User identity slug |
| `memory_scope` | `str` | `"personal"` | Scope filter |
| `workspace_id` | `str` | `"default"` | Workspace filter |
| `surface_id` | `str` | `""` | Surface filter |
| `limit` | `int` | `20` | Max records |

## Privacy Model

All tools enforce Lisa's surface-scoped privacy isolation:

- Memories are keyed by `user_slug`, `workspace_id`, `memory_scope`, and `surface_id`.
- A query for one surface cannot return records from another surface or user.
- Shared surfaces must be explicitly configured by the host; this server does not implement cross-surface sharing on its own.

## Architecture

The `lisa-context` MCP server owns a SQLite-backed memory store and exposes memory tools over MCP. It is the default and only memory implementation; the kernel is a pure client.

## Launch Contract

First-party servers are launched through the MCP launch contract (`kernel/mcp_launch/`):

1. **Registry**: `MCPServerRegistry` holds approved `MCPServerSpec` entries. Only registered specs compile to `MCPClientConfig`.
2. **Compiler**: `compile_registry()` injects `--data-dir` and `--workspace-root` into each spec's CLI args, producing `MCPClientConfig` entries for `MCPToolPool`.
3. **Tool filtering**: `filter_tools()` applies allow/deny lists per server.

See `docs/MCP_LAUNCH_CONTRACT_SPEC.md` for the full specification.

## MCP Client Integration

The `kernel.mcp_client.MCPMemoryClient` class can connect to this server from any Lisa host process.

```python
from kernel.mcp_client import MCPMemoryClient

client = MCPMemoryClient(command="uv", args=["run", "lisa-context", "--data-dir", "./.lisa_data"])
result = await client.recall(query="deployment checklist", user_slug="alice")
```
