# Lisa Context Server (MCP)

A [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes Lisa's surface-scoped memory tools over stdio. Any MCP-compatible host (Claude Desktop, VS Code, custom agents) can use this server to store and retrieve memories with Lisa's privacy isolation model.

## Quick Start

From `lisa_framework/`:

```powershell
uv run lisa-context
```

Or install the package and run directly:

```powershell
uv pip install -e .\mcp_server
python -m lisa_context.server
```

## Configuration

| Env var | Default | Description |
| --- | --- | --- |
| `LISA_MCP_DB_PATH` | `.lisa_mcp_data/memory.db` | Path to the SQLite-backed memory store file |

Hosts that embed this server (e.g. Little Lisa via `kernel.mcp_client.MCPMemoryClient`) can inject `LISA_MCP_DB_PATH` through the subprocess `env` to pin storage to their own data-dir or memory-db without polluting the parent process environment. When the env var is unset, the server falls back to the relative default in its working directory.

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

## MCP Client Integration

The `kernel.mcp_client.MCPMemoryClient` class can connect to this server from any Lisa host process.

```python
from kernel.mcp_client import MCPMemoryClient

client = MCPMemoryClient(command="uv", args=["run", "lisa-context"])
result = await client.recall(query="deployment checklist", user_slug="alice")
```
