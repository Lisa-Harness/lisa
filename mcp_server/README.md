# Lisa Context Server (MCP)

A [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes Lisa's surface-scoped memory tools over stdio. Any MCP-compatible host (Claude Desktop, VS Code, custom agents) can use this server to store and retrieve memories with Lisa's privacy isolation model.

## Quick Start

From `lisa_framework/`:

```powershell
uv run mcp-server
```

Or install the package and run directly:

```powershell
uv pip install -e .\mcp_server
python -m mcp_server.server
```

## Configuration

| Env var | Default | Description |
| --- | --- | --- |
| `LISA_MCP_DB_PATH` | `.lisa_mcp_data/memories.json` | Path to the JSON-backed memory store file |

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

Search for surface-scoped memories by keyword overlap.

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

The server uses `kernel.lite_memory.LiteMemoryStore` — a file-backed JSON store with keyword-overlap matching. It is intentionally minimal: no embeddings, no semantic search, no multi-tenant isolation beyond the key-based filtering, and no TTL. For production use, replace `LiteMemoryStore` with a host-owned memory implementation (e.g., Azure Blob-backed with vector search).

## MCP Client Integration

The `kernel.mcp_client.MCPMemoryClient` class can connect to this server from any Lisa host process. It implements the same `MemoryStore` protocol as `LiteMemoryStore`, so hosts can swap between local and MCP-backed memory without changing turn-runner code.

```python
from kernel.mcp_client import MCPMemoryClient

client = MCPMemoryClient(command="uv", args=["run", "mcp-server"])
result = await client.recall(query="deployment checklist", user_slug="alice")
```
