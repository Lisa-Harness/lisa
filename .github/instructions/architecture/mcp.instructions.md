---
name: MCP Server Architecture
applyTo: "mcp_server/**"
description: Architecture boundaries for the MCP server package.
---

# MCP server architecture

- The MCP server package owns external tool-facing integrations and transport concerns.
- Servers may import from `kernel` and `sdk`, never hosts. Server subpackages do not import each other; shared CLI helpers live in `mcp_server.config`.
- Every server entry point uses `mcp_server.config.cli.build_parser(server_name)`, requires `--data-dir`, and is registered in `mcp_server/pyproject.toml`.
- Do not resolve data paths from environment variables in server code; pass them through the CLI contract.
- Keep protocol adapters thin: `FastMCP`, tool decorators, `main()`, and server-local state stay here, while domain logic stays in the SDK.
- MCP-native ambient tools are declared through `default_available_tools`; include the resolved default tool IDs in catalog cache keys to prevent leakage between host configurations.
