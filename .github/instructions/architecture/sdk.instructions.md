---
name: SDK Architecture
applyTo: "sdk/**"
description: Architecture boundaries for the SDK package.
---

# SDK architecture

- The SDK package holds reusable content discovery, customization parsing, and repository-facing helpers.
- Keep SDK abstractions generic and dependency-light.
- The SDK may import from `kernel`, never from hosts. Avoid circular dependencies between SDK subpackages; promote shared contracts to the kernel when necessary.
- Public helpers must be import-safe: no registration or environment reads at module import time. Accept resolved configuration explicitly and use explicit re-exports.

## SDK and server boundary

- First-party MCP servers are transport facades. Put domain logic and protocols in the SDK; keep `FastMCP`, tool decorators, CLI entry points, and server-local backend state in `mcp_server`.
- File I/O must use the storage sandbox unless an integration has an explicit equivalent. Reject absolute paths and parent traversal; tool paths remain relative to the sandbox.
- Fetch policy is caller-owned, but SSRF protections and the absolute response-size ceiling are non-bypassable. Keep environment translation in host or tool adapters, not in the fetch primitive.
