# Public Boundary

This staged tree is the intended public framework boundary. It contains reusable mechanics for building Lisa-style hosts and packs while leaving the production Lisa app and CloudServers-specific intelligence private.

## Public Layers

| Layer | Public responsibility | Must not own |
| --- | --- | --- |
| `kernel` | Runtime-neutral contracts, mode catalog loading, workflow state, operation registry, tool and storage protocols | Host transport SDKs, production deployment, proprietary app behavior |
| `sdk` | Author-facing helpers for rendering, workshop execution, integration contracts, and pack composition | Private app imports, private model policy, production storage |
| `sdk/packs` | Bundled reference packs (checkpoint, core, introspect, memory, models) with modes, operations, authored content, and pack-registered slash commands. Packs may import `host` for transport-neutral runtime services (thread loading, model resolution, state provider, content loader). | Private product strategy, market-intel workflows, CloudServers-specific content |
| `../lisa-packs` (public repo: `github.com/Lisa-Framework/lisa-packs`) and `../lisa-packs-wip` (private repo: `github.com/Lisa-Framework/packs-wip`) | External integration packs under the `lisa_packs.*` namespace (public: brave, letta, markdown, mem0, okf, search; WIP: cloudflare, coding, github, gmail, stripe) with modes, operations, content, pack-registered slash commands, and all SDK extension kinds | Private app imports, kernel internals |
| `host` | Transport-neutral runtime: typed config (`HostConfig`), mutable state (`HostRuntime`), provider dispatch, MCP pool compilation, command dispatch, context assembly, thread persistence. No Rich, Typer, aiohttp, or transport imports. | Production chat transport, durable memory, production-grade model policy/fallback |
| `../lisa-hosts` (umbrella repo: `github.com/Lisa-Framework/lisa-hosts`) and `../lisa-hosts/cli` (submodule: `github.com/Lisa-Framework/lisa-host-cli`) | Terminal CLI host (Typer + Rich + prompt_toolkit). Wraps `host` with console rendering and REPL loop. Entry point: `lisa`. Distribution: `lisa-host-cli` (`uv tool install lisa-host-cli`); meta-package `lisa-host` (`uv tool install lisa-host`). | Production transport, multi-user routing |

## Host Responsibilities

A real host must inject concrete implementations for runtime seams:

- Model completion and model resolution callbacks.
- Tool catalog and tool execution callbacks.
- Blob/object storage implementation (via MCP storage server wiring; `lisa-storage` is the built-in).
- Memory store and recall implementation when memory operations are enabled.
- Browser automation implementation when surf operations are enabled.
- Transport adapter for chat, proactive delivery, and rich attachments.
- Scheduler or background worker for signal connector polling.

The framework keeps these as protocols and callbacks so package authors can build reusable behavior without depending on the production Lisa app.

The shared host runtime (`host`) wires a starter subset: `/search`, `/markdown`, `/discuss`, `/help`, default chat parsing, and locally added action/chat modes discovered through `SDK_CONTENT_ROOTS` when they only need the shipped callbacks. It supports live OpenAI-compatible completions through a small manifest contract. The terminal host (`../lisa-hosts/cli`, submodule) implements `TransportCallbacks` to render turns in its native surface — the runtime never imports Rich, Typer, aiohttp, or any transport-specific package. See `../lisa-hosts/docs/BOUNDARIES.md` for the host layer architecture.

## Privacy Boundary

The public contracts still model privacy explicitly:

- `SurfaceContext` carries `surface_type`, `surface_id`, and `memory_scope`.
- Memory writes and recalls must be scoped by the active surface and memory scope.
- Shared surfaces must not receive personal context unless a host implements explicit, audited sharing semantics.
- Proactive delivery targets are host-owned and must be explicit.

Prompt wording is not a privacy control. Hosts must enforce scope in storage keys, retrieval filters, and delivery routing.
