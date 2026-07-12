# Public Boundary

This staged tree is the intended public framework boundary. It contains reusable mechanics for building Lisa-style hosts and packs while leaving the production Lisa app and CloudServers-specific intelligence private.

## Public Layers

| Layer | Public responsibility | Must not own |
| --- | --- | --- |
| `kernel` | Runtime-neutral contracts, mode catalog loading, workflow state, operation registry, tool and storage protocols | Host transport SDKs, production deployment, proprietary app behavior |
| `sdk` | Author-facing helpers for rendering, workshop execution, integration contracts, and pack composition | Private app imports, private model policy, production storage |
| `sdk/packs` | Core reference pack (modes, operations, authored content) | Private product strategy, market-intel workflows, CloudServers-specific content |
| `lisa_pack_*` | External integration packs (cloudflare, github, stripe) with modes, operations, content, and all SDK extension kinds | Private app imports, kernel internals |
| `hosts/little_lisa` | Local terminal host for package discovery, command routing, and a narrow operation-shape test with manifest-driven model routing | Production chat transport, durable memory, production-grade model policy/fallback, workflow-stage execution |

## Private Layers

The repository root still owns the private app:

- Teams transport and rendering adapter.
- Production provider and model routing policy.
- Azure Container Apps deployment and Key Vault wiring.
- Production blob layout and app-owned event handlers.
- CloudServers operations extensions and packs.
- Market-intel collectors, synthesis, and proprietary prompt assets.
- Advanced private proactive and reflective loops.

Those private layers may consume the public framework, but the public framework must not import them.

## Host Responsibilities

A real host must inject concrete implementations for runtime seams:

- Model completion and model resolution callbacks.
- Tool catalog and tool execution callbacks.
- Blob/object storage implementation.
- Memory store and recall implementation when memory operations are enabled.
- Browser automation implementation when surf operations are enabled.
- Transport adapter for chat, proactive delivery, and rich attachments.
- Scheduler or background worker for signal connector polling.

The framework keeps these as protocols and callbacks so package authors can build reusable behavior without depending on the production Lisa app.

Little Lisa intentionally wires only a starter subset: `/search`, `/markdown`, `/discuss`, `/help`, `/deep`, default chat parsing, and locally added action/chat modes discovered through `SDK_CONTENT_ROOTS` when they only need the shipped callbacks. It supports live OpenAI-compatible completions through a small manifest contract, but public workflow, browser, image, memory, or storage-heavy modes still report the host capability they require.

## Privacy Boundary

The public contracts still model privacy explicitly:

- `SurfaceContext` carries `surface_type`, `surface_id`, and `memory_scope`.
- Memory writes and recalls must be scoped by the active surface and memory scope.
- Shared surfaces must not receive personal context unless a host implements explicit, audited sharing semantics.
- Proactive delivery targets are host-owned and must be explicit.

Prompt wording is not a privacy control. Hosts must enforce scope in storage keys, retrieval filters, and delivery routing.

## Exclusion Rule

Do not add private app logic to this tree to make an example work. Prefer a small protocol, callback, stub, or host-owned adapter. If a feature requires CloudServers data, market-intel assets, or production Lisa state, keep it outside the public framework and document the missing host capability.

