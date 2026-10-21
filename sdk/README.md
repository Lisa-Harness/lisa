# SDK

`lisa-harness-sdk` (`sdk`) is the reusable content-discovery, customization-parsing, and repository-facing helper layer of the Lisa Harness. It sits between the runtime-neutral `kernel` and the concrete runtime that packs compose: the kernel owns contracts and the turn loop, the SDK owns the catalog scanners, model router, provider gateways, and the memory/thread/guard/storage/fetch primitives that first-party MCP servers expose.

## Design Philosophy

The SDK holds **domain logic for everything the first-party MCP servers expose**, plus the pluggable runtime backends (providers, routers) that hosts discover through entry points. `mcp_server` stays a thin transport facade over these modules.

Key principles:

- **Generic and dependency-light**: SDK abstractions must work for any runtime. The SDK may import from `kernel`, never from `runtime`.
- **Import-safe public helpers**: no registration, environment reads, or I/O at module import time. Resolved configuration is accepted explicitly.
- **Server boundary**: file I/O goes through the storage sandbox (relative paths only, deny-glob policy); SSRF protections and the response-size ceiling in the fetch primitive are non-bypassable. Environment translation belongs in host or tool adapters, not in the primitives.
- **Namespace package**: `sdk` is a PEP 420 namespace shared with the bundled packs under `sdk/packs/`, which ship their own wheels. The `sdk` content root is exported from [`src/sdk/_content_root.py`](src/sdk/_content_root.py).

See [architecture](../docs/BOUNDARIES.md), [memory](../docs/MEMORY_ARCHITECTURE.md),
[storage](../docs/STORAGE.md), and the [documentation index](../docs/README.md)
for cross-package contracts.

## Module Map

### Catalog, Content & Context

| Module | Responsibility |
| --- | --- |
| `content/` | Content catalog and authored-text reading: mode/operation/tool catalog JSON builders, content-root resolution, `ContentReader`, path safety, Copilot customization discovery/parsing (applyTo globbing). Also owns the fetch primitive (`ContentFetcher`): L1/L2 caching, conditional GET, a shared cookie jar, the non-bypassable SSRF gate (re-applied on every redirect hop), and the navigation headers that keep bot-defended pages reachable. Powers the `lisa-content`, `lisa-catalog`, and `lisa-fetch` MCP servers. |
| `context/` | Entry-point-aware extension item building: discovers `lisa.extensions.context_items` entry points and feeds extension evidence into the session-owned context mapper via `ContextSeams.extension_item_builder`. |
| `config.py` | Public runtime `Settings` (pydantic-settings) for packs/extensions — deliberately narrower than root app config, with owned-source filtering so packs only consume env keys they declare. |
| `pack_settings.py` | `OwnedSettingsSource` / `build_owned_settings_sources`; importable by external packs without depending on `sdk_packs`. |
| `layout.py` | Single source of truth for the canonical data-dir categories (`config/`, `state/`, `data/`, `cache/`, `logs/`, `run/`) shared by hosts, SDK, MCP servers, and packs. |

### State & Memory Primitives

| Module | Responsibility |
| --- | --- |
| `memory/` | SQLite-backed memory store library (vec0 embeddings, backups). The `lisa-memory` MCP server wraps it. |
| `thread/` | Lossless, segmented conversation-thread store (`ThreadStore`): three-tier layout (active/sealed segments + sparse index) with zstd sidecars and redaction policy. Owns the read path; the write path stays in `runtime.thread`. |
| `storage/` | Sandboxed file I/O: `StorageBackend` owns the files sandbox, deny-glob policy, and model-manifest read/write. The `lisa-storage` MCP server wraps it. |
| `workproducts.py` | Typed, pack-neutral `WorkProduct` contract plus `WorkProductStore` protocol — durable, approvable units of work shared across packs (cross-pack imports are forbidden, so the contract sits on neutral SDK ground). The host injects the file-based implementation. |

### Safety & Fetch

| Module | Responsibility |
| --- | --- |
| `guard/` | Content-safety validation backends: `GuardBackend` protocol, default `PassthroughGuardBackend`, result formatting, blocked-check audit logging. The `lisa-guard` MCP server owns transport. |
| `mcp_host/` | Pure config construction for first-party MCP servers (`build_content_mcp_config`, `build_catalog_mcp_config`). Pool lifecycle is deliberately host-owned because anyio cancel scopes bind to the host's event loop. |

### Pluggable Runtime Backends

| Module | Responsibility |
| --- | --- |
| `providers/` | Reference `ProviderGatewayProvider` implementations for OpenAI, Azure OpenAI, Anthropic, Google Gemini, and AWS Bedrock, registered via `lisa.extensions.provider_gateways`. Explicitly not a pack — a pluggable runtime backend. |
| `router/` | The built-in `lisa-router`: a pure-Python CEL/rules engine that selects a model per turn-loop iteration from the manifest's `router.rules` block with zero model calls, registered via `lisa.extensions.model_routers`. See [`src/sdk/router/README.md`](src/sdk/router/README.md). |

### Shared Infrastructure

| Module | Responsibility |
| --- | --- |
| `logging_setup.py` | Shared stdlib-logging infrastructure for all hosts and first-party MCP servers: `JSONFormatter`, `RateLimitFilter`, `configure_logging`, `emit_structured_event`. Stdlib-only so subprocesses need not import the kernel. |
| `workspace.py` | Single shared `resolve_workspace_root` with explicit strictness (`require_git` knob) so packs no longer derive the user's project root independently. |

## Bundled Packs

The `sdk/packs/` subtree contains the bundled reference packs — `core`, `memory`, `introspect`, `recovery` — which ship as their own distributions and are excluded from the `lisa-harness-sdk` wheel. For pack structure, mode front matter, operations, signal connectors, and the testing workflow, see:

- [docs/PACK_AUTHORING_GUIDE.md](../docs/PACK_AUTHORING_GUIDE.md) — pack anatomy, mode files, extension entry points, and the hello-world walkthrough.
- [AUTHORING.md](AUTHORING.md) — the operation authoring deep-dive (runtime vs. declarative, metadata fields, the `execute(ctx)` contract).
- The [lisa-packs](https://github.com/Lisa-Harness/lisa-packs) repo — the public external packs, their installation, and the per-pack submodule workflow for authoring real integration packs.

## Testing

```bash
uv run pytest sdk/tests sdk/src/sdk/context/tests sdk/src/sdk/guard/tests sdk/src/sdk/memory/tests sdk/src/sdk/storage/tests -m "not live" -q
```

Run from the workspace root; this selects SDK tests, not bundled pack suites.
See [testing](../docs/TESTING.md) for the complete CI selection.

Markers: `live` (hits real external signal or provider APIs), `integration` (multi-module in-process; may require Azurite on localhost:10000), `eval` (deterministic agent regression evals — no model calls, no I/O). Pack validation itself belongs to the `packstest` harness; see [`../packstest/README.md`](../packstest/README.md).
