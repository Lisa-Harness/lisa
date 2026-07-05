# MCP-Native Kernel Design

Status: Proposed 2026-07-04
Owner: Architecture

This document specifies the next hard cutover of the Lisa kernel: MCP becomes the
canonical tool protocol, the `kernel/context/` boundary breach is fixed by inverting
the missing seams, and memory becomes a discoverable external capability with a
built-in fallback. OpenAI and Anthropic tool schemas become provider-edge translations,
not first-class kernel concepts.

## Design Verdicts

| Question | Answer |
| --- | --- |
| MCP as the universal lowest layer? | **No.** MCP is a wire protocol, not an in-process interface. Use MCP at the edges (tool dispatch, external memory, kernel-as-server), typed Protocols in the middle (memory, state, blob, browser). |
| MCP as the canonical *tool* layer? | **Yes.** The kernel speaks MCP natively for tool discovery and dispatch. OpenAI/Anthropic schemas are provider-edge translations. |
| Move app context/memory into the kernel? | **Yes, by inversion.** Define the missing seams (`ContentRegistry`, `ModelPolicy`, `ProviderGateway`, `EventBus`, `DataSanitizer`) and ship public default implementations. The app becomes one provider of those seams, not a dependency. |
| Discoverable memory systems? | **Yes.** Entry-point group `lisa.memory.providers` yields MCP server configs or in-process `MemoryStore` implementations. Probe at startup; fall back to the built-in SQLite store. |

## Layered Architecture

```mermaid
flowchart TD
    subgraph Host[Host App — Little Lisa or production]
        Transport[Transport adapter]
        HostHooks[TurnHooks wiring]
    end

    subgraph Kernel[lisa-kernel]
        TR[turn_runner]
        FD[frontdoor + ipsar_fsm]
        OPS[operations registry]
        TOOLS[tool layer — MCP native]
        SEAMS[seams/protocols]
        CTX[context broker — public default]
        MEM[memory recall — public default]
        SLICE[slice — pressure model]
    end

    subgraph MCPClient[MCP Client Pool]
        TC[Tool Client]
        MC[Memory Client]
        SC[Search Client]
        BC[Browser Client]
    end

    subgraph ExternalMCP[External MCP Servers]
        Mem0[mem0 MCP server]
        RefMem[@modelcontextprotocol/server-memory]
        Brave[Brave Search MCP]
        FSFilesystem[Filesystem MCP]
        GitMCP[Git MCP]
        LisaCtx[lisa-context MCP server — kernel-owned]
    end

    subgraph Providers[Provider Gateways]
        OpenAI[OpenAI / Azure OpenAI]
        Anthropic[Anthropic — deferred]
        Ollama[Ollama]
    end

    Transport --> HostHooks
    HostHooks --> TR
    TR --> FD
    TR --> OPS
    TR --> TOOLS
    TR --> CTX
    TR --> MEM
    TR --> SLICE
    TOOLS --> TC
    MEM --> MC
    CTX --> SEAMS
    SEAMS --> HostHooks
    TC --> ExternalMCP
    MC --> ExternalMCP
    TR --> Providers
    TOOLS -.translates.-> Providers
    LisaCtx -.exposes.-> MC
```

The rule: **MCP crosses a process boundary; Protocols cross a module boundary.**
If two pieces live in the same process, they talk through a `Protocol`. If one piece
is an external server (memory service, search service, browser service, third-party
tool), the kernel speaks MCP to it.

## Part 1 — Fix the `kernel/context/` boundary breach

Eight modules under `kernel/context/` import from `app.*`. Per BOUNDARIES.md, the
public kernel must not import private app modules. The fix is **inversion**, not
relocation: define the missing seams, ship public default implementations, and let
the app override through `TurnHooks`.

### New seams in `kernel/seams/protocols.py`

```python
@runtime_checkable
class ContentRegistry(Protocol):
    """Load authored text assets from merged content roots."""

    def read_authored_text(self, relative_path: str) -> str: ...


@runtime_checkable
class ModelPolicy(Protocol):
    """Resolve model aliases and availability without coupling to a provider."""

    def resolve_alias(self, alias: str) -> str: ...
    def is_available(self, model_id: str) -> bool: ...


@runtime_checkable
class ProviderGatewayProvider(Protocol):
    """Return a ProviderGateway for a given provider name."""

    def gateway_for(self, provider: str) -> "ProviderGateway | None": ...


@runtime_checkable
class EventBus(Protocol):
    """Lightweight pub/sub for salience events and consolidation triggers."""

    async def publish(self, topic: str, payload: dict[str, Any]) -> None: ...
    def subscribe(self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]) -> None: ...


@runtime_checkable
class DataSanitizer(Protocol):
    """Apply redaction before memory writes or context assembly."""

    def redact(self, text: str, *, scope: str) -> str: ...
```

### Public default implementations

Ship these in the kernel so third-party hosts get working context/memory without the
private app:

| Module | Default impl | Backed by |
| --- | --- | --- |
| `kernel/context/content_registry.py` | `DiskContentRegistry` | `content_roots.discover_*` + filesystem |
| `kernel/context/provider_gateway_provider.py` | `KernelProviderGatewayProvider` | `kernel.provider_core` gateways |
| `kernel/context/event_bus.py` | `InProcessEventBus` | `asyncio` queues, no external deps |
| `kernel/context/data_sanitizer.py` | `PassThroughSanitizer` | identity, no redaction |
| `kernel/context/model_policy.py` | `ManifestModelPolicy` | `kernel.model_manifest` |

### What moves where

| Current `kernel/context/` module | App dependency | New seam | Public default |
| --- | --- | --- | --- |
| `context_broker.py` | `app.config`, `app.context_frames`, `app.extension_context_evidence` | `ContentRegistry` + `settings` (kernel-owned) | `DiskContentRegistry` |
| `context_budget.py` | `app.config` | `settings` (kernel-owned — already exists) | n/a |
| `context_frames.py` | `app.content_registry` | `ContentRegistry` | `DiskContentRegistry` |
| `conversation_digest.py` | `app.model_policy`, `app.providers`, `app.config`, `app.content_registry` | `ModelPolicy` + `ProviderGatewayProvider` + `ContentRegistry` | `ManifestModelPolicy` + `KernelProviderGatewayProvider` + `DiskContentRegistry` |
| `memory_consolidation.py` | `app.config`, `app.memory_store` | `settings` + `MemoryStore` (already a seam) | n/a |
| `memory_recall.py` | `app.model_policy`, `app.providers`, `app.config`, `app.content_registry`, `app.memory_store` | `ModelPolicy` + `ProviderGatewayProvider` + `ContentRegistry` + `MemoryStore` | all four defaults |
| `memory_store.py` | `app.config`, `app.blob_client`, `app.providers`, `app.data_sanitizer`, `app.event_bus` | `settings` + `BlobIO` + `ProviderGatewayProvider` + `DataSanitizer` + `EventBus` | `InProcessEventBus` + `PassThroughSanitizer` |
| `passive_intelligence.py` | `app.blob_client`, `app.config`, `app.memory_store`, `app.thread_store` | `BlobIO` + `settings` + `MemoryStore` + `ThreadStore` | n/a |

### Wiring through `TurnHooks`

Add a `context_seams` bundle to `TurnHooks` so hosts inject the five seams in one
place. Defaults wire the public implementations; the production app overrides with its
own (`AppContentRegistry`, `AppModelPolicy`, etc.) — no shim, no dual path.

```python
@dataclass
class ContextSeams:
    content_registry: ContentRegistry = field(default_factory=DiskContentRegistry)
    model_policy: ModelPolicy = field(default_factory=ManifestModelPolicy)
    provider_gateways: ProviderGatewayProvider = field(default_factory=KernelProviderGatewayProvider)
    event_bus: EventBus = field(default_factory=InProcessEventBus)
    data_sanitizer: DataSanitizer = field(default_factory=PassThroughSanitizer)
```

`TurnHooks` gains one field: `context_seams: ContextSeams = field(default_factory=ContextSeams)`.

### Import-boundary enforcement

The existing `lisa_sdk/tests/test_import_boundaries.py` pattern extends to
`kernel/tests/test_context_import_boundaries.py`: assert that no module under
`kernel/context/` imports `app.*` or any private package. This is the regression
guard for the breach that exists today.

## Part 2 — MCP as the canonical tool layer

### The split

| Concern | Owner | Transport |
| --- | --- | --- |
| Tool discovery (`list_tools`) | Kernel MCP client | MCP |
| Tool dispatch (`call_tool`) | Kernel MCP client | MCP |
| Tool schema for the provider | Kernel translator | MCP schema → OpenAI/Anthropic |
| Operation execution | `kernel.operations.registry` | in-process (Protocol) |
| Memory recall/store | `MemoryStore` Protocol | in-process; MCP is one impl |
| State, blob, browser | `BlobIO` / `StateProvider` / `BrowserAutomation` | in-process (Protocol) |

Operations are **not** tools. Operations are internal reusable behaviors called by
modes (pre-turn, post-turn, entry). Tools are LLM-callable capabilities surfaced in the
provider's tool catalog. The kernel keeps the operation registry in-process and
exposes a curated subset of operations as MCP tools when the LLM should call them.

### New module: `kernel/tools/mcp_client.py`

Replaces the ad-hoc `kernel/mcp_client.py` (which is memory-only) with a general
MCP tool client pool.

```python
@dataclass
class MCPClientConfig:
    """One configured MCP server connection."""
    server_id: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    cwd: str | Path | None = None


class MCPToolPool:
    """Manages N MCP server subprocesses and exposes a unified tool catalog."""

    def __init__(self, configs: list[MCPClientConfig]) -> None: ...

    async def list_tools(self) -> list[MCPToolSpec]:
        """Merge tool lists from all connected servers."""

    async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Route one call to the server that owns the tool."""

    async def aclose(self) -> None: ...
```

### New module: `kernel/tools/schema_translator.py`

The kernel speaks MCP tool schema internally. At the provider edge, translate to
whatever the provider expects.

```python
def mcp_tool_to_openai_function(tool: MCPToolSpec) -> dict[str, Any]:
    """Translate one MCP tool to an OpenAI function-tool schema."""

def mcp_tool_to_anthropic_tool(tool: MCPToolSpec) -> dict[str, Any]:
    """Translate one MCP tool to an Anthropic tool-use schema."""

def openai_tool_call_to_mcp(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Translate one OpenAI tool_call back to (tool_id, arguments)."""

def anthropic_tool_use_to_mcp(block: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Translate one Anthropic tool_use block back to (tool_id, arguments)."""
```

### `TurnHooks` changes

| Hook | Today | After |
| --- | --- | --- |
| `build_extra_tool_schemas` | returns OpenAI-shaped dicts | **removed** — replaced by `mcp_tool_servers` |
| `execute_tool` | host dispatches by `tool_id` | **removed** — replaced by `MCPToolPool.call_tool` |
| `mcp_tool_servers` | n/a | `list[MCPClientConfig]` — the host declares which MCP servers to spawn |

The turn runner's tool loop becomes:

1. `pool.list_tools()` → list of `MCPToolSpec`
2. Translate to provider schema (`mcp_tool_to_openai_function` or `mcp_tool_to_anthropic_tool`)
3. Send to provider
4. On `tool_calls`, translate back (`openai_tool_call_to_mcp` or `anthropic_tool_use_to_mcp`)
5. `pool.call_tool(tool_id, arguments)` → `MCPToolResult`
6. Append to messages, loop

### Built-in MCP servers (kernel-owned)

The kernel ships a small set of MCP servers that expose its own capabilities to
external clients (and to itself when a host wants to run them out-of-process):

| Server | Package | Tools |
| --- | --- | --- |
| `lisa-context` | `mcp_server/` (already exists) | `add_memory`, `search_memory`, `list_memories`, `get_surface_context` |
| `lisa-operations` | new, `mcp_server/operations/` | one tool per operation in the registry |
| `lisa-search` | new, `mcp_server/search/` | `web_search` (wraps `kernel.web_search`) |
| `lisa-surf` | new, `mcp_server/surf/` | `surf_url`, `flush_session` |

A host can either wire these as in-process Protocols (default, zero subprocesses) or
spawn them as MCP servers for isolation or cross-process access. The kernel does not
prefer one path; both are first-class.

### What gets removed (hard cutover, no compat)

- `kernel/openai_function_tools.py` — replaced by `schema_translator.mcp_tool_to_openai_function`
- `kernel/mcp_client.py` (memory-only) — replaced by `kernel/tools/mcp_client.py` (general)
- `TurnHooks.build_extra_tool_schemas` — replaced by `mcp_tool_servers`
- `TurnHooks.execute_tool` — replaced by `MCPToolPool.call_tool`
- Operation-sourced OpenAI schemas in `kernel/operations/registry.py` — operations are not tools; if an operation should be LLM-callable, expose it through `lisa-operations` MCP server

## Part 3 — Memory: discoverable external systems with fallback

### Discovery via entry points

New entry-point group: `lisa.memory.providers`. Each entry point yields a
`MemoryProviderSpec`:

```python
@dataclass
class MemoryProviderSpec:
    """One discoverable memory backend."""
    provider_id: str
    display_name: str
    kind: Literal["mcp", "in_process"]
    # For kind="mcp": the MCP server config to spawn
    mcp_config: MCPClientConfig | None = None
    # For kind="in_process": a callable returning a MemoryStore implementation
    factory: Callable[[], "MemoryStore"] | None = None
    # Priority: lower = preferred. Built-in fallback is priority 100.
    priority: int = 50
```

### Discovery and selection

```python
def discover_memory_providers() -> list[MemoryProviderSpec]:
    """Return all installed memory providers, sorted by priority."""
    specs = list(entry_points(group="lisa.memory.providers"))
    return sorted(specs, key=lambda s: s.priority)


async def select_memory_store(
    *,
    preferred: str | None = None,
    surface_context: SurfaceContext,
) -> MemoryStore:
    """Return the best available MemoryStore for this host."""
    providers = discover_memory_providers()
    if preferred:
        providers = [p for p in providers if p.provider_id == preferred]
    for spec in providers:
        if spec.kind == "mcp" and spec.mcp_config:
            try:
                return MCPMemoryStore(spec.mcp_config, surface_context)
            except MCPConnectionError:
                continue
        if spec.kind == "in_process" and spec.factory:
            return spec.factory()
    return SQLiteMemoryStore.default()
```

### Built-in fallback: `SQLiteMemoryStore`

Replace `LiteMemoryStore` (keyword-overlap, file-backed) with a real SQLite-backed
store that supports the full `MemoryStore` protocol including embeddings via
`sqlite_vec`. This is the same shape as the private `app/memory_store.py` but with
the app dependencies inverted through the new seams.

| Capability | `LiteMemoryStore` (today) | `SQLiteMemoryStore` (new fallback) |
| --- | --- | --- |
| Storage | JSON file | SQLite + `sqlite_vec` |
| Recall | keyword overlap | BM25 + cosine (when embeddings configured) |
| Scope filtering | yes | yes |
| Consolidation | no | yes (promotion + eviction) |
| Embeddings | no | optional (when `ProviderGatewayProvider` returns an `EmbeddingGateway`) |
| Privacy scoping | yes | yes (enforced in storage keys, not prompt text) |

### Popular discoverable memory systems

These are the systems the kernel should be able to consume via
`lisa.memory.providers`. Each ships (or can ship) an MCP server; the kernel wraps
that server config as a `MemoryProviderSpec`.

| System | Stars | License | MCP server | Notes |
| --- | --- | --- | --- | --- |
| [mem0](https://github.com/mem0ai/mem0) | 60k | Apache 2.0 | yes (`mem0` CLI + server) | Multi-signal retrieval (semantic + BM25 + entity). Most popular. |
| [Letta (MemGPT)](https://github.com/letta-ai/letta) | 23k | Apache 2.0 | yes | Stateful agents with memory; heavier. |
| [@modelcontextprotocol/server-memory](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) | (reference) | Apache 2.0 | yes (npx) | Knowledge-graph based; lightweight reference. |
| [Zep](https://github.com/getzep/zep) | lower | Apache 2.0 | community | Long-term memory; less momentum in 2026. |
| [Chroma](https://github.com/chroma-core/chroma) | high | Apache 2.0 | community | Vector store; wrap as MCP for memory use. |

The kernel ships adapter specs for mem0, Letta, and the MCP reference memory server.
Hosts opt in by installing the target package; the entry point activates only when the
package is importable.

### `MCPMemoryStore` — the MCP-backed `MemoryStore` implementation

Wraps the existing `kernel/mcp_client.py` pattern but implements the full
`MemoryStore` Protocol so the kernel can use any MCP memory server as a typed store,
not just as a tool.

```python
class MCPMemoryStore:
    """MemoryStore backed by an MCP memory server."""

    def __init__(self, config: MCPClientConfig, surface: SurfaceContext) -> None: ...

    async def record_memory(self, request: MemoryWriteRequest) -> MemoryRecord: ...
    async def recall(self, query: str, **kwargs: Any) -> str: ...
    async def list_memories(self, *, user_slug: str, ...) -> list[MemoryRecord]: ...
    async def list_recent_memories(self, *, user_slug: str, ...) -> list[MemoryRecord]: ...
```

Privacy scoping is enforced by passing `surface_id` and `memory_scope` into every MCP
call. The MCP server is responsible for filtering; the kernel never trusts prompt
wording as a privacy control.

## Part 4 — What stays as Protocols (and why)

These stay as in-process `Protocol` seams, **not** MCP:

| Seam | Why not MCP |
| --- | --- |
| `MemoryStore` | In-process by default; MCP is one impl among many. Forcing MCP adds a subprocess for the common case. |
| `BlobIO` | Host-owned storage layout; in-process. |
| `StateProvider` | Hot cache of `UnifiedState`; in-process. |
| `BrowserAutomation` | Playwright session; in-process. |
| `ContentRegistry` | Filesystem reads; in-process. |
| `ModelPolicy` | Alias resolution; in-process. |
| `ProviderGateway` | HTTP to provider; in-process. |
| `EventBus` | Pub/sub; in-process. |
| `DataSanitizer` | Redaction; in-process. |

The rule: **if it's in-process, it's a Protocol. If it crosses a process boundary, it's MCP.**

## Migration Plan (hard cutover, no compat)

Per the "no compat or migration" preference, this is one cutover, not a staged migration.

### Phase 1 — Seams (1–2 days)

1. Add the five new Protocols to `kernel/seams/protocols.py`.
2. Ship the five public default implementations under `kernel/context/`.
3. Add `ContextSeams` to `TurnHooks`.
4. Rewrite every `kernel/context/*.py` module to depend on the seams, not `app.*`.
5. Add `kernel/tests/test_context_import_boundaries.py` asserting no `app.*` imports.
6. Delete the old `app.*` imports. No shim.

### Phase 2 — MCP tools (2–3 days) ✅

1. ✅ Add `kernel/tools/mcp_client.py` (general MCP tool pool — `MCPToolPool`, `MCPClientConfig`, `MCPToolSpec`, `MCPToolResult`).
2. ✅ Add `kernel/tools/schema_translator.py` (`mcp_tool_to_openai_function`, `openai_tool_call_to_mcp`, Anthropic stubs).
3. ✅ Replace `TurnHooks.build_extra_tool_schemas` and `execute_tool` with `mcp_tool_servers: list[MCPClientConfig]`.
4. ✅ Rewrite the turn runner's tool loop: `pool.list_tools()` → translate to OpenAI → send → translate back → `pool.call_tool()`.
5. ✅ Remove `build_openai_operation_schema` (operations no longer advertised as tools). `kernel/openai_function_tools.py` deferred to Phase 4 (still used by operation registry for declarative operation tool schemas and by `get_transition_tool_schema`). `kernel/mcp_client.py` (`MCPMemoryClient`) deferred to Phase 3 (still used by Little Lisa for memory).
6. ✅ Update Little Lisa: removed `execute_tool` from `TurnHooks`, wired `mcp_tool_servers=[]` (empty until Phase 4 creates `lisa-search` etc.), removed `operation_catalog` kwarg from `run_turn` call.

### Phase 3 — Memory discovery (1–2 days)

1. Add `kernel/memory/providers.py` with `MemoryProviderSpec` and `discover_memory_providers`.
2. Add `SQLiteMemoryStore` as the built-in fallback.
3. Add `MCPMemoryStore` implementing the full `MemoryStore` Protocol.
4. Ship adapter specs for mem0 (default when installed), Letta, and
   `@modelcontextprotocol/server-memory`.
5. Update Little Lisa to use `select_memory_store()` in `TurnHooks.recall_memory` / `store_memory`.
6. Delete `LiteMemoryStore` from the public path (it stays as a test fixture only).
7. Declare `mem0ai` as an optional extra in `pyproject.toml`; the mem0 adapter
   activates only when importable and runs out of the host `data-dir`.

### Phase 4 — Kernel-owned MCP servers (1–2 days)

1. Promote `mcp_server/` from memory-only to a small server family: `lisa-context`, `lisa-operations`, `lisa-search`, `lisa-surf`.
2. Each server wraps an existing kernel capability; no new business logic.
3. Document the server configs in `docs/` and register them as entry points.

### Phase 5 — Provider-edge translation (deferred)

Anthropic tool-use translation lands when the first Anthropic provider gateway is
wired. The `schema_translator.anthropic_tool_use_to_mcp` function is stubbed today and
implemented when needed. OpenAI translation ships in Phase 2.

## What this does not change

- The slice system (`kernel/slice.py`) — pressure model, unchanged.
- The front-door router (`kernel/frontdoor.py`) — unchanged.
- The IPSAR FSM (`kernel/ipsar_fsm.py`) — unchanged.
- The operation registry (`kernel/operations/registry.py`) — unchanged, just no longer
  doubles as a tool catalog.
- The pack/extension discovery model — unchanged.
- The privacy model — unchanged and now enforced through typed seams, not prompt
  wording.

## Resolved decisions

1. **MCP server lifecycle.** The kernel owns the pool lifecycle (restart, health
   checks, graceful shutdown). The host declares configs via `mcp_tool_servers`;
   the kernel manages the rest.
2. **Streaming.** Out of scope. MCP tool calls are request/response. The kernel
   `CompletionResult` is non-streaming. Streaming is a separate, orthogonal change.
3. **Operation-as-tool exposure.** All operations are tools and are always exposed
   through `lisa-operations`. No opt-in flag. The operation registry's full catalog
   is surfaced as MCP tools.
4. **mem0 as default.** mem0 is included as a default memory provider when its
   pip dependencies are installed and it can run out of the host's `data-dir`. The
   adapter spec activates `mem0ai` when importable; otherwise falls back to
   `SQLiteMemoryStore`. No bundling — `mem0ai` is an optional dependency declared in
   `pyproject.toml` extras.
