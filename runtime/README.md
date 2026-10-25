# Runtime

`lisa-harness-runtime` is an abstract runtime, i.e., host for the Lisa Harness kernel.
It offers a transport-neutral but otherwise near complete runtime implementation.
The separate [lisa-cli host](https://github.com/Lisa-Harness/lisa-cli) builds
on it to create a concrete running host.

See [host integration](../docs/HOST_INTEGRATION.md), [configuration](../docs/CONFIGURATION.md),
and the [documentation index](../docs/README.md) for maintained task guides.

## Design Philosophy

The host owns **runtime integration, process boundaries, and environment-specific wiring** — nothing more. Domain logic lives in the kernel or SDK; rendering lives in the transport.

Key principles:

- **Transport-neutral**: `runtime` never imports Rich, Typer, aiohttp, or any transport-specific package. Transports implement `TransportCallbacks` at their edge to receive stage updates, streamed text, final responses, debug context, and errors.
- **Typed composition root**: `HostConfig` is the frozen configuration; `HostRuntime` is its mutable counterpart, grouping responsibilities into five focused sub-objects (`paths`, `providers`, `resources`, `session`, `services`) so call sites name the domain they touch.
- **Single message entry point**: all transports call `handle_message(runtime, text, transport)` — host command dispatch, approval resume, model-turn routing, and turn execution live behind it.
- **Runtime invariants**: MCP resources start and close in the same task and event loop that runs turns. Housekeeping is slice-budgeted — no `cron`, `sched`, or wall-clock background schedulers.

## Usage

```python
from runtime import NullTransport, handle_message
from runtime.bootstrap import build_runtime

runtime = build_runtime(host_kind="headless", data_dir="./agent-data", user_slug="local-user")
try:
    await runtime.start()
    result = await handle_message(runtime, "/help", NullTransport())
finally:
    await runtime.aclose()
```

This fragment runs inside one async task with a configured model manifest.
`HostRuntime(config)` alone does not perform the bootstrap/wiring sequence.
For a complete program and an observable production transport, see
[host integration](../docs/HOST_INTEGRATION.md).

## Module Map

### Composition Root

| Module | Responsibility |
| --- | --- |
| `config.py` | `HostConfig` — typed, frozen configuration built once at startup from CLI args, env vars, or direct construction. Owns surface identity defaults and per-lane context budgets. |
| `runtime.py` | `HostRuntime` — the typed composition root replacing the old `_RUNTIME_STATE` dict. Owns `config`, `session_plan`, `execution_policy`, and the five sub-objects. `RuntimeResources` implements the async MCP lifecycle beside the object that owns its event loop. |
| `bootstrap.py` | Shared composition root: builds a `HostRuntime` from CLI args + env vars. |
| `handle_message.py` | Transport-neutral message dispatcher. Owns host command dispatch (`/help`, `/settings`, `/tokens`, `/debug`, `/consolidate`, `/intake`, `/pack`, `/models`, `/modes`, `/operations`, `/tools`), implicit approve/deny resume, and the model-turn pipeline. |
| `types.py` | Transport-neutral `Protocol` callable shapes that `HostRuntime` and `handle_message` depend on. |
| `runtime_paths.py` | `HostPaths` — frozen, pure value object projecting the host's derived filesystem layout (threads, state, logs, scratch). |
| `paths.py` | Framework root, venv root, and packs-venv resolution that works both in a repo checkout and a `uv tool install` environment. |

### Turn Wiring

| Module | Responsibility |
| --- | --- |
| `turn_hooks_builder.py` | Builds the kernel's `TurnHooks` dataclass from a `HostRuntime`, replacing the old closure-based factory. |
| `turn_hooks/` | Turn-hook factory submodules — each owns one cohesive group of hooks (e.g. `_auto_capture.py`). |
| `context_assembly.py` | Transport-neutral system prompt and user context assembly. Pure / parameterized functions. |
| `token_accounting.py` | Token/latency ledger and session usage tracking. |
| `turn_logging.py` | Structured log-record assembly for the JSONL turn log and runtime event log. |
| `state_provider.py` | Transport-neutral state provider and slice compaction. |
| `operator_signal.py` | Typed `OperatorSignal` enum for host↔kernel communication, polled once per iteration. |

### Providers & Models

| Module | Responsibility |
| --- | --- |
| `providers.py` | `ProviderRegistry` — lazily-populated per-provider SDK client cache (Azure OpenAI, Azure Responses, OpenAI, Anthropic HTTP, Ollama, OpenRouter). |
| `gateway_provider.py` | Host-side `ProviderGatewayProvider` — the kernel's single live completion path routes through it. |
| `provider_endpoints.py` | Single source of truth for each provider's direct base URL, used when no proxy route is active. |
| `model_runtime.py` | Model-runtime resolution helpers. |
| `model_manifest.py` | Host-side model manifest handling. |
| `model_discovery/` | Provider capability probing and manifest generation (`probes/` per provider, `classify.py`, `discover.py`, `yaml_emit.py`, `redact.py`, `enrichment`). Backs the `/models discover` command. |

### MCP Integration

| Module | Responsibility |
| --- | --- |
| `mcp_pool.py` | Compiles MCP server configurations from the host manifest (`mcp_servers.yaml`) and pack-contributed declarations. |
| `mcp_binary_cache.py` | Downloads release-hosted binary MCP servers, verifies release checksums, and caches them under the host data directory. |
| `mcp_servers.yaml` | The host's first-party MCP server manifest (shipped as package data). |
| `capability_servers.py` | Selects one MCP implementation for each declared capability. |
| `ambient_resolvers.py` | Resolves runtime identity for capability-owned MCP tools. |

Binary server manifests may fetch a GitHub release with `binary_download`
(`repository`, `binary_name`, and optional `version`). Pack manifests should
omit `version`: the runtime then downloads the release tagged `v<pack
version>`, so the tag the pack was installed from is the only version
selector. The runtime downloads only selected implementations, verifies the
asset against the release's `checksums.txt`, and caches it under
`<data_dir>/cache/mcp-binaries/`.

### Approvals, Policy & Recovery

| Module | Responsibility |
| --- | --- |
| `approval/` | Transport-neutral exact-action approval flow: identity rebuild, affirmative resume, policy auto-continue, post-approval continuation (`presentation.py`, `orchestration.py`). |
| `approvals.py` | File-backed host implementation of the kernel approval service. |
| `policy.py` | Host-owned approval policy engine — the only place the "does this action need operator approval?" decision lives (mode dial, allow/ask/deny). |
| `policy_service.py` | `PolicyApprovalService` — decorates the durable file-backed service with the policy engine. |
| `execution_policy.py` | Host-side execution policy defaults and resolution. |
| `recovery.py` | File-backed host implementation of the kernel recovery service. |

### Session & Content

| Module | Responsibility |
| --- | --- |
| `session_context.py` | Resolves a session-context reference into a strict v2 `SessionPlan`. |
| `session_from_resolver.py` | Host-layer `from:`/`body:` directive resolver for session contexts. |
| `content_roots.py` | Content root and pack activation discovery (packs venv injection, installed-pack deduplication). |
| `pack_autoinstall.py` | Startup auto-install of session-required packs not importable at host boot. |

### Services, Intake & Introspection

| Module | Responsibility |
| --- | --- |
| `services.py` | `HostServices` — the host's durable service handles (approval, recovery, etc.) reachable by any turn or command. |
| `intake.py` | Intake tick — one-shot signal collection, processing, and state write driven by a systemd timer or cron. |
| `introspect.py` | Live Python-level introspection: process memory, asyncio task count, MCP pool health, token usage trends. |
| `code_artifacts.py` | Atomic file-backed storage for pack-neutral WorkProducts. |

### Commands & Thread

| Module | Responsibility |
| --- | --- |
| `commands/` | Transport-neutral slash-command handlers that do not require the model (`_mcp.py`, `_packs.py`, `_models.py`, `_policy.py`, `_routing.py`, `_slice.py`, `_lists.py`, `_log.py`, `_render.py`, `_simple.py`). |
| `host_commands.py` | Dynamic registry of slash-command prefixes that are host control directives, not conversational content. |
| `thread/` | Per-user segmented thread store (`_store.py` — paths, segment index, zstd compression, sealing; `_excerpt.py`). |
| `testing/` | `runtime_factory.py` — builds a test `HostRuntime` with injected fakes and a seeded model manifest. |

## Data Directory Precedence

`HostConfig.from_env` resolves the data root in this order: CLI arg > `LISA_DATA_DIR` > `LISA_PROFILE` > `~/.lisa`. `LISA_PROFILE=name` resolves to `<base>/profiles/<name>` while the packs venv stays the base's. Export only one of the two variables.

## Testing

```bash
uv run pytest runtime/tests runtime/src/runtime/model_discovery/tests -m "not live" -q
```

Run from the workspace root; see [testing](../docs/TESTING.md) for full gates.

Test through `HostConfig` and `HostRuntime` with injected fakes (via `runtime/testing/runtime_factory.py`) rather than monkeypatching module-level interfaces. A test `HostRuntime` needs a seeded model manifest with `mode_models`, not only a `models` list.

Architecture boundaries for this package are defined in [.github/instructions/architecture/runtime.instructions.md](../.github/instructions/architecture/runtime.instructions.md).
