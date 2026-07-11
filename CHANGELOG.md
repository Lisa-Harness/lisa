# Changelog

All notable changes to the Lisa Framework will be documented in this file.

## [Unreleased]

### Changed

#### Non-core pack externalization (Phase 1 + 2)

- **Upgraded `cloudflare`, `github`, `stripe` packs to full packs**: each now includes modes (`/cloudflare`, `/github`, `/stripe` action modes), operations (`cloudflare_status`, `github_status`, `stripe_status`), content (system prompts), `get_content_root()`, and `content_roots` entry points — in addition to their existing SDK extensions (signal connectors, anomaly detectors, heartbeat context, digest sections, context evidence, refresh summary sections).
- **Added `anomalies.py` + `anomaly_detectors` entry point to `stripe` pack** (was the only pack missing this extension kind).
- **Migrated pack import paths from `sdk_packs.*` to top-level `lisa_pack_*`**: `sdk_packs.cloudflare` → `lisa_pack_cloudflare`, `sdk_packs.github` → `lisa_pack_github`, `sdk_packs.stripe` → `lisa_pack_stripe`. Each pack is now a standalone top-level Python package installable independently.
- **Moved `sdk_packs/_settings.py` to `sdk/pack_settings.py`**: the shared `OwnedSettingsSource` and `build_owned_settings_sources` helpers now live in the SDK so external packs can import them without depending on `sdk_packs`.
- **Updated root `pyproject.toml` workspace members**: `sdk_packs/cloudflare`, `sdk_packs/github`, `sdk_packs/stripe` replaced by `lisa_pack_cloudflare`, `lisa_pack_github`, `lisa_pack_stripe` top-level directories. `sdk_packs/core` and `sdk_packs/search` remain as in-repo reference implementations.
- **Pack tests**: added `test_catalog.py` to each pack verifying mode discovery, operation discovery, and operation-mode allowlist consistency.

### Added

#### MCP memory + context (Part 3)

- `lisa-context` MCP server owns the SQLite-backed memory store and context broker.
- Kernel no longer ships in-process memory stores or memory hooks; memory is MCP-only.
- Memory tools exposed over MCP: `add_memory`, `search_memory`, `list_memories`, `get_surface_context`.
- Context broker exposed over MCP: `broker_context` returns `ContextBrokerResponse` (packet + markdown).

#### Kernel-as-MCP servers (Part 4)

- Added four MCP server packages under `mcp_server/src/`: `lisa_context` (memory tools), `lisa_operations` (operation tools), `lisa_search` (search tools), and `lisa_surf` (surf tools).
- **`mcp_server/pyproject.toml`** updated with script entry points: `lisa-context`, `lisa-operations`, `lisa-search`, `lisa-surf`.
- **`mcp_server/tests/test_mcp_servers.py`** — 17 tests validating tool registration and in-process dispatch for all four servers.

#### AWS Bedrock gateway (Part 1.2)

- **`BedrockGateway`** class in `kernel/provider_core.py` — AWS Bedrock runtime gateway via boto3 (optional dependency). Uses the Converse API (`bedrock-runtime.converse()`) for normalized cross-model interface across Claude, Llama, Nova, and Titan. Sync boto3 calls wrapped with `asyncio.to_thread`.
- **`_build_bedrock_messages`** helper — translates OpenAI-style messages to Converse `messages[]` format: `tool`-role → `user` with `toolResult` content blocks; `assistant`+`tool_calls` → `assistant` with `text` + `toolUse` content blocks; other roles pass through.
- **`_openai_tool_to_bedrock_tool`** helper — converts `{"type":"function","function":{...}}` to Converse `{"toolSpec":{"name":...,"description":...,"inputSchema":{"json":...}}}` shape.
- **`_extract_bedrock_response`** helper — extracts text and `toolUse` blocks from Converse response into flat OpenAI-shape `tool_calls`.
- **`_extract_bedrock_usage`** helper — extracts `inputTokens`, `outputTokens`, `totalTokens` from Converse usage.
- **Optional boto3**: `BedrockGateway.__init__` raises `ImportError` with pip instructions if `boto3` is not installed. No new required dependency in `kernel/pyproject.toml`.
- **Provider routing**: `KernelProviderGatewayProvider.gateway_for("aws")` returns `BedrockGateway`.
- **`kernel/tests/test_bedrock_gateway.py`** — 11 tests covering text-only, tool-use, tool-use-only, multi-block text, tool translation, assistant+tool_calls round-trip, system prompt, usage, `supports()`, and routing. All tests stub the boto3 client.

#### Google Gemini gateway (Part 1.1)

- **`GoogleGeminiGateway`** class in `kernel/provider_core.py` — REST API gateway for Google Gemini via `httpx.AsyncClient` (no new deps). Supports text generation, tool calling (functionDeclarations → functionCall → functionResponse), system instructions (top-level `systemInstruction`), and usage extraction from `usageMetadata`.
- **`_build_gemini_contents`** helper — translates OpenAI-style messages to Gemini `contents[]` format: `user`→`"user"`, `assistant`→`"model"`, `tool`-role→`functionResponse` parts, `assistant`+`tool_calls`→`text`+`functionCall` parts.
- **`_openai_tool_to_gemini_function`** helper — converts `{"type":"function","function":{...}}` to Gemini's `{"name":...,"description":...,"parameters":...}` shape.
- **`_extract_gemini_response`** helper — extracts text parts and `functionCall` parts from Gemini response, synthesizes `gemini_call_N` IDs (Gemini has no native tool-call IDs).
- **`_extract_gemini_usage`** helper — extracts `promptTokenCount`, `candidatesTokenCount`, `totalTokenCount`, `cachedContentTokenCount` from `usageMetadata`.
- **Provider routing**: `KernelProviderGatewayProvider.gateway_for("google")` returns `GoogleGeminiGateway`.
- **`kernel/tests/test_google_gateway.py`** — 11 tests covering text-only, tool-use, tool-use-only, multi-block text, tool translation, assistant+tool_calls round-trip, system prompt, usage, `supports()`, and routing.

#### Streaming support (Part 2 — v1: OpenAI + Anthropic)

- **`StreamedChunk` and `StreamedCompletion` dataclasses** in `kernel/provider_core.py`. `StreamedChunk` carries `content` (delta text), `finish_reason` (`'stop'`, `'length'`, `'tool_calls'`, or `None`), and `usage` (only on the last chunk). `StreamedCompletion` aggregates chunks with `content`, `tool_calls`, `usage`, `finish_reason`, and the original `chunks` list.
- **`_aggregate_streamed_chunks` helper** — concatenates chunk content, picks the last non-`None` `finish_reason` and `usage`, returns a `StreamedCompletion`.
- **`TurnHooks.complete_stream`** field — optional async generator hook. When set, the turn runner uses it instead of `complete()` for the model call. v1 streams tokens only; tool-call streaming deferred to v2.
- **Turn runner streaming path** — if `hooks.complete_stream` is set, the tool loop iterates chunks via `async for`, aggregates them, and wraps the result in `StreamedCompletionAdapter`. When `finish_reason == 'tool_calls'`, falls back to `complete()` for that iteration to get the structured tool-call payload.
- **`StreamedCompletionAdapter`** in `kernel/turn_runner.py` — bridges `StreamedCompletion` to the `CompletionResult`-like interface (`.content`, `.tool_calls`, `.usage`) expected by the tool loop.
- **`OpenAICompatibleGateway.complete_stream()`** — streams via `client.chat.completions.create(stream=True, stream_options={"include_usage": True})`. Yields `StreamedChunk` for each SSE event.
- **`AnthropicGateway.complete_stream()`** — streams via `client.messages.stream()` async context manager. Extracts `text_delta` events for content deltas, `message_stop` for finish_reason, and `get_final_message()` for usage.
- **`kernel/tests/test_streaming.py`** — 14 tests covering type defaults, chunk aggregation, OpenAI gateway streaming, Anthropic gateway streaming, turn runner streaming path, and tool_calls fallback.

#### Anthropic gateway tool-use path (Part 1.3)

- **`AnthropicGateway.complete()`** rewritten to translate OpenAI-format messages and tools into Anthropic format and extract `text`/`tool_use` content blocks back into the normalized `CompletionResult`. Handles system-prompt extraction (top-level `system` param), `temperature`, `tools` (via `_openai_tool_to_anthropic_tool`), and `thinking` budget when `model_def.thinking_model` is set.
- **`_build_anthropic_messages`** helper in `kernel/provider_core.py` — translates OpenAI chat messages to Anthropic messages format: `tool`-role messages become `user` messages with `tool_result` content blocks; `assistant` messages with `tool_calls` become `assistant` messages with `text` + `tool_use` content blocks; all other messages pass through.
- **`_openai_tool_to_anthropic_tool`** helper in `kernel/provider_core.py` — converts `{"type":"function","function":{...}}` to Anthropic's `{"name":...,"description":...,"input_schema":...}` shape.
- **Provider routing fix**: `KernelProviderGatewayProvider.gateway_for("anthropic")` now returns `AnthropicGateway` instead of incorrectly falling through to `OpenAICompatibleGateway`.
- **`kernel/tests/test_provider_core.py`** — 13 tests covering text-only, tool-use, tool-use-only, multi-block text, tool round-trip, assistant+tool_calls round-trip, temperature pass-through, system-prompt extraction, usage extraction, and provider routing.

#### Guardrails seam (Part 7.1–7.4)

- **`GuardResult` dataclass** in `kernel/seams/protocols.py` with `allowed`, `blocked_reason`, `sanitized_text`, and `annotations` fields.
- **`InputGuard` and `OutputGuard` Protocols** (`@runtime_checkable`) in `kernel/seams/protocols.py`. `InputGuard.validate(query, surface) -> GuardResult` runs in Phase 1 before the pre-turn slice; `OutputGuard.validate(text, surface) -> GuardResult` runs in Phase 3 after the tool loop and before `store_memory`.
- **`PassThroughInputGuard`** (`kernel/context/input_guard.py`) and **`PassThroughOutputGuard`** (`kernel/context/output_guard.py`) — default implementations that allow everything. Hosts override with real safety implementations (Azure AI Content Safety, Bedrock Guardrails, Presidio, regex, etc.).
- **`ContextSeams.input_guard` and `ContextSeams.output_guard`** fields with lazy default factories, mirroring the existing seam pattern.
- **`TurnResult.blocked`** field (`bool = False`) — set to `True` when an input or output guard blocks the turn.
- **Turn runner wiring**: input guard short-circuits before the model call (no house-keeping budget consumed); output guard replaces blocked content with a safe fallback, publishes a `guardrail.audit` event on the `EventBus`, and skips the post-turn slice. `on_turn_complete` and `store_memory` still fire with the safe-fallback text so the audit trail is complete.
- **`kernel/tests/test_guardrails.py`** — 7 tests covering pass-through, blocking, sanitizing, and no-regression paths.
- **`docs/KERNEL_ARCHITECTURE.md`** — guardrails section documenting the two insertion points, the `GuardResult` contract, and the enterprise implementation guide.

#### MCP-native kernel cutover

- **Phase 1 — Context seams**: 6 new Protocols in `kernel/seams/protocols.py` (`ContentRegistry`, `ModelPolicy`, `ProviderGatewayProvider`, `EventBus`, `DataSanitizer`, `ExtensionEvidenceBuilder`) with 6 default implementations under `kernel/context/`. `ContextSeams` dataclass on `TurnHooks` with lazy default factories. 3 context modules rewritten to depend on seams instead of `app.*` imports. 5 dead modules deleted. ~50 new config fields for context broker, memory, and passive intelligence settings. Full design spec in `docs/MCP_NATIVE_DESIGN.md`.
- **Phase 2 — MCP tool pool**: `kernel/tools/mcp_client.py` (`MCPToolPool`, `MCPClientConfig`, `MCPToolSpec`, `MCPToolResult`) and `kernel/tools/schema_translator.py` (OpenAI + Anthropic stub translators). `TurnHooks.build_extra_tool_schemas` and `execute_tool` replaced by `mcp_tool_servers: list[MCPClientConfig]`. Turn runner tool loop rewritten: `pool.list_tools()` → translate to OpenAI → send → translate back → `pool.call_tool()`. `build_openai_operation_schema` removed — operations are no longer advertised as tool schemas. Little Lisa updated: `execute_tool` removed from `TurnHooks`, `mcp_tool_servers=[]` wired, `operation_catalog` kwarg removed from `run_turn`.

#### Kernel (`lisa-kernel`)

- Durable token-usage telemetry handler (`kernel/token_usage.py`) with `TokenUsageTelemetryHandler` (appends every `TokenUsageEvent` to `token-usage.jsonl` under the host's data directory, with timestamp+nanosecond idempotency keys) and `build_token_usage_event` factory. Closes the gap where `TokenUsageEvent` was defined but never published or persisted.
- `CompletionUsage.duration_ms` field on provider results so time-based providers (e.g. Ollama local GPU time) can be billed by wall-clock duration alongside token counts.
- `TokenUsageEvent.duration_ms`, `TokenUsageEvent.time_cost_usd`, and `TokenUsageEvent.kind="time_billed"` for duration-based billing accounting.

#### Little Lisa (`little-lisa`)

- `StubCompletion` now carries `cached_tokens`, `reasoning_tokens`, and `duration_ms` so sub-counters and time are no longer silently dropped at the host boundary.
- `TurnResult` now carries `cached_tokens`, `reasoning_tokens`, `duration_ms`, and `latency_ms` so the turn runner accumulates them across tool-loop iterations.
- `_complete_live` publishes one `TokenUsageEvent` per LLM call (Anthropic and OpenAI-compatible paths) to the durable JSONL handler.
- `_apply_turn_usage` records `cached_tokens`, `reasoning_tokens`, `duration_ms`, and `latency_ms` in the per-turn ledger and per-model/mode/stage breakdowns.
- Session summary renders cached, reasoning, and duration columns in the Token Totals, By Model, By Mode, and By Stage tables.
- `_ensure_usage_entry` upgrades pre-existing breakdown dicts (with the old 3-key schema) in place so the accumulator math never raises `KeyError`.

### Changed

#### Kernel (`lisa-kernel`)

- Renamed kernel subpackages for narrative coherence: `kernel/io/` → `kernel/seams/` (host-injection protocols), `kernel/signal_connectors/` + `kernel/data_sources/` + `kernel/domain_knowledge/` → `kernel/intake/` (unified external data collection), `kernel/intelligence/` → `kernel/synthesis/` (post-collection pipeline). Entry-point group names unchanged in this phase; migration planned with deprecation window.
- Operation registry now skips `.py` modules that have a sibling `<stem>.operation.md` declarative entrypoint. These modules hold the Pydantic response-format contract for a prompt-authored operation, not a runtime `execute(ctx)` entrypoint, so they no longer trigger spurious `missing '<stem>.yml'` warnings during catalog load.
- `_extract_completion_usage` (turn runner) now returns a 7-tuple including `cached_tokens`, `reasoning_tokens`, `duration_ms`, and `latency_ms`; callers updated accordingly.
- `_extract_usage_counts` (Little Lisa) now returns a 5-tuple including `cached_tokens` and `reasoning_tokens`; both call sites updated.

## [0.1.0] — Unreleased

### Added

#### Kernel (`lisa-kernel`)

- Runtime-neutral FSM engine (`ipsar_fsm.py`) with mode catalog loading, stage validation, transition tool schema, and operation lifecycle hooks (entry, pre-turn, post-turn).
- IPSAR route resolution (`ipsar.py`) for detecting user-facing mode targets from natural-language messages.
- Shared front-door routing engine (`frontdoor.py`) with fast-path regex matching, verifier-model confirmation, model-bounded routing fallback, and structured-output retry.
- Turn runner (`turn_runner.py`) — the single model+tool loop shared by lightweight hosts and the full app, with `TurnHooks` extension points and three-phase execution (pre-turn → tool loop → post-turn).
- Opportunistic slice system (`slice.py`) with pressure-model budget math: budget grows as a function of idle time, not a wall-clock tick.
- Pydantic contracts (`models.py`) for surface context, route decisions, operation execution, tool execution, workflow FSM, workshop fan-out, memory, surf sessions, web search, intelligence pipeline, unified state, context evidence, anomaly alerts, domain knowledge, self-model, and more.
- Provider gateway abstractions (`provider_core.py`) with `AzureOpenAIGateway`, `OpenAICompatibleGateway`, completion/embedding request/result dataclasses, and Anthropic usage extraction.
- Model manifest loader (`model_manifest.py`) with YAML/JSON parsing, designation-based model resolution, and per-mode model overrides.
- Content roots discovery (`content_roots.py`) via `lisa.extensions.content_roots` entry points and `SDK_CONTENT_ROOTS` env var.
- Extension runtime contracts (`extension_runtime.py`) for scheduled jobs, state hydration, heartbeat context, digest sections, debug context, anomaly detectors, and context evidence builders.
- Signal connector base class (`intake/connectors.py`) and data source base class (`intake/sources.py`).
- Brave Web Search integration (`web_search.py`) with configurable endpoint, market, and result count.
- Durable surf session helpers (`surf_workflow.py`) for iterative browser workflows with TTL, trimming, and page/commentary tracking.
- MCP memory client (`mcp_client.py`) for delegating memory operations to the `lisa-context` MCP server over stdio.
- OpenAI function-tool schema builder (`openai_function_tools.py`) with strict-mode support.
- Blank state provider (`state_providers.py`) for hosts that don't need memory.
- OpenTelemetry span helpers (`telemetry.py`).
- Kernel-owned settings (`config.py`) for surf, web search, and workshop budgets.
- Extension harness (`testing/extension_harness.py`) for validating connectors and data sources without a host app.
- Operation registry (`operations/registry.py`) for discovering and executing operations from packs.

#### SDK (`lisa-sdk`)

- Author-facing helper contracts for packs, workshop execution, rendering, and integration contracts.
- `WorkshopExecutionService` (`workshop_execution.py`) for safe child-workflow fan-out with isolated session keys and structured concurrency.
- `RichAttachment` and `RefreshSummarySection` rendering contracts (`rendering.py`) for host-neutral rich reply content.
- SDK-owned settings (`config.py`) for heartbeat interval, signal timeout, memory recall limit, and surf/web search defaults.
- Import-boundary test enforcing `sdk` isolation from private app modules.

#### Integration Packs (`lisa-pack-*`)

- `lisa-pack-cloudflare`: Cloudflare signal connector with GraphQL analytics (worker requests/errors/CPU, edge status codes, KV/Durable Objects/D1 metrics, WAF events, Pages projects), anomaly detection, context evidence, heartbeat context, digest sections, and refresh summary.
- `lisa-pack-github`: GitHub signal connector with repo status, commit info, workflow runs, open PRs, anomaly detection, context evidence, heartbeat context, digest sections, refresh summary, read-only code search/file excerpt/tree listing client, and write-capable PR creation client with protected-branch safety.
- `lisa-pack-stripe`: Stripe billing signal connector with active/trialing subscriptions and MRR, context evidence, heartbeat context, digest sections, and refresh summary.

#### Packs (`lisa-pack-*`)

- `lisa-pack-core`: baseline public modes (default, workshop), authored front-door routing prompts, user context templates, and research workshop operation with planner+synthesis pipeline.
- `lisa-pack-search`: research & reasoning pack owning discuss, search, and thinkabout modes plus the web_search tool, the web_search research-loop operation, and the discussion_briefing pre-turn operation. Discuss was absorbed from core so each mode is co-located with its enrichment operations.

#### Little Lisa (`little-lisa`)

- Local terminal host CLI with seven subcommands: `cli` (interactive REPL), `desktop` (localhost web UI), `verify` (environment checks), `setup` (interactive provider configuration wizard), `modes` (list supported/installed modes), `models` (list configured models with designations), and `once` (single-turn execution).
- Tray-first desktop mode: `little-lisa` with no arguments launches a system tray icon (Windows/macOS/Linux via `pystray`) that manages a desktop web host subprocess.
- Local aiohttp desktop web server with token-authenticated chat API, health check, and last-turn context endpoint.
- Manifest-driven model routing with `front-door`, `default`, and `deep` designations supporting `azure_openai`, `openai`, `anthropic`, and `ollama` providers.
- Terminal streaming support (`--streaming none|stage|text|hybrid[:delay]`) with configurable character delay.
- Structured JSONL turn logging with `--log`, `--no-log`, and `--log-level` options.
- Front-door diagnostics via `/debug frontdoor` with planner reasoning, confidence scores, fallback tracking, and structured-output retry recording.
- Runtime introspection commands: `/help capabilities`, `/settings`, `/debug context`, `/debug capabilities`, `/debug self`.
- File-backed thread persistence with `--thread-dir` and `--no-thread` options.
- Configurable artifacts root via `--data-dir` and memory store path via `--memory-db`.
- `--user`, `--session-context`, `--timezone`, and `--state-snapshot` runtime options.
- `prompt_toolkit` integration for command history and slash-command completion in CLI mode.
- MCP-backed memory via `lisa-context` with `--memory-db` override.
- `LittleLisaStateProvider` with bounded thread compaction and memory eviction via the slice system.
- `LocalBlobIO` for file-backed blob operations used by Markdown artifact generation.

#### MCP Server (`mcp-server`)

- FastMCP server (`"Lisa Context Server"`) exposing four memory tools over stdio: `add_memory`, `search_memory`, `get_surface_context`, `list_memories`.
- Surface-scoped privacy isolation enforced at the tool level.
- SQLite-backed memory store with configurable database path via `LISA_MCP_DB_PATH`.

#### Tooling & CI

- Hermes installer scripts (`install.sh`, `install.ps1`) with provider selection (`ollama`, `openai`, `azure_openai`, `anthropic`, `google`), custom install directory/branch, and optional diagnostics skip.
- CI matrix (Ubuntu, macOS, Windows × Python 3.11, 3.12).
- PyPI publish workflow on `v*` tags.
- `pytest` markers: `live` (real external APIs), `integration` (multi-module in-process), `eval` (deterministic agent regression, no model calls).
- Import-boundary tests across all packages enforcing public/private separation.

### Changed

- `little-lisa` turn logs now emit top-level resolved model fields and per-invocation token totals for both front-door planning and mode execution, making `turn.completed` JSONL entries readable without deep payload expansion.
- Front-door routing retries once without `response_format` when the first structured-output call fails, using explicit JSON-only instructions before deciding fallback.
- Slice budget is pressure-driven: budget grows as a function of idle time since the last slice finished, not a wall-clock tick. This is the canonical pattern for opportunistic house-keeping (reindex, eviction, sync) — everything not wall-clock bound is a candidate.

