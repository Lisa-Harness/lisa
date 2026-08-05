# Changelog

All notable changes to the Lisa Framework will be documented in this file.

## [Unreleased]

### Added

#### Unified child-turn classification with effect receipts

- **`EffectReceipt` model** (`kernel/models.py`) — frozen Pydantic model
  carrying the `effect` kind, `capability_id`, and optional `artifact_id`
  for one side-effecting capability that executed. Collected at dispatch
  time from the capability's declared `EffectKind`, not inferred from tool
  names or response prose. The child-turn completion contract checks
  receipts, so a child that called a `mode_allowlist`-admitted write
  operation (invisible to the old `authored_allowed_*`-only detector) is
  correctly counted as having completed.
- **`BlockDiagnostic` model** (`kernel/models.py`) — structured reason
  for a blocked child turn: `reason` (capability_gate / input_guard /
  output_guard), `resolved_model_id`, `missing_capabilities`, `message`.
  Carried on `ChildTurnOutcome` when `status` is `blocked` so the parent
  surfaces the actual cause (router declined to escalate) instead of the
  old misleading "model too weak to follow the sub-turn contract" framing.
- **`ChildTurnOutcome` model and `ChildStatus` Literal** (`kernel/models.py`)
  — the single canonical child-turn result: `status`, `output_text`,
  `error_message`, `duration_ms`, `tool_call_count`, `called_tool_names`,
  `effect_receipts`, `yield_action`, `block_diagnostic`. The fork and
  delegate adapters map this to their host-facing models
  (`ForkChildResult` / `DelegationResult`).
- **`mode_completion_effects` function** (`kernel/operations/registry.py`)
  — the single canonical resolver for a mode's required side-effecting
  capabilities. Reads authored bindings plus `mode_allowlist`-admitted
  operations and tools, excluding ambient `default_available` capabilities.
  Replaces the faulty `_fork_mode_action_tool_ids` which read only
  `authored_allowed_*` and missed `mode_allowlist`-admitted ops (the
  cross-pack pattern `markdown_forge_document` uses). The advertiser
  (`build_operation_tool_schemas`) and the validator now share one source
  of truth.
- **`classify_child_turn` function** (`kernel/turn_phases/_child_turn.py`)
  — the single shared classifier called by both the fork and delegate
  paths. Fixed classification order: `blocked` → `model_yield` →
  `interrupted` → `timeout`/`error`/`cancelled` → completion-contract →
  `complete`. Replaces 130+ lines of duplicated classification logic in
  `_run_fork_child_turn` and `_default_delegate_fn`.
- **`TurnResult.effect_receipts`** field — list of `EffectReceipt`s
  collected during the turn. Threaded through the turn loop, the sub-turn
  contract briefing, and the midpoint write-nudge (which is now
  receipt-aware).
- **`ForkChildStatus` and `DelegationStatus`** gain `"blocked"` (and
  `"max_depth"` on `ForkChildStatus` for union compatibility).

#### Memory backend CRUD completion

- **`get_memory` and `delete_memory` tools** added to the `lisa-memory` MCP
  server, completing the essential CRUD subset (add / search / list / get /
  delete). Both are surface-scoped: the caller's `user_slug` +
  `workspace_id` + `memory_scope` + `surface_id` filters are applied
  server-side, so cross-surface reads and deletes by id guessing are
  impossible.
- **`memory.get` and `memory.delete` synthetic aliases** registered on the
  generic memory router (`host/memory_aliases.py`), routing to the
  default backend's `get_memory` / `delete_memory` native tools. The alias
  contract grows from 3 to 5; backends that don't declare a `get` or
  `delete` tool name simply don't get those aliases registered (graceful
  partial-manifest handling).
- **`SqliteMemoryStore.get_memory` and `delete_memory`** methods, scoped by
  user + workspace + memory_scope + surface_id.
- **`dispute_memory` operation simplified**: the old list-50-and-filter
  client-side hack is replaced with a direct `memory.get(id)` call.

#### MCP server introspection in the CLI/REPL

- **`/mcp` REPL command and `lisa mcp` CLI subcommand now show a full
  per-server readout**: status (warm/cold/failed), transport (the
  `runtime_kind` from `mcp_servers.yaml` — `venv_bin`, `http`, `python`,
  …), URL or command, tool count, and last error. `/mcp <server_id>` (and
  `lisa mcp <server_id>`) drill into one server's tools with name,
  description, and required schema args. Cold servers are warmed on demand
  via `pool.list_tools(allowed_server_ids={server_id})` so their tools
  appear without a full pool warm-up.
- **`McpServerHealth` extended** with `transport`, `url`, `command`,
  `tool_count`, `tool_names`, `permanently_failed`, and `restart_count`
  fields. The introspection collector (`host/introspect.py`) now
  reads new pool accessors (`cold_failed_ids`, `permanently_failed_ids`,
  `restart_counts`, `last_errors`, `tools_by_server`) so a cold server
  that tried and failed reports `failed` (not `cold`), and the operator
  sees *why* it is down via `last_error`.
- **`MCPClientConfig.runtime_kind`** passthrough field (a string — the
  `RuntimeKind.value`) set by `compile_mcp_config` and the in-code
  `lisa-content`/`lisa-catalog` configs. Carried as a plain string so
  `kernel.tools` does not import from `kernel.mcp_launch` (preserving the
  `mcp_launch → tools` dependency direction).
- **`MCPToolPool` captures `_last_errors`** on every connect/restart
  failure (URL-stripped and length-bounded via `_safe_exc_desc`) and
  clears them on successful connect. Five new read accessors
  (`cold_failed_ids`, `permanently_failed_ids`, `restart_counts`,
  `last_errors`, `tools_by_server`) expose the pool's existing private
  bookkeeping without adding new mutation surface.
- **Hard cutover**: the lossy `McpToolEntry = tuple[str, str, bool]` type
  alias and the 2-column `render_mcp_tools` renderer are deleted.
  `_collect_mcp_tools` now returns `(dict[str, McpServerHealth],
  dict[str, list[MCPToolSpec]])` so the render layer stops discarding
  tool descriptions and schemas.

### Removed

#### Memory backend: redundant `get_memories` tool

- **`get_memories` tool removed** from the `lisa-memory` MCP server. It was
  a thin presentational wrapper over `list_memories` (same query, same
  params, newline-serialized instead of JSON) with no alias target, already
  denied from the model catalog. The standalone server now exposes five
  coherent tools (add / search / list / get / delete) instead of four where
  two were redundant.

#### Kernel config: obsolete memory-store settings removed

- **`kernel/config.py` no longer defines the memory-store settings** that
  the MCP-memory cutover obsoleted (`memory_sync_batch_size`,
  `memory_final_sync_timeout_seconds`, `memory_flush_interval_seconds`,
  `memory_snapshot_interval_seconds`, `memory_embedding_dimensions`,
  `memory_recall_candidate_pool_size`, `memory_recall_recent_limit`,
  `memory_recall_lexical_limit`, `memory_recall_semantic_limit`,
  `memory_recall_final_limit`, `memory_promotion_threshold`,
  `memory_eviction_min_age_days`, `memory_eviction_default_ttl_days`,
  `memory_eviction_min_access_count`,
  `memory_eviction_max_active_per_scope`). Their entries were also dropped
  from the `_coerce_numeric_with_inline_comments` field validator so
  `kernel.config.Settings` imports cleanly again (the dangling references
  had made the whole `kernel` package unimportable under Pydantic v2).
  The active recall/auto-capture knobs (`memory_recall_enabled`,
  `memory_recall_pre_turn_limit`, `memory_recall_min_query_chars`,
  `memory_auto_capture_enabled`, `memory_auto_capture_min_salience`) are
  unchanged.

### Changed

#### Evidence freshness: expiry based on `collected_at`, no TTL

- **`freshness_expires_at` on context evidence is now the raw
  `collected_at` string** instead of `_parse_iso(collected_at)`'s
  `datetime`/`None`, which contradicted the field's `str` type and would
  raise a Pydantic validation error at construction. The TTL-based freshness
  config is gone; expiry is anchored to when evidence was collected.

#### CLI host: flowing single-column transcript replaces the four-pane layout

- **The terminal turn display is now one flowing column of transcript rows** —
  no panes, headers, or boxes. Each stage event appends (or updates) one
  left-aligned line; tool calls are one-liners that update in place as they
  complete (`→ mcp read_file(path) ✓ 400ms`).
- **The final answer is re-rendered clean on completion.** The streamed
  answer is staged during generation but never shown as the final — on
  completion the Live region tears down and a fully formatted Markdown Panel
  renders the complete answer, so it reads cleanly instead of interspersed
  with the tool calls that produced it.
- **Thinking is interspersed, collapsible segments.** A reasoning model's
  thinking becomes multiple transcript rows that interleave with the tool
  calls in arrival order (a tool/model event closes the current segment so the
  next reasoning opens a fresh one). `Esc+t` expands/collapses ALL visible
  thinking segments together; each collapsed segment shows a compact
  `▸ Thinking… N char` summary.
- **Togglable call-stats summary.** After the answer, a summary line renders
  (call count with ✓/✗, total model time, handoff note). `--debug` upgrades it
  to a per-row trajectory table so the operator can audit the full turn.
- **Three-tier detail surface for live rows.** Default is a clean view (stage
  lines + tool one-liners with status glyphs). `Esc+d` toggles a power-user
  tier that adds the graphic column back (iteration/tool pip counters,
  duration sparklines, token-flow suffixes). `--debug` selects the full
  operator rendering at construction (every bookkeeping stage event + the
  debug-context panel), so the pane-era information density survives as
  opt-in telemetry, not the default surface.
- **Differentiators kept.** The live `[mode] → handoff` transition lines and
  the fork fan-out per-child table — telemetry no competitor shows — survive
  as flowing rows. The `▾ mode · model · tokens · ms` footer is unchanged.
- **Model calls are a single in-place-updated row.** `model_call` opens a
  model row and `model_call_complete` / `model_call_failed` update it in
  place with ✓/✗ + duration — the same pattern as
  `tool_dispatch` / `tool_dispatch_complete`. Previously each model
  iteration appended two rows (one for the start, one for the complete),
  doubling the row count and splitting the status from the call it
  describes. Now one iteration = one row:
  `[mode] stage: model_call haiku #1 ✓ 340ms`.
- The `TransportCallbacks` Protocol in `host/callbacks.py` is untouched;
  this is a CLI render-layer cutover only (hard cutover per the project
  contract — the pane machinery is deleted, not shimmed).

### Added

#### `default_available_tools` for MCP-native tools

- **MCP-native tools can now be ambient-available to every mode**, mirroring
  the `default_available: true` flag pack `.tool.yml` tools declare. MCP-native
  tools have no pack YAML front matter (they're discovered at runtime from
  server subprocesses and keyed namespaced as `server_id.tool`), so the flag
  lives on the host manifest instead: `default_available_tools:` on each
  server entry in `mcp_servers.yaml`.
- **The host resolves the namespaced ids** (`server_id.tool`, or
  `<pack_slug>.<server_id>.tool` for pack-contributed servers) and passes
  them to the `lisa-catalog` subprocess via the `LISA_DEFAULT_MCP_TOOLS` env
  var. The SDK catalog loader (`_merge_default_capabilities`) unions them
  into every mode's `allowed_tool_ids` with the same opt-out and idempotent
  semantics as pack `default_available: true` tools. The turn-runner's
  existing MCP-tool filter admits them with no further change.
- **`lisa-storage.read_file` is now ambient-available** by default, declared
  in `host/mcp_servers.yaml`. Modes can read persisted artifacts
  without re-listing the tool or paying the cold-server spawn cost unless
  they actually call it. The previous delegate pack tool under
  `sdk/packs/core/tools/read_file/` is removed — `lisa-storage` stays the
  sole storage owner.

### Changed

#### `FileWorkflowInstanceStore` moved from `kernel/` to `host/`

- **File-backed workflow-instance store relocated from the kernel to the host
  runtime.** `FileWorkflowInstanceStore` does file I/O, holds an
  `asyncio.Lock`, and writes under the host's `data_dir` — it is a
  host-runtime implementation, not a runtime-neutral contract. This mirrors
  the existing `recovery` split (`kernel/recovery.py` contracts vs
  `host/recovery.py` `FileRecoveryService`) and the `approvals` split.
- **Kernel keeps only the seam.** `kernel/workflow_instance_store.py` is
  trimmed to the `WorkflowInstanceStore` Protocol; its imports drop
  `asyncio`, `json`, `os`, `Path`, and `is_complete`.
- **Hard cutover, no compat.** The implementation now lives in
  `host/workflow_instance_store.py`. The single non-test importer
  (`host/runtime.py`) and the host wiring test import from the new
  location. The seven `FileWorkflowInstanceStore` tests moved from
  `kernel/tests/test_workflow_instance.py` to
  `host/tests/test_workflow_instance_store.py`; the kernel test file
  no longer imports a host implementation.

#### `kernel/introspect.py` moved to `host/introspect.py`

- **Live introspection gatherer relocated from the kernel to the host runtime.**
  `gather_introspection` reads host-owned concrete resources (the MCP tool
  pool, turn log path, token usage path), so it is a host-runtime
  implementation, not a runtime-neutral contract. The kernel only owns the
  `IntrospectSnapshot` / `McpServerHealth` / `TokenUsageSummary` /
  `LogErrorEntry` / `OperationalHealth` models and the `introspect_fn` seam
  on `OperationExecutionContext` and `TurnRunnerTurnConfig.introspect`.
- **Hard cutover, no compat.** `kernel/introspect.py` and
  `kernel/tests/test_introspect.py` are deleted. The single non-test
  importer (`host/turn_hooks_builder.py`) now imports
  `from host.introspect import gather_introspection`. The test moved
  to `host/tests/test_introspect.py`. `kernel/README.md`'s Built-in
  Capabilities table drops the `introspect.py` row; the `introspect` hook
  row (seam contract) stays.

### Removed

#### Model roles and `/deep` command

The `roles` mapping, the `deep` model role, and the `/deep` host command
have been removed entirely. The router (`lisa-router`) now owns per-iteration
model selection; `mode_models.default` is the single static baseline.

- `roles:` block removed from `models.yaml`; `mode_models.default` is the
  required baseline model.
- `/deep <query>` host command removed; the router's `rules` and `fallback`
  handle escalation patterns instead.
- `RouterRequest.roles_needed` field removed; CEL rules use
  `route.required_capabilities` for capability-based routing.
- `lisa.extensions.model_roles` entry-point group removed.
- `kernel/model_roles.py` deleted.
- `default_for_roles` field removed from `ManifestModelEntry`.
- Operations that fell back to `resolve_model_fn("deep")` now use
  `"default"`.

### Added

#### Session context auto-install of required packs

A session plan can now declare install sources for required packs via
`resources.packs.auto_install`. At host boot, any required pack that is not
yet importable is installed from its declared source before content-root
discovery runs.

- Auto-install applies **only** to `resources.packs.required` (never `include`).
- Git refs **must** pin `@tag/@branch/@sha` — mandatory for auto-install
  regardless of `LISA_PACK_REQUIRE_GIT_REF`.
- Trust policy env vars (`LISA_PACK_ALLOWED_PREFIXES`,
  `LISA_PACK_GIT_ALLOW_HOSTS`, `LISA_PACK_GIT_ROOT`) are honored; a session
  context cannot override them.
- Failure of any required pack is a hard startup error.
- When at least one pack is installed, the host requests a restart so
  entry-point discovery sees the new packs on the next boot pass.

#### Per-session model manifest path

A session plan can now point at a custom `models.yaml` via
`context.models.path`. The path is resolved relative to the session-context
file's directory or as an absolute path. When omitted, the host loads
`data_dir/models.yaml` as before.

### Changed

#### CLI status line reflects mid-turn transitions (not routing)

The initial thinking-spinner stage was renamed from `routing` to `dispatch`. The separate stateless router planner was removed in a hard cutover; routing is now an emergent property of the turn loop (mode handoff happens mid-turn via `transition_stage`), so `routing` was a misnomer for the pre-pipeline phase the spinner actually covers (command parse, `start_mode` bypass, `current_mode_id` selection).

Along with the rename, the CLI status line now surfaces turn detail that was already emitted by the kernel but not rendered:

- **Active mode prefix.** Every status line is prefixed with `[mode]`, tracked live from `model_call.mode_id` and updated when a mid-turn handoff lands (`tool_dispatch_complete` with `transition="handoff"`). A `default → coding` switch is now visible on the spinner line itself.
- **Legible tool dispatch.** `tool_dispatch` renders a glyph + kind word per `tool_kind` — `→ op:`, `→ tool:`, `→ mcp:`, and `↦ transition_stage →  <target>` (or `fork (N children)`) — instead of a flat `tool_kind=…` suffix key, so the user can distinguish in-process operations, MCP calls, and mid-turn handoffs/forks at a glance.
- **Per-child fork table through `fork_complete`.** The fork fan-out's per-child table now survives `fork_complete` so the user sees *which* children succeeded/failed (`#0 coding · implement · ok`, `#1 markdown · no_tools`), not just ok/failed counts. The table is cleared on the next `start_thinking`.
- **Turn-summary footer.** A one-line footer after the response renders `▾ mode · model · tokens · elapsed`, plus a `→  <new mode>` note when the model handed off mid-turn. Sourced from `runtime.last_turn_context` and the per-turn elapsed timer.
- **Distinct dispatch spinner.** The `dispatch` phase uses a `dots2` spinner (vs the normal `dots`) so the user can tell "still deciding where this goes" from "model is generating."

Affected: `hosts/cli/transport.py` (`start_thinking`, `on_stage`, `_render_status_line`, `_render_tool_dispatch`, `_render_fork_stage`), `hosts/cli/repl.py` (`_render_turn_footer`), tests in `hosts/cli/tests/test_transport_stage.py` and the new `hosts/cli/tests/test_repl_footer.py`.

### Added

#### Exportable and default-available capabilities

- **`exportable` and `default_available` flags on operation/tool YAML.** Two pack-authored visibility flags control cross-pack capability access. `exportable: true` (default `false`) marks a capability as referenceable by modes in other packs; a foreign mode referencing a non-exported capability is rejected at startup by `validate_export_boundaries` (`kernel/operations/registry.py`). `default_available: true` (default `false`) marks a capability for ambient merge: the SDK catalog loader (`sdk/content/_catalog.py :: _merge_default_capabilities`) auto-unions the id into every mode's effective `allowed_operation_ids` / `allowed_tool_ids` before the kernel sees them, so modes inherit ambient capabilities (e.g. `recall_memory`) without re-listing them. The merge is idempotent and happens entirely in the SDK layer; the kernel's `tools_authoritative` filter is unchanged.
- **Mode-level opt-out keys: `accept_default_capabilities`, `excluded_operation_ids`, `excluded_tool_ids`.** `accept_default_capabilities: false` (default `true`) disables all ambient inheritance for a mode. `excluded_operation_ids` / `excluded_tool_ids` subtract specific ambient ids while keeping the rest. Added to `WorkflowModeConfig` (`kernel/models.py`), the FSM loader protected-keys sets, the SDK loader, and the kernel deserializer.
- **`authored_allowed_operation_ids` / `authored_allowed_tool_ids` on `WorkflowModeConfig`.** Captures the explicit front-matter allowlists before the default-available merge, so `mode_completion_effects` (`kernel/operations/registry.py`) checks only the authored set plus `mode_allowlist`-admitted capabilities — a chat-only mode that inherits `recall_memory` is not misflagged as requiring tools to complete.
- **Seeds: `recall_memory` is `exportable: true, default_available: true`; `web_search` (tool + operation) is `exportable: true` only** (search is costly, stays opt-in per mode).

### Changed

#### `search`, `markdown`, `compaction`, `letta` moved out of `sdk/packs` to external `lisa_packs` submodules

- **Four packs migrated from bundled workspace members to external packs.** `lisa-pack-search`, `lisa-pack-markdown`, `lisa-pack-compaction`, and `lisa-pack-letta` moved out of `sdk/packs/<name>` (import `sdk.packs.<name>`) to `lisa_packs/<name>/` submodules (import `lisa_packs.<name>`), each in its own `github.com/Lisa-Framework/lisa-pack-<name>` repository pinned into `lisa-framework/packs`. Import paths renamed from `sdk.packs.<name>` to `lisa_packs.<name>`; distribution names unchanged. `compaction` keeps its `from sdk.packs.checkpoint...` import because `checkpoint` stays bundled. The bundled `sdk/packs` set is now `core`, `introspect`, `memory`, `models`, `checkpoint`.
- **All four are on-demand via `/pack install`.** Removed from the framework `pyproject.toml` `[project.dependencies]` and `[tool.uv.sources]`, and from every host `pyproject.toml`. A fresh `uv sync` no longer installs them; install the ones you want with `uv pip install -e ../lisa_packs/<name>` or the in-app `/pack install lisa-pack-<name>`.
- **`lisa-pack-search` removed from `host` hard deps.** The transport-neutral runtime no longer depends on a research-personality pack; `/search` and `/discuss` are a starter-subset choice wired by the transport hosts (and now on-demand there too).
- **Test conftest repointed.** `kernel/tests/conftest_catalog.py` and `host/tests/conftest.py` now discover migrated packs via the sibling `lisa_packs/<name>/lisa_packs/<name>` content roots when the submodules are checked out, and omit them when absent. `kernel/tests/test_compaction_seams.py` imports the compaction factory from `lisa_packs.compaction` with `pytest.importorskip` so it skips when the pack isn't installed.

#### `lisa-pack-models` moved back into the framework; `lisa_packs` split into per-pack submodule repos

- **`lisa-pack-models` (formerly `lisa-pack-model-picker`) is now a bundled framework pack at `sdk/packs/models`.** Import path renamed from `lisa_packs.model_picker` back to `sdk.packs.models`; distribution name renamed from `lisa-pack-model-picker` to `lisa-pack-models`. Consumed as a uv workspace member (`sdk/packs/models`) rather than a path dependency. The bundled pack is seeded enabled by default.
- **`/models update <NL>` replaces `/advise-models` (hard cutover, no compat).** The `advise-models` mode and `models advise` hint are removed. `/models update` runs the `advise_models` operation directly from `handle_models_command`: it edits + validates the proposal via an LLM, writes it to `models.proposed.yaml`, then promotes it to `models.yaml` in one step. The operation's `mode_allowlist` is dropped so it runs as a host-command op from any mode. `complete_fn` is wired via `complete_live` (the same primitive `turn_hooks_builder` uses), `resolve_model_fn` via `resolve_little_lisa_model`, and storage tools via the host `mcp_tool_pool`.
- **`lisa_packs` is now a parent repo with 7 per-pack submodule repos.** `brave_search`, `cloudflare`, `coding`, `github`, `gmail`, `mem0`, and `stripe` each live in their own `github.com/Lisa-Framework/lisa-pack-<name>` repository and are pinned into `lisa_packs/<pack>/` as git submodules, preserving the `lisa_packs/<pack>/lisa_packs/<pack>/...` namespace layout. `model_picker` is removed from `lisa_packs` (it now lives in this framework tree). Existing clones of `lisa-framework/packs` must run `git submodule update --init --recursive` after pulling the split commit.

#### `route_hints` and `examples` front-matter fields removed (hard cutover)

- **`route_hints` and `examples` removed from all `_mode.md` front matter and the loader.** Both fields were vestigial from the removed separate-stateless-planner router (`kernel/router.py:plan_route`). Under the unified FSM, no runtime path read either field — they were parsed, validated, stored on `WorkflowModeConfig`, and never consumed at turn time. Removed from: all mode `_mode.md` files (`default` ×2, `learn`, `markdown`, `discuss`, `search`, `ponder`), `kernel/ipsar_fsm.py` (loader blocks + `_KERNEL_OWNED_FRONT_MATTER_KEYS`), `kernel/models.py` (`WorkflowModeConfig.route_hints` + `WorkflowModeConfig.examples` fields), `kernel/context/mcp_catalog.py`, `sdk/content/_catalog.py`, and `kernel/tests/test_catalog_loader.py`.

#### Pack repository split

- **External packs moved to `github.com/lisa-framework/packs`**: `lisa_pack_brave_search`, `lisa_pack_cloudflare`, `lisa_pack_github`, `lisa_pack_gmail`, `lisa_pack_mem0`, `lisa_pack_model_picker`, and `lisa_pack_stripe` are now in the standalone `lisa_packs` repo. Import paths renamed from `lisa_pack_*` to the shared `lisa_packs.*` namespace (e.g. `lisa_packs.cloudflare`). The framework consumes them via path dependencies (`../lisa_packs/<pack>`).
- **`lisa-pack-model-picker` moved to `../lisa_packs/model_picker`**: Moved out of the bundled `sdk/packs/model_picker` workspace member to the external `lisa_packs` repo. Import path renamed from `sdk.packs.model_picker` to `lisa_packs.model_picker`. Now consumed via a path dependency (`../lisa_packs/model_picker`) alongside the other external integration packs.

### Added

#### Coding workflow (`lisa-pack-coding`)

- Added the catalog-native local coding workflow: bounded workspace inspection, isolated Git worktrees, checked patch application, named finite checks, protected-branch commit and non-force push controls, and artifact-backed GitHub pull-request publication. Effectful steps require a host-owned exact approval and resume only the approved invocation.

#### Kernel (`lisa-kernel`)

- Extension front matter support: `WorkflowModeConfig.extra_front_matter` now carries every app/pack-owned front-matter key the kernel does not interpret (e.g. `lenses` — mode perspective tags like `lens.conversation`, `lens.reflection`). The kernel owns a fixed set of front-matter keys (routing, FSM, operation bindings, workshop config) and passes everything else through opaquely as extension data. Hosts and packs read `extra_front_matter` directly; the kernel never names, validates, or curates app-owned keys, so adding a new app-owned front-matter key requires no kernel change.
- Front-matter overrides via an optional sibling `_overrides.yaml` file beside each `_mode.md`. Apps can manage extension front matter (and a small set of kernel-owned but overridable fields: `description`, `default_model`) without editing pack-owned `_mode.md` files. The boundary is a **denylist**: the kernel names only the keys it owns and protects (routing, FSM, operation bindings, workshop config); attempting to override any kernel-owned key is rejected with a startup validation error. Every other key is app/pack-owned and passes through untouched. Overrides are deep-merged on top of the authored front matter; list values replace (not concatenate).

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
- Memory tools exposed over MCP: `add_memory`, `search_memory`, `list_memories`, `get_memories`.
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
- Shared routing engine (`router.py`) with fast-path regex matching, verifier-model confirmation, model-bounded routing fallback, and structured-output retry.
- Turn runner (`turn_runner.py`) — the single model+tool loop shared by lightweight hosts and the full app, with `TurnHooks` extension points and three-phase execution (pre-turn → tool loop → post-turn).
- Opportunistic slice system (`slice.py`) with pressure-model budget math: budget grows as a function of idle time, not a wall-clock tick.
- Pydantic contracts (`models.py`) for surface context, route decisions, operation execution, tool execution, workflow FSM, workshop fan-out, memory, web search, intelligence pipeline, unified state, context evidence, anomaly alerts, domain knowledge, self-model, and more.
- Provider gateway abstractions (`provider_core.py`) with `AzureOpenAIGateway`, `OpenAICompatibleGateway`, completion/embedding request/result dataclasses, and Anthropic usage extraction.
- Model manifest loader (`model_manifest.py`) with YAML/JSON parsing, designation-based model resolution, and per-mode model overrides.
- Content roots discovery (`content_roots.py`) via `lisa.extensions.content_roots` entry points and `SDK_CONTENT_ROOTS` env var.
- Extension runtime contracts (`extension_runtime.py`) for scheduled jobs, state hydration, heartbeat context, digest sections, debug context, anomaly detectors, and context evidence builders.
- Signal connector base class (`intake/connectors.py`) and data source base class (`intake/sources.py`).
- Brave Web Search integration (`web_search.py`) with configurable endpoint, market, and result count.
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

- `lisa-pack-core`: baseline public modes (default, workshop), authored routing prompts, user context templates, and research workshop operation with planner+synthesis pipeline.
- `lisa-pack-search`: research & reasoning pack owning discuss, search, and ponder modes plus the web_search tool, the web_search research-loop operation, and the discussion_briefing pre-turn operation. Discuss was absorbed from core so each mode is co-located with its enrichment operations.

#### Little Lisa (`little-lisa`)

- Local terminal host CLI with seven subcommands: `cli` (interactive REPL), `desktop` (localhost web UI), `verify` (environment checks), `setup` (interactive provider configuration wizard), `modes` (list supported/installed modes), `models` (list configured models with designations), and `once` (single-turn execution).
- Tray-first desktop mode: `little-lisa` with no arguments launches a system tray icon (Windows/macOS/Linux via `pystray`) that manages a desktop web host subprocess.
- Local aiohttp desktop web server with token-authenticated chat API, health check, and last-turn context endpoint.
- Manifest-driven model routing with `router`, `default`, and `deep` designations supporting `azure_openai`, `openai`, `anthropic`, and `ollama` providers.
- Terminal streaming support (`--streaming none|stage|text|hybrid[:delay]`) with configurable character delay.
- Structured JSONL turn logging with `--log`, `--no-log`, and `--log-level` options.
- Router diagnostics via `/debug router` with planner reasoning, confidence scores, fallback tracking, and structured-output retry recording.
- Runtime introspection commands: `/help capabilities`, `/settings`, `/debug context`, `/debug capabilities`, `/debug self`.
- File-backed thread persistence with `--thread-dir` and `--no-thread` options.
- Configurable artifacts root via `--data-dir` and memory store path via `--memory-db`.
- `--user`, `--session-context`, `--timezone`, and `--state-snapshot` runtime options.
- `prompt_toolkit` integration for command history and slash-command completion in CLI mode.
- MCP-backed memory via `lisa-context` with `--memory-db` override.
- `LittleLisaStateProvider` with bounded thread compaction and memory eviction via the slice system.
- `LocalBlobIO` for file-backed blob operations used by Markdown artifact generation.

#### MCP Server (`mcp-server`)

- FastMCP server (`"Lisa Context Server"`) exposing four memory tools over stdio: `add_memory`, `search_memory`, `get_memories`, `list_memories`.
- Surface-scoped privacy isolation enforced at the tool level.
- SQLite-backed memory store with configurable database path via `LISA_MCP_DB_PATH`.

#### Tooling & CI

- Hermes installer scripts (`install.sh`, `install.ps1`) with provider selection (`ollama`, `openai`, `azure_openai`, `anthropic`, `google`), custom install directory/branch, and optional diagnostics skip.
- CI matrix (Ubuntu, macOS, Windows × Python 3.11, 3.12).
- PyPI publish workflow on `v*` tags.
- `pytest` markers: `live` (real external APIs), `integration` (multi-module in-process), `eval` (deterministic agent regression, no model calls).
- Import-boundary tests across all packages enforcing public/private separation.

### Changed

- `little-lisa` turn logs now emit top-level resolved model fields and per-invocation token totals for both router planning and mode execution, making `turn.completed` JSONL entries readable without deep payload expansion.
- Router routing retries once without `response_format` when the first structured-output call fails, using explicit JSON-only instructions before deciding fallback.
- Slice budget is pressure-driven: budget grows as a function of idle time since the last slice finished, not a wall-clock tick. This is the canonical pattern for opportunistic house-keeping (reindex, eviction, sync) — everything not wall-clock bound is a candidate.
