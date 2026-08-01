# Lisa Framework Roadmap

> Generated 2026-07-23 from a review of the codebase, documentation, known
> incident records, and the wish list.

This roadmap organizes work into four horizons — **stabilization**,
**architecture hardening**, **feature completion**, and **future
expansion** — ordered by urgency and dependency.

---

## Horizon 0 — Stabilization (fix now)

These are live bugs or regressions that affect daily use. They should be
addressed before any new feature work.

### 0.1 Model-router factory-factory crash *(fixed this session)*

- **Status:** ✅ Fixed in `hosts/core/turn_hooks_builder.py`.
- **Root cause:** `get_router_factory` is a factory-factory (0 args → returns
  the real `_factory(manifest, session_plan)`), but `_discover_active_router`
  called it directly with 2 args. The logging handler also had a format-string
  mismatch that masked the real error. See
  `/memories/repo/model-router-factory-factory-bug.md`.
- **Pattern risk:** The same factory-factory arity issue was already fixed for
  slice providers but not generalized. **A shared helper** for arity-based
  factory resolution would prevent this recurring across future entry-point
  groups.

### 0.2 Fork fan-out reliability

- **Status:** ✅ Fixed across 2026-07-17 to 2026-07-18 (see `docs/lessons/`).
- **Symptoms:** Fork children reported "completed successfully" but wrote nothing
  to disk; distillation produced empty output on Ollama; markdown child never
  called its write tool; fork reported false success.
- **Root causes (documented in incident files):**
  - Empty distillation output from Ollama (HTTP 200, empty body) — no
    structured retry.
  - `max_tokens=300` truncating JSON mid-fact → invalid fenced JSON.
  - `content: <nil>` rejected by Ollama's OpenAI-compatible endpoint when
    assistant messages have tool_calls but no content.
  - Fork child status not validated against actual side-effects (writes,
    memory records).
- **What shipped (all four action items applied):**
  1. Structured-output retry with increased `max_tokens` (300→512) and
     `response_format=None` on retry — `kernel/operations/declarative.py`
     and `kernel/operations/structured_output.py`.
  2. `normalize_openai_messages` sets assistant `content` to `""` (not
     omit) when tool_calls present — `kernel/provider_core.py`.
  3. Fork-child result validation: `ForkChildStatus` now includes `"no_tools"`;
     `_fork_mode_requires_tools` rejects a child that lacks its required
     tools — `kernel/turn_phases/_fork.py`, `kernel/models.py`.
  4. Fork regression tests added — see `test_run_turn_fork_no_tools*`,
     `test_distill_retry.py`, `test_normalize_*`.
- **Follow-on (2026-07-27): model refusal to fork.** The above fixes
  caught children that hallucinated success, but the upstream gap
  remained: a weak model could refuse to call `transition_stage` with
  the `fork:` target at all, burning iterations on read-only tools or
  returning empty completions until the iteration cap, ending the turn
  with no fork ever running. Fixed by the forced-fork enforcer — see
  `docs/lessons/lisa-fork-refusal-2026-07-27.md` and the Forced-Fork
  section of `docs/ROUTER_DESIGN.md`. Regression tests:
  `test_forced_fork_when_model_refuses_to_fork`,
  `test_forced_fork_when_model_returns_empty_completion`.

### 0.3 Pre-existing test environment issues

- **Status:** `kernel/tests/test_model_router.py` fails at collection time
  with `ModuleNotFoundError: No module named 'lisa_packs'` because
  `kernel.turn_runner` transitively imports an external pack.
- **Action:** Break the transitive import dependency or gate it behind a
  `TYPE_CHECKING` guard / lazy import so kernel tests can run without external
  packs installed.

### 0.4 Stale Pyright / lint issues

- `hosts/core/turn_hooks_builder.py` has several pre-existing type-check
  warnings (`complete_live` missing mandatory kwargs via `**kwargs`,
  `execute_operation` defined outside `__init__`, `ModelRouter` not callable
  through factory-factory indirection). These are largely false positives from
  `**kwargs` forwarding but should be suppressed cleanly or fixed with proper
  typing.

---

## Horizon 1 — Architecture Hardening (next sprint)

### 1.1 Generalize factory-factory resolution

- Extract the `inspect.signature`-based arity detection (currently duplicated
  in slice-provider wiring ~line 470 and model-router wiring ~line 187 of
  `turn_hooks_builder.py`) into a shared utility, e.g.
  `kernel/entry_point_utils.py:resolve_factory(entry_factory, *args)`.
- All future `lisa.extensions.*` entry-point groups should use it.

### 1.2 Fork child observability

- **Status:** ✅ Shipped. `ForkChildResult` exists in `kernel/models.py` with
  `status: ForkChildStatus` (where `ForkChildStatus = Literal["ok",
  "timeout", "cancelled", "error", "no_tools"]`), `output_text: str`, and
  `duration_ms: int`. The parent aggregates child results into
  `TurnResult.fork_results`; `_fork_mode_requires_tools` rejects children
  that lack their required tools (status `"no_tools"`).
- **Note on shape:** The shipped fields differ from the original proposal
  (`succeeded`/`side_effects`/`error`). The `status` enum captures the same
  intent more precisely (the `"no_tools"` case was the 2026-07-18 false-success
  root cause). A future enhancement could add an explicit `side_effects` list
  if richer attribution is needed.

### 1.3 Ollama provider compatibility

- **Status:** ✅ Shipped. `normalize_openai_messages` in
  `kernel/provider_core.py` sets assistant `content` to `""` when tool_calls
  are present, for Ollama compatibility. Distillation retry handles the
  empty-body / truncated-JSON cases (see 0.2).

### 1.4 Consolidate incident learnings into regression tests

- **Status:** ✅ Shipped. Regression tests added for each incident mode:
  `test_run_turn_fork_no_tools*` (false-success / `no_tools` status),
  `test_distill_retry.py` (empty distillation retry), `test_normalize_*`
  (message normalization). See `docs/lessons/` for the incident records.

---

## Horizon 2 — Feature Completion (next quarter)

### 2.1 Capability-manifest-aware proactive routing hints

- **Status:** ✅ Shipped as `kernel/turn_phases/_routing_hint.py` and
  `kernel/turn_runner.py::_compute_capability_routing_hint`. Before the model
  call, the compiled `CapabilityManifest` is compared against the user's
  query; if the current mode lacks tools the query needs, a routing hint is
  injected. This is proactive (zero wasted iterations) and costs no model
  calls. The hint also escalates complex multi-step queries (3+ gap intents or
  2+ sequential) to a `plan` action mode that runs `plan_workflow`.
- **Durable chain instances:** ✅ Both planned (`plan_workflow_op`) and
  ad-hoc (`adhoc`) multi-step queries now persist a `WorkflowInstance`
  so the chain survives a yield → "continue" resume. The ad-hoc path
  compiles steps via `compile_steps_from_routing_targets` and is guarded
  by `is_plan_escalation_hint` so it skips when plan mode owns the chain.
  See [docs/ROUTER_DESIGN.md §5c](ROUTER_DESIGN.md#5c-durable-workflow-chains-ad-hoc-vs-planned).

### 2.2 Slice-driven pre-computed routing context

- **From `docs/ROUTING_RFC_HANDOFF.md` §enhancement 2.**
- A post-turn slice analyzes the last N turns, detects topic shifts, and
  pre-computes a routing hint for the next turn, written into a state blob
  that `assemble_user_context` reads.
- Stateful routing context with zero extra model calls — unique to Lisa's
  turn-anchored opportunistic slice mechanism.

### 2.3 Mode-to-mode direct handoff

- **Status:** ✅ Shipped as a full routable mesh — see `kernel/ipsar_fsm.py`'s
  `synthesize_routing_transitions` and `docs/ROUTER_DESIGN.md` §4.
- **Note on shape:** The original proposal was an authored `can_handoff_to:`
  front-matter key with per-mode opt-in. The shipped implementation is instead a
  **permissive default**: every routable mode can hand off to every other
  routable mode. Per-edge restrictions (e.g. forbidding `search` → `coding`)
  remain a future, smaller, authored opt-in. A `can_handoff_to` key is therefore
  not needed today.

### 2.4 Model router expansion

- **From `sdk/router/README.md`.**
- The external `lisa-pack-aoi-router` (vendor-backed, ML-based routing) is
  deferred. When ready, it registers under the same
  `lisa.extensions.model_routers` group and activates when
  `router.backend == "aoi-router"`.
- Consider adding more rule conditions to the built-in `lisa-router`:
  `mode_id` matching, `budget_remaining` thresholds.

### 2.5 External pack dependency management

- `lisa_packs` is an external repo that some kernel tests transitively import.
  Formalize the boundary: kernel tests must not require external packs. Either
  move shared test fixtures into the kernel, or mock the entry points.

---

## Horizon 3 — Future Expansion (backlog)

### 3.1 Containerization

- (Backlog item — also tracked in `docs/SPEC/todo.md`.)
- Containerize MCP servers first, then the REPL. This enables reproducible
  deployments and isolates pack dependencies.

### 3.2 Systemd / init for data-gathering extensions

- (Backlog item — also tracked in `docs/SPEC/todo.md`.)
- The `intake/` subpackage is designed for a one-shot process triggered by a
  systemd timer or cron. Provide systemd unit templates and an init wrapper.

### 3.3 Additional transport hosts

- (Backlog item — also tracked in `docs/SPEC/todo.md`.)
- WhatsApp / Telegram / Signal hosts. Each is a thin transport adapter over
  `hosts/core`, implementing `TransportCallbacks`. The architecture is ready
  (see `../hosts/README.md`) — only the transport adapter needs writing.

### 3.4 Stuck-session detection improvements

- **From `docs/ROUTING_RFC_HANDOFF.md` §enhancement 3.**
- The `_STAGE_NUDGE_CAP` (6 turns in the same stage) is a blunt instrument.
  Consider smarter progress detection: has the model made any tool calls?
  Has it produced any non-empty text? Has it changed stage at all?

### 3.5 `/models discover` robustness

- The proposed manifest (`artifacts/models.proposed.yaml`) currently includes
  `gpt-5.2-codex` and similar frontier models that may not be available in
  the local environment. Add a validation step that checks model availability
  before proposing them, or mark unavailable models distinctly.

---

## Summary Table

| ID | Item | Horizon | Priority |
| --- | --- | --- | --- |
| 0.1 | Model-router factory-factory crash | Stabilization | ✅ Done |
| 0.2 | Fork fan-out reliability | Stabilization | ✅ Done |
| 0.3 | Test env import breakage | Stabilization | 🟡 High |
| 0.4 | Stale Pyright/lint issues | Stabilization | 🟡 Medium |
| 1.1 | Generalize factory-factory resolution | Hardening | 🟡 High |
| 1.2 | Fork child observability | Hardening | ✅ Done |
| 1.3 | Ollama provider compatibility | Hardening | ✅ Done |
| 1.4 | Incident regression tests | Hardening | ✅ Done |
| 2.1 | Proactive routing hints | Completion | ✅ Done |
| 2.2 | Slice-driven routing context | Completion | 🟢 Medium |
| 2.3 | Mode-to-mode direct handoff | Completion | ✅ Done (full mesh) |
| 2.4 | Model router expansion | Completion | 🟢 Low |
| 2.5 | External pack dependency boundary | Completion | 🟡 Medium |
| 3.1 | Containerization | Future | 🔵 Backlog |
| 3.2 | Systemd for extensions | Future | 🔵 Backlog |
| 3.3 | Additional transport hosts | Future | 🔵 Backlog |
| 3.4 | Stuck-session detection | Future | 🔵 Backlog |
| 3.5 | `/models discover` robustness | Future | 🔵 Backlog |
