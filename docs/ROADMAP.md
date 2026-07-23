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

- **Status:** Recurring failure across 4 incident records (2026-07-17 to
  2026-07-18).
- **Symptoms:** Fork children report "completed successfully" but write nothing
  to disk; distillation produces empty output on Ollama; markdown child never
  calls its write tool; fork reports false success.
- **Root causes (documented in incident files):**
  - Empty distillation output from Ollama (HTTP 200, empty body) — no
    structured retry.
  - `max_tokens=300` truncating JSON mid-fact → invalid fenced JSON.
  - `content: <nil>` rejected by Ollama's OpenAI-compatible endpoint when
    assistant messages have tool_calls but no content.
  - Fork child status not validated against actual side-effects (writes,
    memory records).
- **Action items:**
  1. Add structured-output retry with increased `max_tokens` on empty or
     truncated distillation.
  2. Normalize assistant messages: set `content: ""` (not omit) when
     tool_calls present, for Ollama compatibility.
  3. Add fork-child result validation: a child that claims success must
     produce a non-empty `ToolResult` or `OperationResult` — otherwise
     mark the child as failed.
  4. Add a fork-level integration test that verifies both children produce
     observable side-effects.

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

- Add structured fork-child status tracking: each child must report a
  `ForkChildResult` with `succeeded: bool`, `side_effects: list[str]`, and
  `error: str | None`.
- The parent turn should not report aggregate success unless all children
  report `succeeded=True` with non-empty side-effects.

### 1.3 Ollama provider compatibility

- The Ollama OpenAI-compatible endpoint has known quirks:
  - Rejects `content: null` on assistant messages with tool_calls.
  - May return HTTP 200 with empty body.
  - Truncates `max_tokens` without warning.
- Add provider-specific message normalization in `kernel/provider_core.py`
  or `hosts/core/model_runtime.py` that detects `ollama` provider and
  patches message content before sending.

### 1.4 Consolidate incident learnings into regression tests

- Each of the 4 incident docs (`lisa-coding-mode-fork-failure`,
  `lisa-fork-empty-distillation`, `lisa-fork-false-success`,
  `lisa-markdown-stage-failure`) documents a specific failure mode. Convert
  each into a deterministic regression test (no model calls) that asserts the
  fixed behavior.

---

## Horizon 2 — Feature Completion (next quarter)

### 2.1 Capability-manifest-aware proactive routing hints

- **From `docs/ROUTING_RFC_HANDOFF.md` §enhancement 1.**
- Before the model call, compare the compiled `CapabilityManifest` against the
  user's query. If the current mode lacks tools the query needs, inject a
  routing hint into the system prompt ("You don't have file-writing tools;
  call `transition_stage` to switch to `markdown`").
- This is **proactive** (zero wasted iterations) and costs no model calls.
- **Prerequisite:** A keyword-to-capability-gap lookup table.

### 2.2 Slice-driven pre-computed routing context

- **From `docs/ROUTING_RFC_HANDOFF.md` §enhancement 2.**
- A post-turn slice analyzes the last N turns, detects topic shifts, and
  pre-computes a routing hint for the next turn, written into a state blob
  that `assemble_user_context` reads.
- Stateful routing context with zero extra model calls — unique to Lisa's
  turn-anchored opportunistic slice mechanism.

### 2.3 Mode-to-mode direct handoff

- **From `docs/ROUTING_RFC_HANDOFF.md` (future).**
- Currently the synthesized routing FSM allows `default → any mode` and
  `any mode → default`. Direct mode-to-mode transitions (e.g. `coding →
  markdown` for documentation) require authored opt-in.
- Add a `can_handoff_to: [markdown]` front-matter key on `_mode.md` and wire
  it into the routing FSM's `allowed_next_stages` computation.

### 2.4 Model router expansion

- **From `docs/MODEL_ROUTER.md`.**
- The external `lisa-pack-aoi-router` (vendor-backed, ML-based routing) is
  deferred. When ready, it registers under the same
  `lisa.extensions.model_routers` group and activates when
  `router.backend == "aoi-router"`.
- Consider adding more rule conditions to the built-in `lisa-router`:
  `roles_needed`, `mode_id` matching, `budget_remaining` thresholds.

### 2.5 External pack dependency management

- `lisa_packs` is an external repo that some kernel tests transitively import.
  Formalize the boundary: kernel tests must not require external packs. Either
  move shared test fixtures into the kernel, or mock the entry points.

---

## Horizon 3 — Future Expansion (backlog)

### 3.1 Containerization

- **From `docs/wish_list.md` item 1.**
- Containerize MCP servers first, then the REPL. This enables reproducible
  deployments and isolates pack dependencies.

### 3.2 Systemd / init for data-gathering extensions

- **From `docs/wish_list.md` item 2.**
- The `intake/` subpackage is designed for a one-shot process triggered by a
  systemd timer or cron. Provide systemd unit templates and an init wrapper.

### 3.3 Additional transport hosts

- **From `docs/wish_list.md` item 3.**
- WhatsApp / Telegram / Signal hosts. Each is a thin transport adapter over
  `hosts/core`, implementing `TransportCallbacks`. The architecture is ready
  (see `docs/HOSTS_GUIDE.md`) — only the transport adapter needs writing.

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
|---|---|---|---|
| 0.1 | Model-router factory-factory crash | Stabilization | ✅ Done |
| 0.2 | Fork fan-out reliability | Stabilization | 🔴 Critical |
| 0.3 | Test env import breakage | Stabilization | 🟡 High |
| 0.4 | Stale Pyright/lint issues | Stabilization | 🟡 Medium |
| 1.1 | Generalize factory-factory resolution | Hardening | 🟡 High |
| 1.2 | Fork child observability | Hardening | 🟡 High |
| 1.3 | Ollama provider compatibility | Hardening | 🟡 High |
| 1.4 | Incident regression tests | Hardening | 🟡 Medium |
| 2.1 | Proactive routing hints | Completion | 🟢 Medium |
| 2.2 | Slice-driven routing context | Completion | 🟢 Medium |
| 2.3 | Mode-to-mode direct handoff | Completion | 🟢 Medium |
| 2.4 | Model router expansion | Completion | 🟢 Low |
| 2.5 | External pack dependency boundary | Completion | 🟡 Medium |
| 3.1 | Containerization | Future | 🔵 Backlog |
| 3.2 | Systemd for extensions | Future | 🔵 Backlog |
| 3.3 | Additional transport hosts | Future | 🔵 Backlog |
| 3.4 | Stuck-session detection | Future | 🔵 Backlog |
| 3.5 | `/models discover` robustness | Future | 🔵 Backlog |