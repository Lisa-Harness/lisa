# Residency RFC — Residency by Allocation, Coherence by Address

> **Status:** Active. Carve-out opened 2026-10-07 in
> `docs/CODING_WORKFLOW.md` § "Declared carve-outs" (authorized by the
> operator). Phase 2 (turn-boundary residency) implemented 2026-10-07:
> replay working-set budget (`84d0984a` mechanism, wiring + knob +
> core-default commits) — see "Phased plan" below for per-commit receipts.
> **Design thread:** "residency by allocation, coherence by address" —
> one principle, three rules.

## Current approval and cache behavior

Model-loop approval boundaries now save a kernel-owned
[`TurnContinuation`](../../kernel/src/kernel/turn_continuation.py).
The runtime persists it by content address before returning the approval
panel. An approval resumes the saved dispatch cursor, including the remaining
calls in the model's batch, without replay, prompt assembly, entry operations,
or a synthetic continuation query. The kernel revalidates approval ownership,
expiry, mode configuration, and the normal registry authorization.

Policy-approved actions execute inside that same loop. The
`MAX_AUTO_APPROVED_ACTIONS` allowance defaults to ten; reaching it saves the
same continuation. An explicit approval grants another allowance while
retaining consumed counters, receipts, and cumulative turn budgets.
Consumed or expired approvals cannot replay an action.

Lifecycle guidance is appended initially and when actionable guidance changes,
at complete tool-batch boundaries. Counters no longer rewrite the system
message. Provider residency survives approval checkpoints; previously reduced
evidence is not expanded again on each completion. A genuine mode handoff
still refreshes the active capability context and creates a cache boundary.

Exact model requests are persisted before dispatch, with only their result
references retained in memory and recorded under `turn_input.calls`. Read those
references through the thread readout to recover the request.

This does not yet replace iteration/step-cap continuation or entry-operation
approval recovery.

## Whole-request allocation

The kernel now fits each primary turn-loop request through
[`fit_request_context`](../../kernel/src/kernel/context/request_budget.py).
There is no fixed tool-result window share. The allocation accounts for
serialized messages (including system text and tool-call arguments), schemas,
structured-output instructions, and the actual per-call completion ceiling.
It reserves `CONTEXT_REQUEST_MARGIN_SHARE` of the declared model window
(default 5%) for counting/framing uncertainty. If required input alone crowds
the output ceiling, that ceiling is tightened explicitly in the recorded
budget; required content is never silently dropped.

Full permitted results remain resident while the complete request fits.
Under pressure, the allocator searches a fitting result-byte allowance and
recounts the projected request, including reference and inventory overhead.
It additionally recovers up to one margin's worth of append headroom to
avoid reducing history on every subsequent append. Results allocate newest
first, reserving space for older references before spending on full payloads.
Previously referenced results do not automatically re-expand. This is
recency-based residency, not a general access-tracked LRU implementation.
Truncation retries cannot borrow space already allocated to input.

The SDK tokenizes the serialized request locally with the registered
`tiktoken` encoding for recognized model IDs. Provider-specific framing and
schema translation still make this a request estimate, not an exact wire
count. Unknown tokenizers use an explicitly labelled UTF-8-byte/3 estimate.
Provider-reported input usage is compared with the count; underestimates
increase a per-model multiplier for subsequent requests. That multiplier
survives approval checkpoints. Estimates and the safety margin do not
guarantee that every unrecognized tokenizer or multimodal payload fits.

Exact request snapshots include `context_budget`: window, requested/actual
completion ceiling, safety margin, pressure headroom, required/input tokens,
result-byte allowance, counting method, and reduction flag. Runtime logs
also compare counted and observed input tokens.

Replay no longer pre-reduces each entry against the SDK's 64KB default.
The host replay ceiling defaults to zero (no independent cap); the core
default session likewise omits `context.history.replay_max_bytes`. Explicit
host/session ceilings still apply, choosing the tighter finite limit.
The p920 example session's 24KB override has been removed. Lane budgets
for other prompt-assembly sections remain; this change fills remaining
capacity with tool results rather than redesigning those sections.

The rules below describe the original RFC. Whole-request pressure allocation
supersedes its threshold-first stubbing and turn-boundary-only allocation
proposals. Durable projection restoration across ordinary new turns,
general access-tracked LRU, non-tool history reduction, and provider-native
exact counting remain follow-up work.

## Principle

Context is a cache over the lossless store; coherence is by identity, not
repetition. "Lossless" means every byte persists and is recoverable by
address (**true today** — the thread store, result objects, and turn refs
already guarantee this). "Efficient" means residency is bounded (**false
today** — tool results re-enter context at whatever fidelity they were
first rendered, and nothing recomputes the resident working set at the
turn boundary).

No compression, no model-authored summaries, no information destroyed —
ever. The two sides of the principle:

- **Residency by allocation** — what sits in context is decided by a
  budgeted allocator once per turn boundary, not negotiated per iteration
  by whatever happened to be appended.
- **Coherence by address** — anything not resident is still reachable,
  by a deterministic ref the model already knows how to expand
  (`r<sha256>`, `t<turn>.m<index>`, thread manifest refs). Identity
  carries the pointer; the resident copy is a cache, not the record.

## The three rules

### R1 — Every event has an address; nothing large enters context unaddressed

Already half-exists: result-object refs (`r<sha256>`, content-addressed,
[sdk/thread/_results.py](../../sdk/src/sdk/thread/_results.py)), turn refs
(`t391.m20`, minted in [_readout.py](../../sdk/src/sdk/thread/_readout.py)),
and `thread_search` returning excerpt+ref
([_search.py](../../sdk/src/sdk/thread/_search.py), `_EXCERPT_CHARS = 240`).

The convention goes backward into the write path: any tool result (or
replayed event) is either small enough to live resident in full (below a
threshold) or **stubbed** — a one-line deterministic entry, ~25–30 tokens,
carrying the ref the model would need anyway:

```text
stub t391.m20 tool=coding_read_file bytes=48,212 sha=r3f9a… read_more=thread_read_content
```

This is **not** a summary, and no LLM touched it. It is a
table-of-contents entry — deterministic, derived from the stored object's
own metadata. `abbreviation_template` on read tools is the same knob,
generalized from a special case (currently opt-in per `.tool.yml`,
disabled by default) into the default for anything over threshold.

### R2 — Residency is recomputed once per turn boundary, not negotiated per iteration

At turn start, the replay allocator (the same lane machinery that already
budgets retrieval/memory/reflection — [`ContextBudgetAllocator`](../../kernel/src/kernel/context/context_budget.py)
with the host 8,000-token split in [runtime/config.py](../../runtime/src/runtime/config.py);
Lisa invented budgeted residency and then exempted the one path that grows
fastest: the **replay/tool-result path**) computes the resident working
set:

- task frame,
- the receipts ledger,
- the last-N events verbatim,
- every stub row for the rest.

Within a turn, **append-only** — stubs occupy the slot the full result
would have occupied rather than replacing an appended message, so nothing
is ever rewritten. That property is what makes the hard case below (a
result that started large and stayed large) tractable: the resident
working set is recomputed at the boundary, but no in-flight message is
ever mutated.

### R3 — State is not history

A **state ledger** — deterministic, built from effect receipts
([`EffectReceipt`](../../kernel/src/kernel/models.py), captured in
[_dispatch.py](../../kernel/src/kernel/turn_phases/_dispatch.py)) — is
resident every iteration, at the bottom of the context where attention
lives (the Manus recitation insight, but free: it is generated text,
zero model tokens, O(work-done) not O(content-seen)). Payload: stage,
budget remaining, writes landed/attempted, active refs, plan excerpt.

This unifies two things Lisa already builds into one standing payload
instead of one-shot arrivals:

- the **cap-pause diagnostic**
  ([`_build_cap_pause_diagnostic`](../../kernel/src/kernel/turn_runner/__init__.py)),
  which today appears only when the turn stalls;
- the **lifecycle briefing**
  ([`_build_budget_briefing`](../../kernel/src/kernel/turn_runner/__init__.py)),
  which is rebuilt per iteration as a dynamic system segment.

## What exists vs. what is missing

Recon 2026-10-07 (three parallel codebase maps, verified against source).
This grounds each rule in the real seam and names the gap precisely.

### R1 — stub addressing/projection

| Piece | Status | Where |
| --- | --- | --- |
| Content-addressed result refs (`r<sha256>`) + lossless persist/load | **exists** | `_results.py:13–60` |
| Budget-based inline vs. preview/pointer projection | **exists** | `project_tool_messages()` — newest-first against a model-sized `budget_bytes` (`tool_context_budget_bytes`), `_HEAD_CHARS=600`, `_TAIL_CHARS=300`; the 8KB per-result cap was removed 2026-10-09 |
| Turn refs `t<turn>.m<index>` minted on readout | **exists** | `_readout.py:160–170` |
| `thread_search` → `{ref, excerpt}` | **exists** | `_search.py:108–126` |
| `abbreviation_template` knob | **exists, opt-in, default off** | `models.py:196–215`, `_dispatch.py:210–225` |
| **Missing** | A single deterministic **stub envelope format** (one line, ~25–30 tokens, carrying the ref) that both live dispatch and replay emit for over-threshold results. Today "stub" is two shapes (preview/pointer envelope, retrieval page pointer) with different heads, and the abbreviation knob is off by default. | — |

**The R1 gap is format + default, not plumbing.** The persist/load/address
machinery is done; what is missing is one canonical stub row and making it
the default over-threshold shape.

### R2 — turn-boundary residency

| Piece | Status | Where |
| --- | --- | --- |
| Lane budget machinery (6 lanes, allocator, tighten-only replay caps) | **exists** | `context_budget.py`, `context_assembly.py:594–645` |
| Host 8,000-token lane split | **exists** | `runtime/config.py:376–430` |
| Per-turn allocator build at context assembly | **exists** | `assemble_dynamic_user_context` |
| Replay of unfinished task at turn start | **exists** | `_replay.py:128–165` → `build_open_task_messages`, consumed at `turn_runner/__init__.py:2015–2035` via `prior_messages()` |
| Tool results flow through `project_tool_messages` on replay | **exists** | `_replay.py` calls `project_thread_messages` |
| **Missing** | Residency computed **by the allocator at the turn boundary** as a working set (last-N verbatim + stubs for the rest), rather than the current pass-through where replayed results enter at whatever fidelity they carry and the lane allocator never sees them. The replay path is the one path that grows fastest and is the one the 8k lane budget exempts. | — |

**The R2 gap is the working-set contract.** The allocator exists; the
replay lane exists; they are not the same object. R2 is making the replay
allocator's per-turn recompute own the boundary between "last-N verbatim"
and "stub rows", with the append-only within-turn invariant.

### R3 — resident state ledger

| Piece | Status | Where |
| --- | --- | --- |
| `EffectReceipt` (effect, capability_id, artifact_id), recorded on success | **exists** | `models.py:1863–1890`, `_dispatch.py:429–559` |
| Receipts surfaced on `TurnResult` + persisted to thread record | **exists** | `turn_runner/__init__.py`, `THREAD_STORE.md:63–80` |
| `ExecutionLedger` (model/capability calls, tokens, elapsed) | **exists** | `execution_ledger.py:17–112` |
| `write_attempted` vs `write_landed` distinction | **exists** | `turn_phases/_child_turn.py:205–214` |
| `WorkflowPosition` (active stage/step) | **exists** | `workflow_position.py:105–117` |
| Per-iteration dynamic system segment update path | **exists** | `turn_runner/__init__.py:5800–5850` (`sys_segments.set_dynamic`) |
| **Missing** | A **deterministic per-iteration render** that folds receipts + execution ledger + workflow position + active refs into one standing ledger block, resident every iteration at the bottom of context. No `active_refs` ledger exists. Today the equivalent signal arrives as one-shot events (cap-pause on stall, lifecycle briefing rebuilt per iteration) rather than as a single unified standing payload. | — |

**The R3 gap is the unified deterministic render + an active-refs
source.** All inputs exist except a live `active_refs` set; the render
into the standing per-iteration slot does not exist.

## Carve-out declaration

Recorded in `docs/CODING_WORKFLOW.md` § "Declared carve-outs" as
"Context residency by allocation (opened 2026-10-07, authorized by the
operator)". That entry is the carve-out's receipt; the scope below is its
copy for review and the two are kept identical.

---

**Context residency by allocation (opened 2026-10-07, authorized by the
operator).** Scope:

- `sdk/src/sdk/thread/_results.py` — stub envelope projection, canonical
  over-threshold stub row.
- `sdk/src/sdk/thread/_readout.py` — ref surface for the stub row.
- `sdk/src/sdk/thread/_search.py` — excerpt+ref shape the stub aligns to.
- `runtime/src/runtime/thread/_replay.py` and `runtime/thread/` — replay
  working-set projection at the turn boundary.
- `runtime/src/runtime/context_assembly.py` — replay lane wired into the
  allocator; the working-set recompute at turn start.
- `kernel/src/kernel/context/context_budget.py` — the replay/residency
  lane budget (a new lane is a scoped allocation-rule change named here,
  not an unrelated routing rule).
- `kernel/src/kernel/turn_runner/__init__.py` — turn-start working-set
  assembly from `prior_messages()`; the per-iteration standing `State`
  ledger render via the existing `set_dynamic` segment path; the
  append-only within-turn invariant.
- `kernel/src/kernel/turn_phases/_dispatch.py` — receipt capture feeding
  the ledger; the `active_refs` source.
- `kernel/src/kernel/execution_ledger.py`,
  `kernel/src/kernel/workflow_position.py`,
  `kernel/src/kernel/session_plan.py` — ledger read sources (no signature
  changes to anything another package imports).
- `mcp_server/src/mcp_server/thread/server.py` — `thread_read_content` /
  `thread_search` result shapes the stub points at.

**Purpose:** make context residency bounded by recomputing a budgeted
resident working set at each turn boundary (last-N events verbatim, stub
rows for the rest, a standing receipts-derived state ledger), while every
byte stays persisted and recoverable by content/turn address. No
compression, no model-authored summary, no in-place rewrite of any
in-flight message.

**Rules (same as every carve-out):** each commit (1) names the turn-loop
behavior it changes, (2) ships a test that fails before and passes after
(same-commit, red/green), (3) does not widen scope to unrelated
signatures, and (4) keeps the full verification loop green in its
entirety.

**Closes when:** these trees settle on a behavior the operator ratifies,
or explicitly on operator order — record the closure in the workflow doc.

---

## Phased plan

Landing order **R2 → R1 → R3**. R2 is highest-risk and fixes the contract
(the stub format + the append-only slot semantics) that R1's envelope and
R3's ledger both build on. Each phase is a chain of single-behavior,
red/green commits.

### Phase 2 (first) — Turn-boundary residency working set

1. **Residency working-set contract at turn start** — *shipped 2026-10-07
   (`84d0984a`).* `build_open_task_messages` gained `budget_bytes`; a
   final pass over the assembled replay bounds total tool-result bytes by
   collapsing the oldest results into recoverable ref-stubs via the
   existing `_compact_groups` evidence-inventory machinery. Defaults
   preserved prior behavior (no caller passed a budget in that commit).
   *Red/green:* `test_replay_working_set_is_bounded_by_budget_bytes` /
   `test_replay_working_set_unbounded_without_budget`
   (`runtime/tests/test_thread_replay.py`).
2. **Append-only within-turn invariant** — *verified structurally true,
   no commit needed (2026-10-07).* The in-turn loop never rewrites an
   appended tool result; the only in-place rewrites are `messages[0]`
   keyed dynamic segments (lifecycle briefing, routing hint, handoff
   context, terminal prompt). Confirmed by audit of all nine tool-append
   sites and four in-place rewrite sites in `kernel/turn_runner/__init__.py`.
3. **Replay budget knob + live wiring** — *shipped 2026-10-07.* The
   replay working set is bounded by a three-layer budget (no render-lane
   was added: the section allocator never consumes replay messages, so a
   lane there would be an unused lane — the knob family lives in config
   and the consumer is the replay path itself):
   - Host ceiling: `HostConfig.context_budget_replay_max_bytes`, default
     65536, env `LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES`.
   - Session tighten-only knob: `context.history.replay_max_bytes` on the
     v2 `ContextHistoryPlan` (absent = host ceiling; cannot raise past it).
   - Core default session: 32768 with a note that 16384 is adequate most
     of the time.
   - Wiring: `prior_messages` in `turn_hooks_builder.py` threads the
     effective budget into `build_open_task_messages`.
   *Red/green:* `kernel/tests/test_session_plan.py::
   test_compile_session_plan_accepts_replay_max_bytes`;
   `runtime/tests/test_replay_budget_wiring.py` (bounds-to-knob,
   core-default-bounds, absent-knob-verbatim, knob-cannot-exceed-ceiling).

### Phase 1 (second) — Canonical stub envelope

1. **Recognize every graduated stub scale** — *shipped 2026-10-07.* The
   canonical envelope already exists (`{ref, abbreviated, total_chars,
   head, tail, omitted_chars, expand, find}` — one grammar, emitted by
   both live dispatch and replay); the real bug was that re-projection
   recognized only the two exact endpoint scale pairs, so a
   graduated-pressure stub evicted by the working-set compactor was
   persisted as fresh content — a stub-of-a-stub whose ref expanded to
   preview JSON instead of original bytes. Recognition now rebuilds the
   preview at the stub's own scale. *Red/green:*
   `runtime/tests/test_thread_replay.py::
   test_inventory_refs_expand_to_original_bytes_not_stub_json`.
2. **Pointer complete stored pages** — *shipped 2026-10-07*
   (Lisa-Harness/lisa#5, pre-existing at HEAD). Complete retrieval pages
   (`has_more` false) over the inline ceiling but under the turn ceiling
   fell through a no-pointer window and were persisted as copies of
   already-stored bytes (two expansion hops). The pointer now fires
   whenever a complete stored page cannot stay resident inline.
   *Red/green:*
   `sdk/tests/test_thread_search.py::
   TestProjectionRetrievalPages::test_over_threshold_page_pointers_without_pressure`.
3. **Allocate residency newest-first** — *shipped 2026-10-08.* Projection
   decided residency from a turn-wide pressure flag over all raw tool
   bytes, so once a turn's raw history passed the budget every new result
   was previewed and every expansion pointered to its own ref, even
   though eviction had freed the budget. Results are now allocated
   newest-first against the resident budget at one fixed preview scale;
   the oldest results collapse into the inventory. *Red/green:*
   `sdk/tests/test_thread_search.py::TestProjectionNewestFirstAllocation`.
4. **Stub as the default over-threshold shape** — *not a change; the RFC
   premise was wrong.* The over-threshold default stub is already
   enforced by projection for every tool; `abbreviation_template` is a
   separate opt-in same-turn repetition cache, not the stub mechanism.
   Making it default would be refactor without a named bug — dropped
   from the plan.

### Phase 3 (third) — Standing state ledger

1. **`active_refs` source** — *shipped 2026-10-07.* `_TurnState.active_refs`
   captures every result-object (`r<sha256>`) and thread (`t<turn>.m<idx>…`)
   reference tool outputs carry, at dispatch time (word-bounded regexes
   mirrored from the sdk ref grammar). *Red/green:*
   `kernel/tests/test_effect_receipts.py::
   test_dispatch_captures_active_refs_from_stub_envelopes`.
2. **Deterministic per-iteration ledger render** — *shipped 2026-10-07.*
   Writes landed/attempted (from effect receipts) and refs held render as
   ledger lines inside the standing `## Turn Lifecycle` block via the
   existing `N_LIFECYCLE_BRIEFING` keyed segment — one control channel,
   replaced per iteration, zero model tokens, O(work-done). Omitted
   entirely on receipt-less, ref-less iterations so early-iteration
   briefings stay byte-identical to the pre-ledger format. Continuous
   token budgets stay deliberately omitted (the documented
   "model cannot convert '50k tokens left' into a concrete decision"
   contract). *Red/green:*
   `kernel/tests/test_control_channel.py::
   test_budget_briefing_carries_state_ledger_lines` /
   `test_budget_briefing_ledger_lines_omitted_when_empty`.
3. **Unify cap-pause + lifecycle briefing into the ledger** — *verified
   already satisfied by the channel architecture; no commit needed.* The
   stuck-in-stage nudge and cap-pause diagnostic already ride the same
   Phase 7 control channel (guidance lines and the empty-content path
   into the same `N_LIFECYCLE_BRIEFING` segment), the stage-budget line
   rides the count lines, and the new ledger lines complete the payload
   (stage, writes, refs, receipts). Restructuring further would be
   refactor without a named bug — out of scope under the freeze.

## Risks and mitigations

- **Stub-expanded loops.** The
  `2026-10-06 turn-557/558 micro-window loop` showed zero-omission
  previews *inflating* the context they bound. Mitigation: the stub row
  must be strictly smaller than any verbatim result it replaces, and the
  working set reserves budget for it — verified by the Phase-1 red/green
  test measuring emitted bytes.
- **Ref-expansion failure surfacing.** A stub is only useful if
  `thread_read_content` on its ref works. Mitigation: stub rows name the
  `read_more` path; the expansion path is covered by existing
  `read_content` reference parsing (`_readout.py:470–555`) and its tests.
- **Scope creep into adjacent frozen code.** The per-iteration slot
  already exists (`set_dynamic`); R3 renders into it rather than adding a
  mechanism. Mitigation: carve-out names `execution_ledger.py` /
  `workflow_position.py` / `session_plan.py` as **read sources only** —
  no signature changes.

## Exit / ratification criteria

- A deep verification turn no longer re-reads its own prior tool results
  at bounded fidelity: older results present as stub rows, expandable by
  ref on demand.
- Resident context footprint per turn is bounded by the allocator and
  measurable in [docs/TOKEN_BUDGET.md](../TOKEN_BUDGET.md) terms —
  the 97%-prompt baseline is the number this design exists to move.
- Every commit lands with its red/green receipt; the full verification
  loop stays green in its entirety at each step.
- The carve-out is closed in `docs/CODING_WORKFLOW.md` when the operator
  ratifies the settled behavior.
