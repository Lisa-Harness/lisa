# Workflow system cleanup

This document records what the workflow machinery actually does, measured
against 1,047 real turns (2026-08-23 → 2026-10-27, 915 model turns): what is
already shipped, and the ordering proposed for the remainder.

Read [mode and workflow routing](ROUTER_DESIGN.md) for the current contract and
[model taxonomy](MODE_TAXONOMY.md) for mode design. This guide covers only the
workflow-management surface and what remains to be done on it.

## Status

The status table below is the authority on what is done; everything below the
table is either shipped behavior or a proposal, labelled as such.

| Change | State |
| --- | --- |
| Derivation correctness — one receipt per step, out-of-order work, plan anchoring, plan text on the creating turn | Shipped (`89c4655a`) |
| `/workflow status\|conclude` — a plan can be ended deliberately | Shipped (`89c4655a`) |
| Staged-workflow FSM removed end to end | Shipped (`89c4655a`) |
| Receipts scoped by dispatching mode | Shipped (`08507538`) |
| Diagnostic marker matches what the documentation promises | Shipped (Phase A) |
| ~~Task-gate contract resolution by active mode~~ | **Withdrawn — premise was wrong; see Phase B** |
| Mid-turn multi-step plans reach the operator | Shipped (Phase C) |
| Planner retired as a mode | Withdrawn — the planner works; retiring it would delete the mechanism |
| Yields revisited | Proposed — Phase D |

## What the measurements say

The workflow system was built through several orthogonal strategies. Measured
against real use, most of them never carried work:

| Mechanism | Usage in 1,047 turns |
| --- | --- |
| `## Plan` workflow (planner) | 56 plans produced; **2 visible**. See Phase C — the output is discarded |
| `## Workflow progress` footer | 20 turns — 19 of them the stuck plan |
| Staged FSM (`transitioned_to`) | 13 events, all mode-level re-entry; **no installed pack ever shipped stage files** |
| Ad-hoc step compilation | 0 — it was double-gated unreachable; deleted |
| `route_coordinator` | 0 — the shipped host never wires it |
| Mode handoffs (`handoff_to`) | 203 turns — the one routing mechanism that carries real work |
| Effect receipts | Every dispatched capability — the load-bearing evidence |

The conclusion: **handoffs and receipts are the working system.** Plans are
produced often (56) but shown twice, because the mid-turn path that stages them
for display is not wired — see Phase C. The staged FSM, ad-hoc compilation, and
the route coordinator measured to zero and are gone or proposed for removal.

## Phase A — the diagnostic marker (shipped)

The highest-value item found while verifying the plan, and a correctness bug
rather than a cleanup.

`docs/CODING_WORKFLOW.md` told the operator:

> The match is case-insensitive and tolerates interior whitespace and a missing
> space after the colon.

The implementation did neither — detection compared against the literal
`"[task_goal: diagnostic]"` while *stripping* used a whitespace-tolerant regex:

| Operator typed | Recognized | Stripped before the model saw it |
| --- | --- | --- |
| `[task_goal: diagnostic]` | yes | yes |
| `[task_goal:diagnostic]` | **no** | yes |
| `[task_goal:  diagnostic]` | **no** | yes |
| `[ task_goal : diagnostic ]` | **no** | yes |

The worst shape available: the declaration was removed from the message, so it
looked accepted, while the turn ran as `implementation` and blocked — and the
block message then asked the operator to declare a diagnostic they had just
declared. 38 of 102 task-gate blocks were read-only work in a write-contracted
mode, which is precisely what the marker exists to resolve.

**Shipped:** detection and stripping share one pattern
(`_TASK_GOAL_MARKER_RE`), so a declaration cannot be removed without being
read. An invariant test asserts that directly instead of enumerating spellings,
and a value other than `diagnostic` fails closed with a log line. The block
message names the syntax to type. The regression tests were verified to fail
against the previous implementation.

## Phase B — withdrawn: the contract already follows the handoff

This phase was proposed on a misreading of the evidence, and is recorded here
so the same conclusion is not re-derived later.

The claim was that the completion contract resolved against the *entry* mode
while `action_tool_ids` followed handoffs, so a turn entering in `default` and
handing into `coding_execute` would be gated by the wrong mode. The supporting
number was that 25 of 102 `task_completion` blocks carry `mode_id: default`.

Both parts were wrong:

- **The contract already follows the handoff.** `mode_id` and `mode_config` are
  rebound at the handoff site (`mode_config = target_mode_config`,
  `kernel/turn_runner/__init__.py`), alongside the `action_tool_ids` refresh.
  The contract application and the evidence resolver both receive the rebound
  values.
- **`mode_id` on a thread entry is the entry mode by design.** It labels the
  turn, not the mode that finished it, so a `default`-labelled block is not
  evidence of misresolution. Reading the block text settles it: `"This task has
  no owned worktree with a current checked commit."` is the *verifier's* answer
  for `executor` — the target mode's contract ran correctly.

A regression test for the proposed behavior passed without the change, which is
what exposed the error: a test that cannot fail is not evidence.

**What the 25 `default` blocks actually show** is that those turns entered in
`default`, handed into a checked-commit mode, did not produce a worktree, and
were correctly blocked. That is the contract working, not failing.

The lesson for Phases C and D: entry-mode attribution on a thread entry is weak
evidence about which mode did the work. `EffectReceipt.mode_id` (shipped in
`08507538`) is the field that answers that question, and any further analysis
should read it rather than the entry label.

## Phase C — the planner works and its plans are thrown away

This phase was proposed as "retire the planner as a mode — tidying, 2 plans in
2 months". That was wrong. The correct finding is worse and far more useful.

### What the number actually was

Counting plans by the presence of a `## Plan` block at the start of a turn's
response gives 2. That measured **visibility**, not execution. Counting plans
by the kernel's own signal — a `transition_stage` whose rationale is
`"Kernel-driven handoff from the plan result"`, which only the kernel writes
after `plan_workflow` returns steps — gives:

| Measurement | Count |
| --- | --- |
| Plans visible to the operator (`## Plan` at response start) | **2** |
| Plans that were actually produced | **56** |
| Plans produced and never shown | **54** |

The planner is not underused. It runs constantly and its output is discarded.

### Why it is discarded

Two paths stage a rendered plan for the user, and only one of them is wired:

- **Turn start.** `execute_entry_phase` returns `plan_prefix`, which the
  turn-start helper stores on `state.plan_prefix`; `_append_post_turn_content`
  then prepends it to the response. This path works — it produced the 2 visible
  plans.
- **Mid-turn handoff.** `_run_handoff_entry_operations` calls
  `_record_turn_plan_position`, which sets `state.turn_plan_position` and
  nothing else. It never assigns `state.plan_prefix`, so the prepend never
  runs.

Every escalation into the planner happens mid-turn, so every plan it produces
takes the unwired path. `_record_turn_plan_position` even stores the plan text
on the position (`plan_text=entry_result.output_text`), so the content exists —
it is simply never read for display.

### This is the same defect as the original stuck footer

The footer in the original report persisted because completion could not credit
a mid-turn mode. This is its sibling: presentation cannot see a mid-turn plan.
Both come from a turn-start assumption baked into mid-turn handling, and both
were invisible for the same reason — the plan was real, the bookkeeping was not.

### What the forcing statistics actually show

34 of the 48 transitions into the planner are kernel-forced, which reads like
the enforcer fighting an unwilling model. It is not: 40 of 40 forced turns then
produced a kernel-driven handoff, i.e. the planner ran and returned steps. The
forcing is doing its job. The earlier conclusion that "escalation is forced
against a model that does not want it" was drawn from turn-level response text,
which as established above cannot show a mid-turn plan at all.

### Shipped

`_record_turn_plan_position` now stages `state.plan_prefix` for plans with more
than one step, so `_append_post_turn_content` prepends them exactly as the
turn-start path does. Single-step plans are deliberately not prefixed: a
one-line chain describes the handoff the model is already making, so rendering
it adds ceremony without information — the plan still lands on
`turn_plan_position` and still drives the handoff.

Covered by two tests: a multi-step mid-turn plan must begin the response, and a
single-step one must not. The first was verified to fail against the previous
implementation, per the Phase B lesson.

### Decision taken

Render only multi-step plans. Rendering all 54 would have changed every
escalated response; rendering none would have kept the blindness. Multi-step is
the boundary where a plan carries information the operator did not already
have.

## Phase D — decide what yields are for

Measured: 79 yields across 915 model turns (8.6%), and only 19 of 79 are
followed by a resume phrase (24%). A yield asks the operator a question; three
quarters go unanswered and the turn is effectively abandoned.

Two coherent endings, and the choice is a design decision rather than a bug
fix:

1. **Yields become rare.** Keep `yield` for cases where the kernel genuinely
   cannot proceed, and remove the paths that yield as a soft landing
   (stuck-in-stage nudge, budget fallback question).
2. **Yields become resumable.** Make the resume contract first-class — the
   question is durable, `/resume` is offered, and a reply continues the turn
   rather than starting a new one.

Not proposed for implementation yet: this needs a decision about what a turn
is allowed to ask before it is worth code.

## Non-goals

- **No mutable workflow store.** The thread is the source of truth; the view is
  derived every turn. Both shipped changes preserve this.
- **No rewriting of history.** Legacy receipts carry no `mode_id` and legacy
  entries carry `transitioned_to`; both are read as empty/fallback so existing
  threads keep deriving as they did. Attribution is repaired going forward.
- **No new workflow state.** `/workflow conclude` added one thread-entry
  provenance key, nothing else.

## Verification bar

Each phase lands with its own regression test and the full gate:

```sh
uv run --extra dev ruff check        # lint
uv run --extra dev ruff format --check
uv run --extra dev python -m pytest kernel/tests runtime/tests sdk/tests
```

Current baseline: 2,906 tests, 3 skipped, 1 xfailed. A phase that breaks an
untouched test is out of scope by definition — see
[the coding workflow](CODING_WORKFLOW.md#test-assurances).
