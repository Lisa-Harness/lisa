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
| **Task-gate contract resolution by active mode** | **Proposed — Phase B** |
| Planner retired as a mode | Proposed — Phase C |
| Yields revisited | Proposed — Phase D |

## What the measurements say

The workflow system was built through several orthogonal strategies. Measured
against real use, most of them never carried work:

| Mechanism | Usage in 1,047 turns |
| --- | --- |
| `## Plan` workflow (planner) | 2 plans. One was a single step; one left a footer stuck for 22 turns |
| `## Workflow progress` footer | 20 turns — 19 of them the stuck plan |
| Staged FSM (`transitioned_to`) | 13 events, all mode-level re-entry; **no installed pack ever shipped stage files** |
| Ad-hoc step compilation | 0 — it was double-gated unreachable; deleted |
| `route_coordinator` | 0 — the shipped host never wires it |
| Mode handoffs (`handoff_to`) | 203 turns — the one routing mechanism that carries real work |
| Effect receipts | Every dispatched capability — the load-bearing evidence |

The conclusion: **handoffs and receipts are the working system.** Plans are a
rare, deliberate layer on top. Everything else measured to zero.

## Phase A — the diagnostic marker (shipped)

The highest-value item found while verifying phases A–D, and a correctness bug
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

## Phase B — resolve the task contract by the mode that is active

Second correctness item, same family as the receipt-mode bug just fixed.

A thread entry records the mode the turn *started* in. The completion contract
is resolved from `mode_config` — also the entry mode. When a turn hands off
mid-turn, the contract should follow the mode that ended up doing the work.

Measured: of the 102 `task_completion` blocks, **25 are attributed to
`default`** — a mode that declares no task contract at all. Those are turns
that entered in `default` and handed off into `coding_execute` (or `executor`),
where the contract belonged to the target mode. 14 of the 38 no-write blocks
handed off mid-turn, 12 of them to `coding_execute`.

The receipt fix in `08507538` established the principle for evidence; this
extends it to the contract. `state.active_mode_id` already tracks the live
mode through every handoff, so the change is small: resolve the completion
contract against the active mode's config rather than the entry mode's.

**Open question for review:** whether a turn that spans an
`implementation`-contracted mode and a `diagnostic` one should be gated at all,
or gated on the write receipts only. I lean toward gating on receipts — which
is what the contract already does at its core — and treating the mode's
declaration as the default for a single-mode turn.

## Phase C — retire the planner as a mode

Lower value than A and B, and the reason to sequence it last.

`planner` is an action mode whose entire body is "call `plan_workflow`, do not
call `transition_stage`". It exists to give a *model call* a place to happen
before a handoff. Two plans in 1,047 turns means the indirection is paid on
every turn's catalog, routing-hint, and schema construction for a path used
twice.

Proposal: `plan_workflow` stays an operation; the mode goes. `/plan` becomes a
host command that calls the operation directly and injects the resulting plan
as the turn's query, exactly as `entry_op_kind: routing` does today. The
routing-hint escalation branch that targets `planner` becomes a branch that
invokes the operation.

Deliberately last: it touches the entry-op path that Phase A and B depend on
being stable, and it is the only item here that is tidying rather than a fix.

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
