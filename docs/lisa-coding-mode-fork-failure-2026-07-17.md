# Lisa — Coding-Mode Fork Failure Record

> Captured 2026-07-17. The user asked Lisa to "remember the entire thing and
> write a markdown document of it" while the session was in `coding` mode.
> Lisa stayed in `coding` mode and asked the user to approve a worktree —
> twice — instead of forking to `learn` + `markdown`. This file documents
> the failure and the root cause.

---

## Prompt (user)

```text
I asked it to remember the report it generated from coding mode and it stayed
in coding mode and asked me to approve a worktree, twice! I wanted it to fork
learn and markdown:
excllent; remember the entire thing and write a markdown document of it
```

## Observed response (Lisa)

```text
Approval required for create an isolated coding worktree.

Run /approve 2304829033494bbab5ef2bfe81e3ee33 to continue
or /deny 2304829033494bbab5ef2bfe81e3ee33 to cancel.
```

Issued twice. No `transition_stage` call, no fork fan-out, no memory write,
no markdown file.

---

## Failure Analysis

### What should have happened

The request had **two concurrent goals**:

1. **Remember / learn the report** → belongs in `learn` mode (owns the memory
   tool surface).
2. **Write it out as a markdown document** → belongs in `markdown` mode (owns
   `markdown_forge_document` / `markdown_generate`).

Per `docs/ROUTER_DESIGN.md` §9, the correct dispatch is a **fork fan-out**:

```jsonc
transition_stage(
  target_stage_id = "fork:learn,markdown",
  tool_arguments = { /* two child queries */ }
)
```

The FSM fans out two children — `learn` persists the exchange, `markdown`
forges the document — each running in its own mode with its own tool surface.

### What actually happened

Lisa was in `coding` mode. It did not fork (or even hand off). Instead it
called `coding_create_worktree`, the only write-effect tool available in
`coding` mode, which is `approval_required: true` (see
`lisa_packs/coding/tools/create_worktree/create_worktree.tool.yml`). The host
surfaced the approval prompt. The user did not approve; Lisa re-issued the
same call, producing the second approval prompt.

### Root cause

The two-goal fork detection in `kernel/turn_runner.py ::
_compute_capability_routing_hint` is gated on **both** `learn` and `markdown`
being present in `routing_targets`:

```python
if has_two_goal_learn_and_write and "learn" in targets_lower and "markdown" in targets_lower:
    ...  # emit fork:learn,markdown hint
```

But `routing_targets` for `coding` mode is **`["default"]`** and nothing else.
`synthesize_routing_transitions` in `kernel/ipsar_fsm.py` is intentionally
conservative:

```python
elif mode_config.routable:
    # Routable modes can hand off back to default
    transitions[mode_id] = ["default"]
```

So from `coding`, the guard `"learn" in targets_lower and "markdown" in targets_lower`
is **never true**. No fork hint is emitted. The model has no steering signal
toward `learn`/`markdown` and falls back to the only write tool it can see —
`coding_create_worktree` — which is approval-gated. Hence the worktree prompt,
twice.

### Evidence

| Observation | Evidence |
| --- | --- |
| Session mode was `coding` | User report: "it stayed in coding mode". |
| No fork / handoff attempted | No `transition_stage` call in the transcript; only the worktree approval. |
| Fork hint never emitted | `routing_targets` for `coding` is `["default"]`; the fork branch's `"learn" in targets_lower and "markdown" in targets_lower` guard cannot pass. |
| Worktree prompt surfaced | `coding_create_worktree.tool.yml`: `approval_required: true`, `approval_preview: "create an isolated coding worktree"` — exact text the user saw. |
| FSM synthesis enforces the gap | `kernel/ipsar_fsm.py :: synthesize_routing_transitions`: routable modes hand off only to `default`. `kernel/tests/test_turn_runner.py :: test_routing_fsm_synthesis` asserts exactly this. |

### Why the existing fix did not catch this

The fork-detection fix (see `/memories/repo/routing-hint-fork-detection.md`
and the regression tests in `kernel/tests/test_turn_runner.py`) only covers
the **`default` → fork** path. From `default`, `routing_targets` includes
every routable mode, so the fork hint fires. From any other routable mode
(`coding`, `search`, `ponder`, …), `routing_targets` is `["default"]`,
so the fork hint is silently suppressed. This is a second, distinct failure
surface from the one documented in
`docs/lisa-markdown-stage-failure-2026-07-17.md` (which was the `default`-mode
variant: the stage transition simply never took).

---

## Fix options

### Option A — Widen the routing FSM (preferred)

Allow any routable mode to hand off to any other routable mode (full mesh),
or at minimum to `learn` and `markdown` (the "utility" modes). In
`synthesize_routing_transitions`:

```python
elif mode_config.routable:
    transitions[mode_id] = [m for m in routable_ids if m != mode_id]
```

This makes the existing fork-detection hint fire from `coding` (and every
other routable mode). Matches Adam's "all work items on by default,
externalize later" rule — default to the permissive mesh, defer
restricting specific edges to a later, smaller change.

**Cost:** the routing hint can now name more targets, slightly larger
prompts. **Risk:** a mode could hand off to a semantically-odd peer (e.g.
`search` → `coding`). Acceptable for now; can be tightened per-edge later.

### Option B — Emit the fork hint independent of routing_targets

Make the fork branch in `_compute_capability_routing_hint` emit the hint
even when `learn`/`markdown` are not in `routing_targets`, and let the model
attempt `transition_stage`; the FSM rejects invalid transitions. This avoids
changing the FSM but produces wasted iterations on rejected transitions
(the exact failure class the proactive hint was built to prevent). Not
recommended.

### Option C — Add explicit per-mode routing edges

Author `coding → learn`, `coding → markdown` (and likely
`search → learn`, `ponder → markdown`, …) explicitly. Most precise, but
violates "all on by default, externalize later" and requires a matrix of
edges to maintain. Defer unless Option A proves too permissive in practice.

**Recommendation: Option A.** Smallest change, matches the house pattern,
unblocks the fork hint from every routable mode.

---

## Regression test to add

A test mirroring
`test_routing_hint_forks_for_remember_and_write_markdown` but with
`mode_id="coding"` and `routing_targets=["default"]` (the real
`coding`-mode FSM output), asserting the hint is **non-empty and names
`fork:learn,markdown`** once Option A lands. Today this test would fail,
which is the point.

```python
def test_routing_hint_forks_from_coding_mode() -> None:
    """From coding mode, a remember+write-markdown request must still steer
    to fork:learn,markdown. Regression for the 2026-07-17 coding-mode
    worktree-approval loop."""
    from kernel.turn_runner import _compute_capability_routing_hint
    hint = _compute_capability_routing_hint(
        mode_id="coding",
        query="remember the entire thing and write a markdown document of it",
        capability_manifest=_empty_manifest(),
        routing_targets=["default", "learn", "markdown"],  # post-Option-A
    )
    assert "fork:learn,markdown" in hint
```

---

## Key files

- `kernel/ipsar_fsm.py` — `synthesize_routing_transitions` (the FSM edge
  set; Option A changes one line here).
- `kernel/turn_runner.py` — `_compute_capability_routing_hint` (fork
  detection; unchanged by Option A, but only becomes reachable from
  non-`default` modes once the FSM widens).
- `kernel/tests/test_turn_runner.py` — `test_routing_fsm_synthesis`
  (currently pins the conservative `["default"]`-only edge set; must be
  updated alongside Option A) and the fork-hint regression tests.
- `lisa_packs/coding/tools/create_worktree/create_worktree.tool.yml` — the
  approval-gated tool Lisa reached for in the absence of a fork hint.
- `docs/ROUTER_DESIGN.md` §9 — fork fan-out contract.
- `/memories/repo/routing-hint-fork-detection.md` — the prior
  (default-mode-only) fix record.
- `docs/lisa-markdown-stage-failure-2026-07-17.md` — the related
  default-mode failure (stage transition never took).

---

## Lesson

The fork-detection hint is only as good as the routing FSM edges that feed
it. A hint guarded on `"learn" in routing_targets and "markdown" in
routing_targets` silently no-ops from any mode whose only route is back to
`default`. Proactive routing hints must be validated against **every**
entry mode the user can reach, not just `default` — otherwise the model
falls back to whatever write tool the current mode happens to expose, and
if that tool is approval-gated, the user sees a misleading approval prompt
instead of the intended fork.
