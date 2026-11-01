# Enter-mode consolidation (F1)

Spec for the largest remaining item of the 2026-10-30 turn-loop structural
review. The loop-body refactors are landed (F5+F3 resume dataclass, F2
`LoopControl` union, F4 `_TurnState` by reference, F7 handoff-cap predicate,
F8 `_next_loop_action`); this note makes the F1 spec durable before
execution. Line anchors are verified at commit `fc0824`.

## The defect: two implementations of "enter a mode"

One concept — make `target` the turn's active mode — is implemented twice,
independently:

| | Turn start | Mid-turn handoff |
| --- | --- | --- |
| Entry point | `_run_turn_entry_sequence` (`turn_runner/__init__.py:8018`) | `_execute_mid_turn_handoff` (`:5737`) |
| Entry operations | `execute_entry_phase` (`turn_phases/_entry.py:83`) | `_run_handoff_entry_operations` (`:5431`) |
| Pre-turn operations | `run_mode_pre_turn_operations` (in the entry sequence) | `_run_mid_turn_pre_turn_operations` (`:5564`) |
| Model pin / routing context | entry-phase path | `_apply_mode_routing_context` (`:5696`) via `_resolve_mode_model_pin` (`:5646`) |
| Tool surface / manifest | Phase-1 assembler | `_rebuild_tool_schemas_for_mode` + `_recompile_manifest_and_refresh_user` (`:5085`) via `_refresh_user_message_capability_block` (`:5028`) |
| Plan recording | `state.plan_prefix` from `entry_result.plan_prefix` | `_record_turn_plan_position` (`:5375`) — sets `turn_plan_position`, never `plan_prefix` |
| Workflow position | initial `WorkflowPosition` seed | `_advance_handoff_workflow_position` (`:5606`) |

Each path grew its own copy of every entry concern. The copies agree today only
because each divergence was patched locally after it fired.

## Evidence: three incidents from one root cause

All three are "a turn-start assumption baked into mid-turn handling"
([workflow cleanup](../WORKFLOW_CLEANUP.md)):

1. **Stuck progress footer** — completion could not credit a mid-turn mode.
2. **Discarded plan** — `_record_turn_plan_position` set `turn_plan_position`
   but never `plan_prefix`, which only the turn-start path assigns, so 54 of
   56 produced plans were never shown.
3. **Stale progress renderer** — the post-turn footer renders through
   `workflow_position.render_workflow_progress` (`turn_runner:2551`) while
   the turn-start briefing renders through `workflow_view.render_workflow_progress`
   (`runtime/turn_hooks_builder.py:989`).

Every escalation into the planner happens mid-turn, so the unwired path is
the one production actually exercises.

## The move: one `enter_mode`

Replace both paths with one function owning everything that must happen when
a mode becomes active:

- manifest recompute (tool schemas + capability manifest + user-message
  capability block),
- model pin and routing context (`_apply_mode_routing_context` stays the
  single writer of `state.active_mode_id`),
- entry operations (with plan resume-awareness and the kernel-driven
  next-handoff contract),
- plan recording (position **and** `plan_prefix`),
- hint arming / suppression,
- context refresh (`ctx.mode_id` patch, system-prompt base swap),
- pre-turn operations.

```python
async def enter_mode(
    *,
    target: str,
    origin: Literal["turn_start", "handoff"],
    ...
) -> EnterModeResult
```

`origin` gates only the legitimate differences:

| Concern | `turn_start` | `handoff` |
| --- | --- | --- |
| Early-return semantics | entry op can end the turn (work op / plan failure) | rejection is a tool result, never a turn end |
| Workflow position | seed initial position if absent | advance from the departing mode |
| Handoff context block | not built | `_build_structured_handoff_context` |
| Routing side effects | entry-phase save | `apply_routing` + departed-modes bookkeeping |
| Suppression | sticky caller OR entry flag | post-handoff hint guards |

Anything not in that table has exactly one implementation after F1.

## Sequencing contract (from the live mid-turn path)

The order below is load-bearing (verified at `:5818-6020`); `enter_mode` must
preserve it for the `handoff` origin and the entry sequence's equivalent
order for `turn_start`:

1. gate: target in routing targets, target in catalog;
2. system-prompt base swap + clear routing-hint/handoff-context segments;
3. structured handoff context (handoff origin only);
4. tool schemas rebuild (yield policy resolved as Phase 1 does);
5. manifest recompile + user-message capability block refresh (one seam);
6. routing context / model pin (`_apply_mode_routing_context`);
7. completion-contract action tools refresh;
8. entry operations (plan resume-awareness, routing contract, plan position
   **and** `plan_prefix`);
9. identity swap (`state.departed_modes`, `mode_id`, `handoff_to`);
10. `apply_routing` side effects;
11. `ctx.mode_id` patch;
12. durable workflow-position advance;
13. pre-turn operations;
14. post-handoff hint recomputation and guards.

## Discipline

- **No behavior change.** The two paths' observable behavior is preserved;
   the differences are made explicit in `origin` branches, not silently
   unified. Where the paths disagree today (e.g. `plan_prefix`), the
   turn-start behavior is the contract — that disagreement *is* incident 2.
- **Single writers stay single.** `_apply_mode_routing_context` remains the
   only writer of `state.active_mode_id`; `enter_mode` calls it, never
   assigns the field.
- **Pinned contracts are the safety net.** The guard-ordering test
   (`kernel/tests/test_loop_control_ordering.py`), the handoff-cap
   source-lock, the transition tests, and the scenario matrix must pass
   unchanged; any failure names a real divergence to resolve, not a test to
   update.
- **Land as one focused change** on top of the merged F-series; rebase onto
   the source tip before starting (this region is the most contested in the
   file).
- **F6 rides along opportunistically**: signatures `enter_mode` touches that
   still type `state` as `Any` become `_TurnState` in the same change; the
   remaining sweep is separate.

## Regression entry points

- Mid-turn handoff behavior: `kernel/tests/test_turn_runner_transitions.py`
- Entry-phase contracts: `kernel/tests/test_turn_runner.py`,
  `kernel/tests/test_mode_instructions.py`
- Guard ordering: `kernel/tests/test_loop_control_ordering.py`
- Scenario matrix: `kernel/tests/test_turn_scenario_matrix.py`
