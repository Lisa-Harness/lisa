# Turn-loop invariants

These are maintained safeguards, not a list of unresolved incidents.
Historical failures remain in Git and the changelog; the contracts below
explain what must remain true during fixes. See [mode routing](../ROUTER_DESIGN.md).

## Regression map

Test names below are entry points into the linked suites, not a promise
that one test exhaustively proves every deployment behavior.

| Guard | Implementation | Regression entry |
| --- | --- | --- |
| Full routable mesh | [FSM](../../kernel/src/kernel/ipsar_fsm.py) | [test_routing_hint_forks_from_coding_mode](../../kernel/tests/test_turn_runner.py) |
| Forced-action enforcement | [Lifecycle](../../kernel/src/kernel/turn_phases/_lifecycle.py) | [test_forced_fork_when_model_refuses_to_fork](../../kernel/tests/test_turn_runner_transitions.py) |
| Receipt-based completion | [Completion contract](../../kernel/src/kernel/turn_phases/_completion_contract.py) | [test_parent_turn_no_receipt_blocked_as_no_tools](../../kernel/tests/test_effect_receipts.py) |
| Honest read-only exemption | [Completion contract](../../kernel/src/kernel/turn_phases/_completion_contract.py) | [test_parent_turn_with_receipt_is_not_blocked](../../kernel/tests/test_effect_receipts.py) |
| Stalled-write rescue | [Turn runner](../../kernel/src/kernel/turn_runner/) | [test_stalled_write_rescue_directs_announced_write_to_dispatch](../../kernel/tests/test_effect_receipts.py) |
| Silent read-loop rescue | [Turn runner](../../kernel/src/kernel/turn_runner/) | [test_stalled_write_rescue_fires_on_silent_read_loop_at_threshold](../../kernel/tests/test_effect_receipts.py) |
| Error-enriched nudges | [Lifecycle](../../kernel/src/kernel/turn_phases/_lifecycle.py) | [test_lifecycle_stage_stall_hierarchy](../../kernel/tests/test_turn_scenario_matrix.py) |
| Misrouted prose remains visible | [Completion contract](../../kernel/src/kernel/turn_phases/_completion_contract.py) | [test_routing_misfire_zero_dispatch_prose_surfaced_not_blocked](../../kernel/tests/test_effect_receipts.py) |
| Diagnostic cap expiry | [Lifecycle](../../kernel/src/kernel/turn_phases/_lifecycle.py) | [test_stuck_in_stage_cap_surfaces_diagnostic](../../kernel/tests/test_turn_runner.py) |
| Failed dispatch is still work | [Turn runner](../../kernel/src/kernel/turn_runner/) | [test_operation_dispatch_error_produces_tool_error_message](../../kernel/tests/test_turn_runner_operations.py) |
| Producer/consumer sequencing | [Routing hints](../../kernel/src/kernel/turn_phases/_routing_hint.py) | [test_routing_hint_refuses_fork_for_producer_consumer_mix](../../kernel/tests/test_turn_runner.py) |
| Honest empty completion | [Turn runner](../../kernel/src/kernel/turn_runner/) | [test_empty_final_completion_does_not_claim_iteration_budget](../../kernel/tests/test_turn_runner.py) |

The [scenario matrix](../../kernel/tests/test_turn_scenario_matrix.py)
combines these guards across failure shapes.

## Contracts

1. **Routable mesh:** eligible modes can hand off directly, not only through
   `default`. Workflow stage edges remain separately validated.
2. **Deterministic action:** if a model will not execute a pending computed
   transition/fork, bound non-compliant work and enforce it rather than
   repeatedly sending advisory prose. Keep hint and enforcement target
   computation consistent.
3. **Completion evidence:** validate declared action effects through receipts,
   not tool names, arbitrary calls, or a "done" sentence. Authored bindings
   and secondary `mode_allowlist` grants must agree with the advertiser;
   ambient defaults do not become required action effects.
4. **Read-only honesty:** read/execute evidence with no write attempt can be
   legitimate work in a write-capable mode. Do not block it as a failed write.
5. **Stalled-write rescue:** a write announcement without dispatch receives
   bounded corrective guidance, not unlimited narration.
6. **Silent read rescue:** read loops also need a progress boundary when an
   action is pending; lack of a write announcement is not an exemption.
7. **Error context:** repeated tool failures must inform the nudge, rather
   than asking a model to repeat the same failing action.
8. **Misrouting:** a legitimate conversational answer incorrectly routed
   into an action mode remains visible with an honest no-write diagnostic.
9. **Cap expiry:** exhausted/stalled work exits explicitly under lifecycle
   policy. Never silently label an incomplete turn successful or invent
   a question as though the model requested it.
10. **Dispatch accounting:** failed attempts still consume work and produce
    visible tool errors. They must not reset backstops into an endless retry.
11. **Dependencies:** fork only independent children. When one step produces
    evidence another consumes, sequence them even without a "then" connector.
12. **Empty fallback:** an empty answer does not by itself prove budget
    exhaustion. Diagnostics must name the actual exit reason.

## Additional integration invariants

- Approval resume preserves exact identity, dispatch cursor, receipts, and
  cumulative budgets; it does not replay a completed write or rerun entry
  operations. See [context lifecycle](../CONTEXT_LIFECYCLE.md).
- Result previews expand to original addressed evidence, not previews of
  previews; whole-request allocation accounts for reference overhead.
- Provider usage distinguishes cache counts from processed-token rails and
  physical request-window counts.
- Authenticated surface/memory scope and delivery checks are host-enforced;
  default content guards do not supply a safety policy.
- Unexpected errors remain observable at their handling boundary. A broad
  catch must not turn a real failure into success.

When changing a guard, add the smallest reproducing test and run its related
matrix/suite. Do not delete or loosen a safeguard merely because the original
incident is no longer reproducible in one happy-path run.
