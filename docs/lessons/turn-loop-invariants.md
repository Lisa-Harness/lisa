# Turn Loop Invariant Registry

This registry documents the active architectural invariants of the Lisa kernel
turn loop. Each invariant prevents a concrete, historical failure mode that
previously caused infinite loops, token burns, silent data loss, or misrouted
actions.

## Invariant Registry

| Invariant | Code Site | Incident Provenance | Pinning Test | Matrix Cell |
| --- | --- | --- | --- | --- |
| **1. Full routable mesh** | `kernel.ipsar_fsm :: synthesize_routing_transitions` | 2026-07-17 to 2026-07-27 fork refusal | `test_routing_hint_forks_from_coding_mode` | — |
| **2. Forced-action enforcer** | `kernel.turn_phases._turn_state :: ForcedAction`, `kernel.turn_phases._lifecycle :: resolve_lifecycle_action` (rule 2d) | 2026-07-27 model refusal to hand off / fork | `test_forced_fork_when_model_refuses_to_fork` | — |
| **3. Completion contract via receipts** | `kernel.turn_phases._completion_contract :: resolve_completion_contract` | 2026-07-18 false success on empty writes | `test_parent_turn_no_receipt_blocked_as_no_tools` | `test_cell_2026_08_03_fork_child_read_loop` |
| **4. Honest read-only exemption** | `kernel.turn_phases._completion_contract :: resolve_completion_contract` | 2026-08-27 cloudservers read query warning | `test_parent_turn_with_receipt_is_not_blocked` | `test_cell_2026_08_27_read_receipt_fix` |
| **5. Stalled-write rescue** | `kernel.turn_runner :: _run_turn_inner` | 2026-09-02 write-announcement loop | `test_stalled_write_rescue_directs_announced_write_to_dispatch` | `test_cell_2026_09_02_write_announcement_contract` |
| **6. Silent read-loop rescue** | `kernel.turn_runner :: _run_turn_inner` | 2026-09-04 executor read-loop burn | `test_stalled_write_rescue_fires_on_silent_read_loop_at_threshold` | `test_cell_2026_09_04_fetch_page_read_loop_burn` |
| **7. Error-enriched stage nudge** | `kernel.turn_phases._lifecycle :: resolve_lifecycle_action` (rule 6) | 2026-09-04 crawl on 404/402 errors | `test_consecutive_tool_errors_enriches_stage_nudge` | `test_cell_2026_09_04_fetch_page_read_loop_burn` |
| **8. Routing-misfire prose surfacing** | `kernel.turn_phases._completion_contract :: resolve_completion_contract` | 2026-09-02 turn-70 executor prose discard | `test_routing_misfire_zero_dispatch_prose_surfaced_not_blocked` | `test_cell_2026_09_02_turn70_prose_discard` |
| **9. Freeze over truncate** | `kernel.turn_phases._lifecycle :: resolve_lifecycle_action` (rules 3, 5) | 2026-07-30 silent truncate bug | `test_stuck_in_stage_cap_surfaces_diagnostic` | `test_cell_2026_09_04_fetch_page_read_loop_burn` |
| **10. Failed-dispatch counting** | `kernel.turn_runner :: _run_turn_inner` dispatch error path | 2026-09-06 empty backstop re-arm | `test_operation_dispatch_error_produces_tool_error_message` | — |
| **11. Producer/consumer sequential guard** | `kernel.turn_phases._routing_hint :: _fork_has_producer_consumer_mix` | 2026-07-27 parallel run of dependent steps | `test_routing_hint_refuses_fork_for_producer_consumer_mix` | — |
| **12. Honest empty fallback** | `kernel.turn_runner :: _run_turn_inner` natural completion | 2026-08-30 budget-claim loop | `test_empty_final_completion_does_not_claim_iteration_budget` | `test_cell_2026_08_30_budget_claim_fabrication` |

---

## Detailed Invariant Contracts

### 1. Full Routable Mesh in FSM

- **Code:** `kernel/ipsar_fsm.py` (`synthesize_routing_transitions`).
- **Rule:** Every routable mode can transition to every other routable mode (full mesh).
- **Without it:** From any non-`default` mode, the only edge was back to `default`, breaking fan-out and multi-step plans.

### 2. Forced-Action Enforcer

- **Code:** `kernel/turn_phases/_turn_state.py` (`ForcedAction`), `kernel/turn_phases/_lifecycle.py` (rule 2d).
- **Rule:** When the kernel has computed a deterministic transition or fork, non-compliant dispatches are capped at `forced_action_non_compliant_cap`. Exceeding the cap causes the kernel to synthesize the tool call.
- **Without it:** Weak models refuse advisory hints, burning iterations on read tools until timing out.

### 3. Completion Contract via Effect Receipts

- **Code:** `kernel/turn_phases/_completion_contract.py` (`resolve_completion_contract`).
- **Rule:** A mode declaring a write contract (`action_tool_ids`) must produce at least one `EffectReceipt` with `effect in ("write", "external_write")` to complete cleanly.
- **Without it:** Models hallucinate success ("Done — wrote the file") without calling tools.

### 4. Honest Read-Only Exemption

- **Code:** `kernel/turn_phases/_completion_contract.py` (`resolve_completion_contract`).
- **Rule:** A turn in a write-capable mode that produces read/execute receipts and zero write attempts is honest read-only work; accepted without warning or block.
- **Without it:** Read-only queries in write-capable modes (e.g. `list_vms` in `cloudservers`) were falsely blocked as hallucinated success.

### 5. Stalled-Write Rescue

- **Code:** `kernel/src/kernel/turn_runner.py` (`_run_turn_inner`).
- **Rule:** When a model announces a write in prose while the ledger has zero write attempts, or stalls at threshold across read tools without writing, inject a one-shot directive into the next iteration's briefing.
- **Without it:** Models enter write-announcement loops or crawl read tools to the cap without committing to writes.

### 6. Routing-Misfire Prose Surfacing

- **Code:** `kernel/turn_phases/_completion_contract.py` (`resolve_completion_contract`).
- **Rule:** When a conversational question is misrouted into a write mode and the model answers in prose with zero tool calls, surface the prose with a no-write warning footnote instead of discarding it as `no_tools`.
- **Without it:** Legitimate conversational answers were discarded and replaced with failure text.

### 7. Freeze Over Truncate on Cap Expiry

- **Code:** `kernel/turn_phases/_lifecycle.py` (`resolve_lifecycle_action`).
- **Rule:** Cap expiry (global iteration cap, stage nudge cap, budget exhaustion) returns `PauseAction` or `TerminateAction`, never silent `FinishAction`.
- **Without it:** Exhausted budgets produced empty answers or spurious "How should I proceed?" yields.

### 8. Fail-Loud Exception Policy

- **Code:** `kernel/src/kernel/turn_runner.py`, `kernel/src/kernel/turn_phases/*`.
- **Rule:** Catching broad `Exception` must log diagnostic context with `exc_info=True` or fail closed.
- **Without it:** Silent `TypeError`s inside broad `except` blocks prevent critical state settlement (e.g. recovery stalls).
