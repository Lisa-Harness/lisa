---
name: Kernel Architecture
applyTo: "kernel/**"
description: Architecture boundaries for the kernel package.
---

# Kernel architecture

- The kernel package owns orchestration, routing, policy, and execution-state concepts.
- Do not import from `runtime`, `sdk`, or host applications. Hosts inject wiring; kernel modules define contracts.
- Public state and extension points use typed models, dataclasses, or protocols. Do not replace them with mutable `dict[str, Any]` bags.
- Registration and operation discovery must be side-effect-free on import.

## Scheduling and lifecycle

- The slice contract is opportunistic work: budget derives from elapsed time since the prior slice, not a wall-clock timer. Do not add schedulers to the kernel.
- Keep turn lifecycle transitions explicit. A `break` or `continue` in the tool-call loop does not exit its enclosing iteration loop.
- For deterministic routing, enforce the computed transition after the configured cap instead of repeatedly nudging the model. Store caps in `kernel.config.Settings`, not module constants.
- Add numeric settings to the existing coercion validator so environment values with inline comments remain parseable.
- A synthesized OpenAI tool call must include `id`, `type: "function"`, and `function.name` plus `function.arguments`.
- Only planner operations produce a planned next stage; action-mode output is normal entry output, not a handoff decision.
- Approval resumption must use the mode that owns the pending approval, including after a mid-turn handoff.

## Routing safeguards

- Detect alternating routing targets as well as identical repeated hints, and suppress hints after repeated model overrides.
- Replace a per-iteration budget briefing rather than accumulating prompt blocks. Brief only discrete execution budgets, not token or wall-time estimates.
- When no workflow steps remain, emit completion guidance rather than an unconditional handoff directive.
- Lifecycle resolution must preserve the established priority order and yield on stalled or exhausted work rather than force a final answer.

## Kernel test seams

- Patch the imported binding where the behavior is defined, not a stale re-export. Routing-hint tests patch `kernel.turn_phases._routing_hint.get_workflow_catalog`; routability checks also patch `kernel.ipsar_fsm.get_workflow_catalog`.
- Keep workflow-instance tests self-contained rather than importing fixtures from `test_turn_runner.py`, which creates a catalog import cycle.
- Assert `handoff_to` for mode handoffs; `transitioned_to` is only populated by workflow-stage transitions. Custom routing vocabulary must be reset after each test.
