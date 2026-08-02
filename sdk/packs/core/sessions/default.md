---
schema_version: 2
name: lisa
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    required: [lisa-pack-core]
providers:
  modes:
    default: lisa-pack-core
routing:
  start_mode: default
  # Sequential-connector vocabulary shared by the routing-hint generator
  # and the plan-mode planner. A query containing any of these phrases is
  # treated as a sequential chain (single handoff, step-by-step) instead of
  # a parallel fork fan-out.
  sequential_connectors:
    - then
    - and then
    - after that
    - next
    - followed by
    - based on
    - once
execution:
  # Working session token budget (cumulative across the session's turns).
  # 15M is sized from docs/TOKEN_BUDGET.md: a ~3M max-intensity turn × 5,
  # below the 19.26M inflated 2026-08-01 baseline. Blocks further turns
  # pre-turn once the running session total is exhausted.
  max_session_tokens: 15000000
  # Completion-based per-turn token rail: bounds actual model work (output
  # tokens), not cumulative context re-send. Iterations remain the primary
  # turn-length governor.
  max_completion_tokens: 500000
context:
  sources:
    - rule_id: default
      lane_id: operational
      fallback: true
---
# Lisa Session

Use the available tools and supplied context carefully. Distinguish verified
facts from inferences, preserve the user's intent, and explain important
limits and uncertainties directly.
