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
    configuration:
      # Sandbox containment floor for the shell/bubblewrap packs (Biba-style:
      # the model may only RAISE integrity above this floor, never loosen it).
      # This is the global session floor; a user session_context.md or a
      # per-pack configuration key may tighten (not loosen) it further.
      #   execution:    bare | sandbox     — sandbox = never run bare on host
      #   persistence:  direct-write | overlay | ephemeral | readonly
      #                 overlay = capture workspace changes in a mergeable layer
      #                 (the model may tighten to ephemeral/readonly from here)
      #   network:      deny | allow
      #   device:       dev | minimal | none
      #   ipc:          shared | unshared
      #   paths:        per-path visibility rw | ro | hidden
      #   tmpfs_budget_bytes / timeout_ceiling_seconds: ceilings only.
      lisa-pack-bwrap:
        sandbox:
          floor:
            execution: sandbox
            persistence: overlay
            network: deny
            device: minimal
            ipc: unshared
            paths:
              /run/log/journal: ro
            tmpfs_budget_bytes: 67108864
providers:
  modes:
    default: lisa-pack-core
routing:
  start_mode: default
  # Sequential-connector vocabulary shared by the routing-hint generator
  # and the planner-mode planner. A query containing any of these phrases
  # is treated as a sequential chain (single handoff, step-by-step) instead
  # of a parallel fork fan-out.
  sequential_connectors:
    - then
    - and then
    - after that
    - next
    - followed by
    - based on
    - once
  # Producer/consumer intent roles for the fork-sequential guard. A fork
  # runs children in parallel; none can see another's output. When the gap
  # mixes a producer intent (evidence-gathering) with a consumer intent
  # (artifact-creating), the fork is refused and the hint routes sequential
  # instead (single handoff to the first producer, then chain the consumer).
  # `review` is dual-role: self-sufficient alone, but a producer when paired
  # with `code`.
  producer_intents:
    - research
    - gather
    - learn
    - read
    - analyze
    - review
    - architect
    - ponder
    - question
    - clarify
    - okf
    - plan
  consumer_intents:
    - code
    - write
    - implement
    - review
  # Resume-phrase vocabulary for the affirmative-resume and continuation-
  # query detectors. Applied as prefix matches on the lowered message, so
  # "proceed with the current step" matches "proceed". Keep entries to
  # unambiguous imperatives; avoid common words like "ok" or "yes" that
  # false-positive on normal conversation (those are handled by the exact-
  # match set in host.handle_message).
  resume_phrases:
    - proceed
    - continue
    - go ahead
    - go for it
    - do it
    - carry on
    - pick up
    - resume
    - next step
    - keep going
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
