---
mode_id: planner
mode_provider_id: lisa-pack-core
mode_kind: action
routable: true
priority: 11
fast_path_patterns:
  - "\\bplan(?:ner)?\\b"
  - "\\bprioritize\\b"
  - "\\bmulti.?step\\b"
routing_intent_tags: [plan]
satisfies_intents: [plan]
routing_intent_keywords:
  plan:
    - plan
    - prioritize
    - workflow
    - multi-step
    - orchestrate
entry_operation_ids: [plan_workflow]
entry_op_kind: routing
command_pattern: '^(?:/)?plan(?:er)?(?:[\s,:;]+(?P<query>.+))?$'
needs_full_mode_catalog: true
description: "Orchestrates tasks, breaks down intent, uses fork to dispatch parallel sub-tasks."
---

# Planner Mode

Analyze the user's request and produce a visible multi-step plan.

Call `plan_workflow` to produce the plan. The operation returns a `steps`
array (one entry per detected step, in execution order) and a
`target_stage_id` (the first step's `mode_id`). The kernel renders the
plan as a markdown block shown to the user, persists it as a
`WorkflowInstance`, and drives the handoff itself — do **not** call
`transition_stage`; the framework hands off, fans out, or chains the
next step automatically from the plan result.

When `plan_workflow` returns an empty `target_stage_id` and no `steps`
(no routable mode matched, no model wiring, or a parse failure), respond
to the user directly.
