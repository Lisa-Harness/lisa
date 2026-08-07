---
mode_id: plan
mode_provider_id: lisa-pack-core
mode_kind: action
routable: true
priority: 11
routing_intent_tags: [plan]
satisfies_intents: [plan]
routing_intent_keywords:
  plan:
    - plan
    - workflow
    - multi-step
    - chain
    - orchestrate
entry_operation_ids: [plan_workflow]
# plan_workflow returns a routing contract (output_json.target_stage_id +
# optional steps); the kernel drives the handoff from the result shape and
# treats an empty target as a plan failure. Non-routing action modes (e.g.
# search/web_search) leave this at the default `work`.
entry_op_kind: routing
command_pattern: '^(?:/)?plan(?:[\s,:;]+(?P<query>.+))?$'
required_capabilities: [reasoning, tools]
yield_policy:
  allowed: false
  fallback_mode: discuss
description: "Plan a multi-step workflow and hand off to the first target mode."
---
# Plan Mode

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
