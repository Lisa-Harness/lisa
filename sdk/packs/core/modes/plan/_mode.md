---
mode_id: plan
mode_provider_id: lisa-pack-core
start_stage_id: none
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
command_pattern: '^(?:/)?plan(?:[\s,:;]+(?P<query>.+))?$'
description: "Plan a multi-step workflow and hand off to the first target mode."
---

# Plan Mode

Analyze the user's request and determine the best routing target.

Call `plan_workflow` to produce the target, then call `transition_stage`
with the returned `target_stage_id` to hand off. The framework handles the
handoff, fork fan-out, or sequential chain dispatch automatically.

When `plan_workflow` returns an empty `target_stage_id` (no routable mode
matched, no model wiring, or a parse failure), respond to the user directly
without calling `transition_stage`.
