---
mode_id: workshop
start_stage_id: none
mode_kind: action
require_query: true
frontdoor_routable: true
frontdoor_priority: 7
frontdoor_fast_path_patterns:
  - '\bworkshop\b'
  - '\bfan\s*out\b'
  - '\bmultiple\s+perspectives\b'
  - '\bsynthesize\b'
entry_operation_ids: [workshop_run]
frontdoor_route_hints:
  - Prefer `workshop` when the user wants multiple perspectives, a deep dive on a complex topic, or a comprehensive synthesis of information.
workshop_panels:
  - panel_id: analysis
    title: Strongest Case
    mode_id: thinkabout
    query_prefix: "Build the strongest operator-grade case for this proposal."
  - panel_id: challenge
    title: Risk Review
    mode_id: thinkabout
    query_prefix: "Stress-test this proposal and identify the strongest risks."
command_pattern: "^(?:/)?workshop(?:[\\s,:;]+(?P<query>.+))?$"
description: "Fan out a problem into a small workshop panel with curated context and delegated workflow execution."
---

## Workshop Mode

You are in WORKSHOP mode. Treat the user's question as a coordination problem: frame it cleanly, curate the right cached context for each panel, run the delegated child workflows, and return one synthesized answer.

## Rules

- Keep the panel count small and purposeful.
- Use delegated child runs to create genuine perspective tension, not duplicated summaries.
- Prefer concrete tradeoffs, operator-grade constraints, and next moves over abstract brainstorming.
- Works across personal and shared conversation surfaces.
