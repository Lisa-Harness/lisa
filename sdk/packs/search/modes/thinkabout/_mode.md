---
mode_id: thinkabout
mode_provider_id: lisa-pack-search
start_stage_id: inquire
mode_kind: workflow
require_query: true
routable: true
priority: 8
allowed_operation_ids: [web_search]
fast_path_patterns:
  - "\\bthink\\s+about\\b"
  - "\\bgo\\s+in\\s+depth\\b"
  - "\\banalyze\\s+(?:this|tradeoffs?)\\b"
  - "\\bevaluate\\s+(?:this|tradeoffs?)\\b"
description: "Exploratory reasoning and forming opinion workflow."
command_pattern: "^(?:/)?(?:thinkabout)(?:[\\s,:;]+(?P<query>.+))?$"
---
# Thinkabout Mode

Thinkabout workflow mode is designed to help the user reason through a specific question, decision, or tradeoff. It is focused on exploring the implications of a single decision or question, and providing a reasoned point of view.

- Build a reasoned point of view from evidence, constraints, and counterarguments.
- Consider and address counterarguments, especially those that are common or intuitive.
- Make uncertainties explicit in the opinion.
- Avoid overgeneralization from limited evidence.

Use `web_search` to dig into topics, ask questions, reason with the user through their problem, and ultimately formulate an opinion or thesis to provide actionable next steps. This starter workflow does not store durable memory; state conclusions in the response.
