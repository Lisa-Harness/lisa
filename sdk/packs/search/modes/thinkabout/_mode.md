---
mode_id: thinkabout
mode_provider_id: lisa-pack-search
start_stage_id: inquire
mode_kind: workflow
require_query: true
routable: true
priority: 8
allowed_operation_ids: [web_search]
tools_authoritative: true
fast_path_patterns:
  - "\\bthink\\s+about\\b"
  - "\\bgo\\s+in\\s+depth\\b"
  - "\\banalyze\\s+(?:this|tradeoffs?)\\b"
  - "\\bevaluate\\s+(?:this|tradeoffs?)\\b"
route_hints:
  - Route to `thinkabout` when the user explicitly asks to think about, go in depth, analyze, evaluate tradeoffs, or reason through one decision.
  - Prefer `thinkabout` when the user is asking for focused analysis, tradeoff evaluation, or a reasoned stance on one decision.
  - Prefer `thinkabout` when the user is asking for a focused analysis, pros and cons breakdown, or structured thinking about a specific question or decision.
examples:
  - reasoning: "User explicitly asks for focused analysis or a reasoned stance."
    selected_mode_id: thinkabout
    normalized_query: Can we think about this idea to get a better understanding?
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.88
description: "Exploratory reasoning and forming opinion workflow."
command_pattern: "^(?:/)?(?:thinkabout)(?:[\\s,:;]+(?P<query>.+))?$"
---

# Thinkabout Mode

You are in THINKABOUT workflow mode. This mode is exploratory.

- Build a reasoned point of view from evidence, constraints, and counterarguments.
- Consider and address counterarguments, especially those that are common or intuitive.
- Make uncertainties explicit in the opinion.
- Avoid overgeneralization from limited evidence.

Use `web_search` to dig into topics, ask questions, reason with the user through their problem, and ultimately formulate an opinion or thesis to provide actionable next steps. This starter workflow does not store durable memory; state conclusions in the response.
