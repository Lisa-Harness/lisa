---
mode_id: search
start_stage_id: none
mode_kind: action
artifact_type: search_results
frontdoor_routable: true
frontdoor_priority: 0
frontdoor_fast_path_patterns:
  - "^(?:search(?:\\s+the\\s+web)?|web\\s+search|look\\s+up|find\\s+(?:recent|latest|current)|what(?:'s|\\s+is)\\s+(?:the\\s+)?(?:latest|new)|latest\\s+(?:news|updates?)\\s+(?:on|for)?|recent\\s+(?:news|updates?)\\s+(?:on|for)?)\\b"
frontdoor_route_hints:
  - Route to `search` when the user asks for current, recent, latest, external, or research-based information.
  - Prefer `search` when the user is asking for research or information that is not available in the current context.
  - Prefer `search` when the user is explicitly asking for recent information, data, or signals that would not be in the model's training data.
frontdoor_examples:
  - |-
      User message: "Find recent data about cloud costs"
      Reasoning: The user is asking for fresh information that may not be in model training data.
      selected_mode_id: search
      normalized_query: cloud costs
      clarification_needed: false
      clarification_question:
      confidence_score: 0.92
require_query: true
entry_operation_ids: [web_search]
command_pattern: "^(?:/)?search[\\s,:;]+(?P<query>.+)$"
description: "Web search mode."
---

# Search Mode

Provide a thoughtful summary and interpretation of the results of `web_search` which uses parallel exploration and synthesis to surface the most relevant and high-quality information from the web.
Focus on directly answering the user's query with concrete facts, citing sources when possible, and avoid narrative filler or tangential information.