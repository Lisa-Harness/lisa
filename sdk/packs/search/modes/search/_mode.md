---
mode_id: search
mode_provider_id: lisa-pack-search
start_stage_id: none
mode_kind: action
artifact_type: search_results
routable: true
routing_intent_tags: [research]
routing_intent_keywords:
  research:
    - search
    - latest
    - recent
    - current
    - news
    - update
    - today
fast_path_patterns:
  - "^(?:search(?:\\s+the\\s+web)?|web\\s+search|look\\s+up|find\\s+(?:recent|latest|current)|what(?:'s|\\s+is)\\s+(?:the\\s+)?(?:latest|new)|latest\\s+(?:news|updates?)\\s+(?:on|for)?|recent\\s+(?:news|updates?)\\s+(?:on|for)?)\\b"
require_query: true
entry_operation_ids: [web_search]
command_pattern: "^(?:/)?search[\\s,:;]+(?P<query>.+)$"
description: "Web search mode."
---

# Search Mode

Provide a detailed and reasoned summary and interpretation of the results of `web_search` which uses parallel exploration and synthesis to surface the most relevant and high-quality information from the web.
