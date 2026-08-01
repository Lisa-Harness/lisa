---
mode_id: read
mode_provider_id: lisa-pack-core
start_stage_id: none
mode_kind: chat
require_query: true
routable: true
priority: 6
routing_intent_tags: [read, list]
satisfies_intents: [read, list]
routing_intent_keywords:
  list:
    - list
    - files
    - contents
  read:
    - read
    - file
    - content
  review:
    - review files
    - review content
command_pattern: '^(?:(?:/)?(?:read|review|list)(?:[\s,:;]+|(?=$))?$'
description: "List files and read them."
fast_path_patterns:
  - "^(?:read|review)\\b"
---
# List and Read Mode

Read for comprehension and analysis unless the user explicitly asks you to read for a different purpose.
