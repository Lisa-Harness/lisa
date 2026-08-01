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
    - list files
    - list contents
    - list documents
  read:
    - read documents
    - read files
    - read text
    - read content
  review:
    - review documents
    - review files
    - review reports
    - review content
allowed_tool_ids: [read_file, list_files]
command_pattern: '^(?:(?:/)?(?:read|review)(?:[\s,:;]+|(?=$))?$'
description: "List files and read them."
fast_path_patterns:
  - "^(?:read|review)\\b"
---
# Read Mode

Read for comprehension and analysis unless the user explicitly asks you to read for a different purpose.
