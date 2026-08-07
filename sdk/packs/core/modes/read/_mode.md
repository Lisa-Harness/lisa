---
mode_id: read
mode_provider_id: lisa-pack-core
mode_kind: chat
require_query: true
routable: true
priority: 6
routing_intent_tags: [read, list]
satisfies_intents: [read, list]
routing_intent_keywords:
  read:
    - read file
    - read the file
    - read from sandbox
    - show file
    - cat file
  list:
    - list files
    - list the files
    - show files
    - show contents
command_pattern: '^(?:/)?(?:read|list)(?:[\s,:;]+(?P<query>.+))?$'
description: "List files and read them from the storage sandbox."
fast_path_patterns:
  - "^(?:read|list)\\b"
---
# List and Read Mode

Read for comprehension and analysis unless the user explicitly asks you to read for a different purpose.
