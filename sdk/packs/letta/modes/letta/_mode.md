---
mode_id: letta
mode_provider_id: lisa-pack-letta
start_stage_id: none
mode_kind: chat
require_query: true
routable: true
priority: 8
routing_intent_tags: [letta, agent]
routing_intent_keywords:
  letta:
    - letta
    - memgpt
    - agent
    - delegate
command_pattern: '^(?:/)?(?:letta|ask-letta|agent)(?:[\s,:;]+(?P<query>.+))?$'
entry_operation_ids: [letta_ensure_agent]
allowed_operation_ids:
  - letta_ensure_agent
  - letta_send_message
  - letta_search_archival
  - letta_update_core_memory
  - letta_sync_memory
  - letta_compact_agent
  - recall_memory
allowed_tool_ids: [memory.add, memory.search, memory.list]
description: "Letta stateful-agent delegation mode. Sends the turn to the mapped Letta agent and returns its response. Normal Lisa chat is NOT routed here."
---

# Letta Mode

This mode delegates the turn to a Letta (MemGPT) stateful agent that owns persistent memory blocks and archival memory across conversations.

Use `letta_ensure_agent` at the start of a Letta workflow to resolve or create the mapped agent. Then use `letta_send_message` to delegate the user's query to that agent and return its response verbatim.

Use `letta_search_archival` only when the user explicitly asks to inspect the Letta agent's own archival memory plane. For normal recall, use `recall_memory` — it routes through the generic `memory.search` alias, which is already backed by Letta when this pack is the memory provider.

Use `letta_update_core_memory` for high-salience durable context the user wants pinned in the Letta agent's in-context memory (persona, project rules, user profile).

Use `letta_sync_memory` and `letta_compact_agent` to keep Lisa and Letta state aligned; these are also fired opportunistically by the pack's slice provider.
