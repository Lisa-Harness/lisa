---
mode_id: letta-agent
mode_provider_id: lisa-pack-letta
start_stage_id: none
mode_kind: action
require_query: true
routable: true
priority: 9
routing_intent_tags: [letta, agent, delegate]
routing_intent_keywords:
  letta:
    - letta-agent
    - ask-letta-agent
    - agent
    - delegate
command_pattern: '^(?:/)?(?:letta-agent|ask-letta-agent|agent)(?:[\s,:;]+(?P<query>.+))?$'
entry_operation_ids: [letta_ensure_agent, letta_send_message]
allowed_operation_ids:
  - letta_ensure_agent
  - letta_send_message
description: "Direct Letta-agent delegation. Sends the turn to the mapped Letta agent with no Lisa model call. Use /letta-agent <query>."
---

# Letta-Agent Mode (Direct Delegation)

This action mode delegates the turn to the mapped Letta stateful agent
with **no Lisa model call**. The turn runner runs the two entry
operations in order and returns immediately:

1. `letta_ensure_agent` — resolve or create the Letta agent mapped to
   this Lisa identity (idempotent; short-circuits when an
   `IdentityEntry` already exists).
2. `letta_send_message` — send the user query to that Letta agent and
   return its assistant response verbatim.

The Letta agent's own internal model loop runs server-side; Lisa is
just the dispatcher. No `model_call` stage event fires on the Lisa
host for turns routed here.

For the model-driven dispatch path — where the Lisa model picks among
`letta_search_archival`, `letta_update_core_memory`,
`letta_remember`, `letta_sync_memory`, `letta_compact_agent`, and
`recall_memory` based on the user's phrasing — use the chat-mode
`letta` mode (`/letta <query>`). This action mode intentionally exposes
only the send-and-return delegation path.