---
mode_id: learn
mode_provider_id: lisa-pack-learning
start_stage_id: none
mode_kind: chat
require_query: true
routable: true
priority: 9
allowed_operation_ids: [synthesize_and_store_learning, recall_memory]
allowed_tool_ids: [memory.add, memory.search, memory.list]
description: "LEARN workflow mode to reflect on the conversation and distill durable insights."
route_hints:
  - Route to `learn` when the user asks to take a note, reflect on the conversation, capture a learning, or distill an insight.
  - Prefer `learn` when the user wants to remember something from the current discussion for the long term.
examples:
  - reasoning: "User explicitly asks to take a note of a decision."
    selected_mode_id: learn
    normalized_query: Take a note that we decided to use PostgreSQL.
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.98
  - reasoning: "User requests reflection and durable insight extraction."
    selected_mode_id: learn
    normalized_query: Reflect on what we just discussed and store the key takeaway.
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.95
command_pattern: '^(?:(?:/)?(?:learn|reflect|note)(?:[\s,:;]+|(?=$))|(?:please\s+)?(?:take\s+a\s+note\s+of|note\s+that)\s+)(?P<query>.*)$'
---
# Learn Mode

Reflect on the context and synthesize it into 1-3 memories.
Look for durable facts, actionable insights, strategic intent, decisions, and rules.

Use `synthesize_and_store_learning` to persist them unless the user explicitly requests the use of `memory.add`.

Use `recall_memory` to get prior learnings to inform your reflection and synthesis unless the user says not to. Only use `memory.search` or `memory.list` to retrieve prior learnings, when the user explicitly requests it.
