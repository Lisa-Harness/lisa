---
mode_id: learn
mode_provider_id: lisa-pack-learning
start_stage_id: none
mode_kind: chat
require_query: true
frontdoor_routable: true
frontdoor_priority: 9
allowed_operation_ids: [synthesize_and_store_learning, recall_memory]
tools_authoritative: true
allowed_tool_ids: [memory.add, memory.search, memory.list]
description: "LEARN workflow mode to reflect on the conversation and distill durable insights."
frontdoor_route_hints:
  - Route to `learn` when the user asks to take a note, reflect on the conversation, capture a learning, or distill an insight.
  - Prefer `learn` when the user wants to remember something from the current discussion for the long term.
frontdoor_examples:
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

Reflect on the user's input, conversation and context to distill it into 1-3 highly durable facts or actionable insights.
Extract the strategic intent, decisions, rules, or operational facts that should be remembered over the long term.

Use `synthesize_and_store_learning` to persist this learning to memory.

Use `recall_memory` to retrieve relevant prior learnings to inform your reflection and synthesis.
