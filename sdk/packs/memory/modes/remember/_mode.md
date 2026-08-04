---
type: Lisa Mode
title: Remember Mode
description: Reflect on conversation and distill durable insights into long-term memory.
tags: [memory, remembering, reflection]
status: stable
generated: { by: human:adam, at: 2026-07-25T00:00:00Z }
mode_id: remember
mode_provider_id: lisa-pack-memory
start_stage_id: none
mode_kind: chat
require_query: true
routable: true
priority: 9
routing_intent_tags: [remember]
satisfies_intents: [remember]
routing_intent_keywords:
  remember:
    - remember
    - note
    - reflect
    - capture
    - memory
    - store
    - learning
    - insight
    - distill
allowed_tool_ids: [memory.add, memory.search, memory.list]
allowed_operation_ids: [synthesize_and_store_learning, recall_memory]
command_pattern: '^(?:(?:/)?(?:remember|reflect|note)(?:[\s,:;]+|(?=$))|(?:please\s+)?(?:take\s+a\s+note\s+of|note\s+that)\s+)(?P<query>.*)$'
required_capabilities: [tools]
yield_policy:
  allowed: true
  default_question: "I want to capture the right insight. What specifically should I remember from this?"
---
## Remember Mode

Reflect on the context and synthesize it into 1-3 memories.
Look for durable facts, actionable insights, strategic intent, decisions, and rules.

Use `synthesize_and_store_learning` to persist them unless the user explicitly requests the use of `memory.add`.

Use `recall_memory` to get prior memories to inform your reflection and synthesis unless the user says not to. Only use `memory.search` or `memory.list` to retrieve prior memories, when the user explicitly requests it.

Use `dispute_memory` to challenge or correct prior memories when necessary.
