---
type: Lisa Mode
title: Learn Mode
description: Reflect on conversation and distill durable insights into long-term memory.
tags: [memory, learning, reflection]
status: stable
generated: { by: human:adam, at: 2026-07-25T00:00:00Z }
mode_id: learn
mode_provider_id: lisa-pack-memory
start_stage_id: none
mode_kind: chat
require_query: true
routable: true
priority: 9
routing_intent_tags: [learn]
satisfies_intents: [learn]
routing_intent_keywords:
  learn:
    - remember
    - note
    - reflect
    - learning
    - insight
    - distill
    - learn
    - capture
    - memory
    - store
allowed_tool_ids: [memory.add, memory.search, memory.list]
command_pattern: '^(?:(?:/)?(?:learn|reflect|note)(?:[\s,:;]+|(?=$))|(?:please\s+)?(?:take\s+a\s+note\s+of|note\s+that)\s+)(?P<query>.*)$'
---

# Learn Mode

Reflect on the context and synthesize it into 1-3 memories.
Look for durable facts, actionable insights, strategic intent, decisions, and rules.

Use `synthesize_and_store_learning` to persist them unless the user explicitly requests the use of `memory.add`.

Use `recall_memory` to get prior learnings to inform your reflection and synthesis unless the user says not to. Only use `memory.search` or `memory.list` to retrieve prior learnings, when the user explicitly requests it.

Use `dispute_memory` to challenge or correct prior learnings when necessary.
