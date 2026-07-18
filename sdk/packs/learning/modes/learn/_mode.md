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
command_pattern: '^(?:(?:/)?(?:learn|reflect|note)(?:[\s,:;]+|(?=$))|(?:please\s+)?(?:take\s+a\s+note\s+of|note\s+that)\s+)(?P<query>.*)$'
---
# Learn Mode

Reflect on the context and synthesize it into 1-3 memories.
Look for durable facts, actionable insights, strategic intent, decisions, and rules.

Use `synthesize_and_store_learning` to persist them unless the user explicitly requests the use of `memory.add`.

Use `recall_memory` to get prior learnings to inform your reflection and synthesis unless the user says not to. Only use `memory.search` or `memory.list` to retrieve prior learnings, when the user explicitly requests it.
