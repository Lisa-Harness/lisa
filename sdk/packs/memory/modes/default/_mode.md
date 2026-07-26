---
mode_id: default
mode_provider_id: lisa-pack-memory
start_stage_id: none
mode_kind: chat
routable: true
priority: 10
allowed_operation_ids: [recall_memory]
description: "General conversational mode with opportunistic memory recall."
---

# Default Chat Mode

Avoid making assumptions by seeking clarification.
Use `fetch_page` and `recall_memory` as needed and when requested.
Use `synthesize_and_store_learning` to persist insights and observations.
