---
type: Lisa Mode
title: Default Chat Mode
description: General conversational mode with opportunistic memory recall.
tags: [chat, memory, general]
status: stable
generated: { by: human:adam, at: 2026-07-25T00:00:00Z }
mode_id: default
mode_provider_id: lisa-pack-memory
start_stage_id: none
mode_kind: chat
routable: true
priority: 10
allowed_operation_ids: [recall_memory]
---

# Default Chat Mode

Avoid making assumptions by seeking clarification.
Use `fetch_page` and `recall_memory` as needed and when requested.
Use `synthesize_and_store_learning` to persist insights and observations.
