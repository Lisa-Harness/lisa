---
mode_id: surf
start_stage_id: iterate
mode_kind: workflow
artifact_type: web_page
default_model: gpt-chat-latest
require_query: true
pre_turn_operation_ids: [surf_browse]
post_turn_operation_ids: []
allowed_operation_ids: []
command_pattern: "^(?:/)?surf[\\s,:;]+(?P<query>.+)$"
description: "Iterative page-by-page web browsing workflow."
---

# Surf Mode

You are in SURF workflow mode. Keep a restart-safe page-by-page browsing session with running commentary. Use the surf pre-turn briefing as the source of browser truth, do not invent page content, and stay in this workflow until the user explicitly exits.
