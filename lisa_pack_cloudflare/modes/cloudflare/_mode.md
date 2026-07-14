---
mode_id: cloudflare
mode_provider_id: lisa-pack-cloudflare
start_stage_id: none
mode_kind: action
frontdoor_routable: true
frontdoor_priority: 3
frontdoor_fast_path_patterns:
  - "\\bcloudflare\\b"
  - "\\bzone\\s+status\\b"
  - "\\bworker\\s+(status|health)\\b"
  - "\\bedge\\s+(status|health)\\b"
frontdoor_route_hints:
  - Prefer `cloudflare` when the user asks about Cloudflare zone, worker, edge, WAF, Pages, or D1 status.
  - Prefer `cloudflare` when the user wants a Cloudflare infrastructure health summary.
frontdoor_examples:
  - reasoning: "User explicitly asks about Cloudflare infrastructure health."
    selected_mode_id: cloudflare
    normalized_query: status of our Cloudflare zone
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.95
entry_operation_ids: [cloudflare_status]
command_pattern: "^(?:/)?cloudflare(?:[\\s,:;]+(?P<query>.+))?$"
description: "Cloudflare zone, worker, and edge infrastructure status summary."
---

# Cloudflare Status Mode

You are in CLOUDFLARE STATUS mode. Produce a concise infrastructure health summary
from the cached Cloudflare signal snapshot.

When a `<pre-turn-briefing>` block is present, use it as the data source.
When no snapshot is available, say so directly and explain what configuration is needed
(`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ZONE_ID`).

## Rules

- Report zone status, edge error rates, worker inventory, WAF events, and any triage notes.
- Surface degraded or failing signals first.
- Cite concrete numbers, not generic statements.
- Do not invent metrics that are not in the snapshot.
- Keep the summary under 15 lines unless the user asks for detail.