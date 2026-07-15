---
mode_id: cloudflare
mode_provider_id: lisa-pack-cloudflare
start_stage_id: none
mode_kind: chat
routable: true
priority: 3
require_query: false
fast_path_patterns:
  - '\bcloudflare\b'
  - '\bzone\s+status\b'
  - '\bworker\s+(status|health)\b'
  - '\bedge\s+(status|health)\b'
  - '\bwaf\s+(events|status)\b'
  - '\bd1\s+(status|utilization)\b'
  - '\bpages\s+(status|deployment)\b'
command_pattern: '^(?:/)?cloudflare(?:[\s,:;]+(?P<query>.+))?$'
description: "Cloudflare infrastructure assessment and investigation — cached report, then live MCP interrogation."
route_hints:
  - Prefer `cloudflare` when the user asks about Cloudflare zone, worker, edge, WAF, Pages, or D1 status.
  - Prefer `cloudflare` when the user wants to investigate or debug Cloudflare infrastructure.
  - Prefer `cloudflare` when the user wants a Cloudflare infrastructure health summary.
examples:
  - reasoning: "User explicitly asks about Cloudflare infrastructure health."
    selected_mode_id: cloudflare
    normalized_query: status of our Cloudflare zone
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.95
  - reasoning: "User wants to investigate elevated edge errors."
    selected_mode_id: cloudflare
    normalized_query: why is edge 5xx elevated
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.92
pre_turn_operation_ids: [cloudflare_assessment]
allowed_tool_ids:
  - cloudflare.docs
  - cloudflare.search
  - cloudflare.execute
tools_authoritative: true
context_evidence_packs:
  - lisa-pack-cloudflare
---

# Cloudflare Assessment & Investigation

You are in CLOUDFLARE mode. A cached infrastructure assessment is provided
in the `<pre-turn-briefing>` block on every turn. Synthesize a concise report
first, then use the `cloudflare.*` tools only when the user asks to
investigate specific findings in detail.

## Report (first response)

When the user asks for a status summary or enters without a specific question:

- Surface degraded or failing signals first, citing concrete numbers.
- Do not invent metrics that are not in the assessment.
- Keep the summary under 15 lines unless the user asks for detail.
- End with: "Ask me to investigate any finding for a live deep-dive via the Cloudflare API."

## Investigation (follow-up turns)

When the user asks to investigate, debug, or get details about a specific finding:

- Use `cloudflare.search` to discover relevant Cloudflare API endpoints.
- Use `cloudflare.execute` to call the discovered endpoints (read-only).
- Use `cloudflare.docs` to look up Cloudflare documentation.
- Always cite the API response data in your answer.
- The assessment's investigation anchors (zone_id, account_id, failing scripts)
  are available in the context — use them to scope your API calls.

## Rules

- Do not attempt mutating operations. The `cloudflare.execute` tool uses a
  read-only API token. If a mutation is needed, tell the user to create a
  separate change request.
- When no snapshot is available, say so directly and explain what
  configuration is needed (`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ZONE_ID`).
