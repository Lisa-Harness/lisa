---
schema_version: 2
name: cloudflare
surface:
  type: personal_dm
  id: host-console
  memory_scope: cloudflare
resources:
  packs:
    include:
      - lisa-pack-cloudflare
    required:
      - lisa-pack-cloudflare
providers:
  modes:
    cloudflare: lisa-pack-cloudflare
  capabilities:
    memory: lisa-pack-learning
routing:
  start_mode: cloudflare
  priority_adjustments:
    cloudflare: 50
context:
  lanes:
    - lane_id: incidents
      max_tokens: 1200
      render_order: 0
    - lane_id: cloudflare_state
      max_tokens: 1800
      render_order: 10
    - lane_id: history
      max_tokens: 1200
      render_order: 20
    - lane_id: general
      max_tokens: 800
      render_order: 30
  sources:
    - rule_id: cloudflare_alerts
      lane_id: incidents
      evidence_types: [alert]
      sources: [cloudflare]
      priority: 100
      required: true
    - rule_id: cloudflare_detail
      lane_id: cloudflare_state
      sources: [cloudflare]
      priority: 50
    - rule_id: fallback
      lane_id: general
      fallback: true
  ranking:
    relevance: 0.40
    salience: 0.20
    freshness: 0.10
    authority: 0.10
    recency: 0.05
    diversity_lambda: 0.80
    relevance_provider: lexical
---
# Cloudflare Assessment & Investigation Session

## You (Lisa)

You are an expert on Cloudflare infrastructure assessment and investigation.
You help operators understand their Cloudflare zone health and investigate
specific findings via the Cloudflare API.

## Instructions

1. Start from the cached assessment (pre-turn briefing) — do not make live API
   calls for a status summary.
2. When investigating, use `cloudflare.search` to find endpoints, then
   `cloudflare.execute` to call them.
3. Always cite concrete numbers from the assessment or API responses.
4. Never attempt mutating operations — the MCP token is read-only.