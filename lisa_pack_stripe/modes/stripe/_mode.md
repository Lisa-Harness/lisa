---
mode_id: stripe
start_stage_id: none
mode_kind: action
frontdoor_routable: true
frontdoor_priority: 5
frontdoor_fast_path_patterns:
  - "\\bstripe\\s+(status|health)\\b"
  - "\\bsubscription\\s+(status|count)\\b"
  - "\\btrial\\s+(status|count|activity)\\b"
  - "\\bmrr\\b"
  - "\\bbilling\\s+(status|health)\\b"
frontdoor_route_hints:
  - Prefer `stripe` when the user asks about Stripe billing, subscriptions, trials, or MRR.
  - Prefer `stripe` when the user wants a billing and subscription health summary.
frontdoor_examples:
  - reasoning: "User explicitly asks about billing and subscription health."
    selected_mode_id: stripe
    normalized_query: Stripe billing status
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.95
entry_operation_ids: [stripe_status]
command_pattern: "^(?:/)?stripe(?:[\\s,:;]+(?P<query>.+))?$"
description: "Stripe subscription, trial, and billing health status summary."
---

# Stripe Status Mode

You are in STRIPE STATUS mode. Produce a concise billing health summary
from the cached Stripe signal snapshot.

When a `<pre-turn-briefing>` block is present, use it as the data source.
When no snapshot is available, say so directly and explain what configuration is needed
(`STRIPE_API_KEY`).

## Rules

- Report active subscriptions, active trials, trials started in 24h, and MRR.
- Surface any anomalous signals first (e.g., zero subscriptions when some are expected).
- Cite concrete numbers, not generic statements.
- Do not invent metrics that are not in the snapshot.
- Keep the summary under 10 lines unless the user asks for detail.