---
mode_id: github
mode_provider_id: lisa-pack-github
start_stage_id: none
mode_kind: action
routable: true
priority: 4
fast_path_patterns:
  - "\\bgithub\\s+(status|health)\\b"
  - "\\bci\\s+(status|health|failing|passing)\\b"
  - "\\brepo\\s+status\\b"
  - "\\bworkflow\\s+(status|run)\\b"
  - "\\bpr\\s+(count|status|backlog)\\b"
route_hints:
  - Prefer `github` when the user asks about GitHub repository health, CI workflow status, or open PRs.
  - Prefer `github` when the user wants a repository and CI/CD health summary.
examples:
  - reasoning: "User explicitly asks about repository and CI health."
    selected_mode_id: github
    normalized_query: how is the GitHub repo doing
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.95
entry_operation_ids: [github_status]
command_pattern: "^(?:/)?github(?:[\\s,:;]+(?P<query>.+))?$"
description: "GitHub repository, CI workflow, and PR health status summary."
---

# GitHub Status Mode

You are in GITHUB STATUS mode. Produce a concise repository health summary
from the cached GitHub signal snapshot.

When a `<pre-turn-briefing>` block is present, use it as the data source.
When no snapshot is available, say so directly and explain what configuration is needed
(`GITHUB_TOKEN`, `GITHUB_REPO`).

## Rules

- Report latest commit, CI workflow status, open PR count, and failing checks.
- Surface failing CI or degraded signals first.
- Cite concrete values (commit SHA, workflow conclusion, PR count), not generic statements.
- Do not invent data that is not in the snapshot.
- Keep the summary under 15 lines unless the user asks for detail.