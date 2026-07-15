---
mode_id: gmail
mode_provider_id: lisa-pack-gmail
start_stage_id: gmail_draft
mode_kind: action
routable: true
priority: 10
fast_path_patterns:
  - "\\bgmail\\s+draft\\b"
  - "\\bdraft\\s+email\\b"
  - "\\bcompose\\s+email\\b"
route_hints:
  - Prefer `gmail` when the user wants to draft or compose an email.
examples:
  - reasoning: "User explicitly asks to draft an email."
    selected_mode_id: gmail
    normalized_query: draft an email to Sam about the Q3 update
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.95
entry_operation_ids: [gmail_draft]
allowed_operation_ids: [gmail_draft]
command_pattern: '^(?:/)?gmail(?:[\s,:;]+(?P<query>.+))?$'
description: "Draft an email and save it as a Gmail draft."
---

# Gmail Draft Mode

You are in GMAIL DRAFT mode. Draft an email from the user's request and
create it as a Gmail draft via the `gmail_create_draft` tool.

## Rules

- The entry operation (`gmail_draft`) drafts the body and persists it.
- If Gmail is not configured, say so directly and explain that `GMAIL_TOKEN`
  (an OAuth2 access token with `gmail.compose` scope) is required.
- Never send email — only create drafts. The user reviews and sends from
  Gmail's UI.
