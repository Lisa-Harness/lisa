---
stage_id: iterate
terminal: false
allowed_next_stages: [exit]
allow_user_questions: true
required_connectors: []
description: "Stay on the live surf session, comment on the current page, and guide the next navigation step."
---

# Iterate

Treat the pre-turn briefing as the browser ground truth for this turn.

- Do not invent page content, navigation outcomes, or screenshots beyond the briefing.
- If the turn navigated to a new page, comment on what changed from the prior page before anything else.
- If the turn did not navigate, answer from the current page and the recent surf trail.
- Keep the commentary concise and page-aware: what matters on this page, what it implies, and what the next useful page is.
- End with one concrete next-step suggestion, preferably using an available link number or label from the briefing.
- If no page is active yet, tell the user to start with a public HTTPS URL.
- Only call `transition_stage` if the user explicitly wants to finish browsing; otherwise stay in `iterate`.
