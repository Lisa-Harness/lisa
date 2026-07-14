---
mode_id: default
mode_provider_id: lisa-pack-core
start_stage_id: none
mode_kind: chat
frontdoor_routable: true
frontdoor_priority: 10
frontdoor_route_hints:
  - Prefer the smallest sufficient mode.
  - Prefer `default` for ordinary questions, model-selection questions, and requests outside specialized modes.
  - Prefer `default` for model-selection and routing questions, such as choosing default chat models or switch timing between model families.
  - Prefer `default` when no specialized mode clearly owns the turn.
  - When the user is explicit about one mode (for example says think about, deep dive, or multiple perspectives) or uses a direct mode command, route directly instead of asking for clarification.
  - If more than one specialized mode appears plausible, choose the one implied by the primary verb in the user's message.
  - If the user is ambiguous but still probably wants a specialized mode, route instead of over-clarifying.
  - Use clarification only as a last resource when routing would otherwise be meaningfully wrong.
description: "General conversational mode."
---

# Default Chat Mode

Offer general conversation, questions, and answers.
Always seek clarification when user intent or context is ambiguous.
