---
mode_id: image
start_stage_id: refine
mode_kind: workflow
artifact_type: workflow_summary
frontdoor_routable: true
frontdoor_route_hints:
  - Route to `image` when the user asks to generate, create, draw, render, or visualize an image, picture, logo, or illustration.
  - Prefer `image` for any request involving visual creation or image generation.
frontdoor_examples:
  - |-
      User message: "Can you generate a logo for my new app?"
      Reasoning: The user explicitly asks to generate a logo.
      selected_mode_id: image
      normalized_query: generate a logo for my new app
      clarification_needed: false
      clarification_question:
      confidence_score: 0.98
  - |-
      User message: "Draw a futuristic city in cyberpunk style."
      Reasoning: The user is requesting a visual creation.
      selected_mode_id: image
      normalized_query: Draw a futuristic city in cyberpunk style.
      clarification_needed: false
      clarification_question:
      confidence_score: 0.98
require_query: true
allowed_operation_ids: [image_generate]
command_pattern: "^/image[\\s,:;]+(?P<query>.+)$"
description: "Generates one downloadable image from `/image` or a clear natural-language image request, including PNG, JPEG, GIF, and supported size or aspect hints."
default_intent_bridges:
  - target_mode_id: image
    matcher_patterns:
      - '^(?:now\s+)?(?:can\s+you\s+|could\s+you\s+|please\s+|i\s+(?:want|need|would\s+like)\s+you\s+to\s+)?(?:generate|create|make|render|draw|illustrate)\b.*\b(?:image|picture|portrait|illustration|photo|self(?:\s|-)?portrait|logo|icon|avatar)\b.*$'
      - '^(?:can\s+you\s+|could\s+you\s+|please\s+)?show\s+me\b.*\b(?:image|picture|portrait|illustration|photo|self(?:\s|-)?portrait|logo|icon|avatar)\b.*$'
      - '^(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:visualize|sketch|design|compose)\b.*\b(?:visual|image|picture|portrait|illustration|artwork|photo|logo|icon|avatar)\b.*$'
      - '^(?:ok(?:ay)?|yes|yeah|yep|sure|please|alright|cool|great|go\s+ahead|do\s+it|lets|let\s+us)\b[\s,.:;-]*(?:generate|create|make|render|draw|illustrate)\b.*\b(?:the\s+)?(?:image|picture|portrait|illustration|photo|logo|icon|avatar)\b.*$'
      - '^(?:please\s+)?(?:use|take|turn|convert)\b.*\b(?:description|prompt|concept|idea|direction|version|context|that|this)\b.*\b(?:generate|create|make|render|draw|illustrate)\b.*\b(?:image|picture|portrait|illustration|photo|logo|icon|avatar)\b.*$'
      - '^.*\b(?:description|prompt|concept|idea|direction|version)\b.*\b(?:now\s+)?(?:generate|create|make|render|draw|illustrate)\b.*\b(?:image|picture|portrait|illustration|photo|logo|icon|avatar)\b.*$'
      - '^.*\b(?:where|what)\b.*\b(?:image|picture|portrait|illustration|photo|logo|icon|avatar)\b.*$'
    query_extraction_strategy: passthrough
  - target_mode_id: image
    matcher_patterns:
      - '^(?:no|nah|nope|actually|try\s+again|redo\s+it|regenerate\s+it|revise\s+it|change\s+it|adjust\s+it)\b.*$'
      - '^(?:make|turn|render|generate|create)\s+(?:it|that|this)\b.*$'
      - '^(?:same|that|this|it)\b.*\b(?:png|jpe?g|gif|square|portrait|landscape|wide|wider|tall|taller|vertical|horizontal|16\s*:\s*9|9\s*:\s*16|1\s*:\s*1|1024\s*(?:x|×|by)\s*1024|1536\s*(?:x|×|by)\s*1024|1024\s*(?:x|×|by)\s*1536)\b.*$'
    query_extraction_strategy: passthrough
    prerequisites: [require_recent_handoff]
    recent_handoff_source_modes: [image]
    recent_handoff_required_artifact_keys: [generated_image, image_request]
    recent_handoff_context_key: recent_image_handoff
---

# Image Mode

You are in IMAGE mode.
Your goal is to collaborate with the user to craft the perfect prompt and parameters to generate an image. You maintain context across turns and loop to seek clarification until the user gives the go-ahead. Once approved, use `image_generate` to create the artifact, then exit.
