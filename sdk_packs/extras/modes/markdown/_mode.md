---
mode_id: markdown
start_stage_id: refine
mode_kind: workflow
artifact_type: markdown_document
default_model: gpt-5.4-mini
require_query: true
allowed_operation_ids: [generate_markdown_doc]
command_pattern: "^(?:/)?(?:markdown|md)[\\s,:;]+(?P<query>.+)$"
description: "Generates a complete markdown document."
frontdoor_routable: true
frontdoor_route_hints:
  - Route to `markdown` when the user asks to create, generate, write, or draft a markdown file, document, or .md file.
  - Prefer `markdown` for requests to produce structured documentation in markdown format.
frontdoor_examples:
  - |-
      User message: "Can you create a markdown file for the project overview?"
      Reasoning: The user explicitly asks to create a markdown file.
      selected_mode_id: markdown
      normalized_query: create a markdown file for the project overview
      clarification_needed: false
      clarification_question:
      confidence_score: 0.95
  - |-
      User message: "Generate a .md document for the API specs."
      Reasoning: The user is requesting a markdown document generation.
      selected_mode_id: markdown
      normalized_query: Generate a .md document for the API specs.
      clarification_needed: false
      clarification_question:
      confidence_score: 0.95
default_intent_bridges:
  - target_mode_id: markdown
    matcher_patterns:
      - '^(?:right\s+)?(?:now\s+)?(?:can\s+you\s+|could\s+you\s+|please\s+|i\s+(?:want|need|would\s+like)\s+you\s+to\s+)?(?:generate|create|make|write|draft)\b.*\b(?:markdown|md|document|doc|file)\b.*$'
      - '^(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:write|draft|create)\b.*\b(?:markdown|md|document|doc|file)\b.*$'
      - '^(?:ok(?:ay)?|yes|yeah|yep|sure|please|alright|cool|great|go\s+ahead|do\s+it|lets|let\s+us)\b[\s,.:;-]*(?:generate|create|make|write|draft)\b.*\b(?:the\s+)?(?:markdown|md|document|doc|file)\b.*$'
      - '^(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:make|create|generate)\b.*\b(?:markdown|md)\b.*$'
      - '^(?:i\s+need|i\s+want)\b.*\b(?:markdown|md)\b.*$'
      - '^(?:create|generate|write)\b.*\b(?:markdown|md)\b.*$'
      - '^.*\b(?:generate|create|make|write|draft)\b.*\b(?:markdown|md|document|doc|file)\b.*$'
    query_extraction_strategy: passthrough
---

# Markdown Mode

You are in MARKDOWN mode.
Your goal is to collaborate with the user to craft the outline and parameters before generating a markdown document. You maintain context across turns and loop to seek clarification until the user is ready to generate. Once approved, use `generate_markdown_doc` to create the artifact, then exit.
