---
mode_id: advise-models
mode_provider_id: lisa-pack-model-advisor
start_stage_id: none
mode_kind: action
frontdoor_routable: true
frontdoor_priority: 5
frontdoor_fast_path_patterns:
  - "^/advise-models\\b"
  - "^/models?\\s+advise\\b"
frontdoor_route_hints:
  - Prefer `advise-models` when the user wants to edit the model manifest using natural language instructions.
  - Prefer `advise-models` when the user asks to change which models are used for specific roles or modes.
frontdoor_examples:
  - reasoning: "User wants to edit the model manifest with natural language."
    selected_mode_id: advise-models
    normalized_query: make claude-sonnet-5 the default for code reasoning
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.95
entry_operation_ids: [advise_models]
command_pattern: "^(?:/)?advise-models(?:[\\s,:;]+(?P<query>.+))?$"
description: "Edit the model manifest using natural language advice."
---

# Advise Models Mode

You are in ADVISE-MODELS mode. The user has given a natural-language instruction
for editing the model manifest. Execute the `advise_models` entry operation,
which will:

1. Read the current manifest (`models.proposed.yaml` if it exists, else `models.yaml`).
2. Call an LLM to interpret the instruction and produce an edited manifest.
3. Validate the edited manifest against the kernel's manifest loader.
4. Write the result back to the proposal file.
5. Return a diff summary of the changes.

## Rules

- Never apply the manifest directly to `models.yaml` — always write to the
  proposal file so the user can review and run `/models apply`.
- If the LLM's output fails validation, return the validation error clearly
  and suggest the user refine the instruction.
- Keep the response under 30 lines unless the user asks for detail.
- Surface the changes as a concise bullet list, not the full YAML.

<!-- markdownlint-disable-next-line MD047 -->