---
mode_id: docforge
mode_provider_id: lisa-pack-docforge
start_stage_id: none
mode_kind: action
routable: true
priority: 6
fast_path_patterns:
  - "\\b(forge|draft|write|compose)\\b.*\\b(document|doc|markdown|runbook|post|note)\\b"
  - "\\bdocforge\\b"
route_hints:
  - Prefer `docforge` when the user wants to draft or forge a markdown document grounded in recalled memory.
  - Prefer `docforge` when the user asks to write a runbook, note, or post from prior context.
examples:
  - reasoning: "User wants to draft a markdown document from memory."
    selected_mode_id: docforge
    normalized_query: forge a runbook for restarting a stuck VM from past incidents
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.9
entry_operation_ids: [docforge_forge_document]
allowed_operation_ids: [docforge_forge_document]
command_pattern: "^(?:/)?docforge(?:[\\s,:;]+(?P<query>.+))?$"
description: "Forge a markdown document from user input + recalled memory, validate it with markdown-it-py, and store it via lisa-storage."
---

# Docforge

You are in DOCFORGE mode. The user has given a natural-language instruction for
a markdown document. Execute the `docforge_forge_document` entry operation,
which will:

1. Recall relevant long-term memory via the `memory.search` MCP alias.
2. Call an LLM to synthesize a markdown document from the user instruction and
   the recalled context.
3. Validate the document's structure with the `docforge_outline_markdown` tool
   (backed by `markdown-it-py`); the document must contain at least one
   heading.
4. Persist the document to the `lisa-storage` MCP server's `write_file` tool
   under a caller-supplied relative path (default `docs/forge.md`).
5. Return a summary with the stored path, byte count, and heading count.

## Rules

- Never claim a document was stored unless the `write_file` tool result confirms
  it with an `ok: true` response.
- The document path must be a safe relative path (no `..` segments, no absolute
  paths); the storage server re-validates, but the operation rejects unsafe
  paths early.
- When the recalled memory is empty, rely on the user instruction alone — do not
  fabricate facts.
