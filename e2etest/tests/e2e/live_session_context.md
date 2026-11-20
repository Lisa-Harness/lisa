---
schema_version: 2
name: e2e-live-canary
surface:
  type: personal_dm
  id: e2e-live-canary
  memory_scope: personal
resources:
  packs:
    required:
      - lisa-pack-memory
      - lisa-pack-markdown
      - lisa-pack-personas
      - lisa-pack-serper
providers:
  capabilities:
    memory: lisa-pack-memory
  tools:
    web_search: lisa-pack-serper
routing: { from: { pack: lisa-pack-core, session: default } }
execution:
  max_model_calls: 12
  max_capability_calls: 12
  max_total_tokens: 150000
  max_completion_tokens: 12000
  max_wall_ms: 180000
  max_session_tokens: 200000
  scopes:
    "lisa-pack-personas:researcher":
      max_model_calls: 8
      max_capability_calls: 8
      max_total_tokens: 120000
      max_completion_tokens: 12000
      max_wall_ms: 180000
---
# Live canary session

Complete the user's requested result directly and within the declared
budgets. Do not ask follow-up questions when the available evidence is
sufficient.
