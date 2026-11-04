---
schema_version: 2
name: e2e-memory
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    configuration: {}
    required:
      - lisa-pack-core
      - lisa-pack-memory
providers:
  modes:
    default: lisa-pack-core
  tools:
    web_search: e2etest-offline
  capabilities:
    memory: lisa-pack-memory
routing: { from: { pack: lisa-pack-core, session: default } }
execution:
  max_session_tokens: 200000
  max_completion_tokens: 50000
  max_capability_calls: 50
---
# E2E memory session (scripted dedup-recall scenario)

The scripted model drives this session; no provider is contacted. It
exercises the memory pack's capability surface end to end: a durable fact
is stored through ``memory.add``, then recalled through ``memory.search``,
with the session's surface scope enforced by the store. Turn 2's
auto-capture distiller additionally sees the stored fact as a dedup
candidate and skips it — proving the write-time dedup path in-process.