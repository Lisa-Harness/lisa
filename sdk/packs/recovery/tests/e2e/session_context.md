---
schema_version: 2
name: e2e-recovery
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    configuration: {}
    required:
      - lisa-pack-bash
      - lisa-pack-bwrap
      - lisa-pack-markdown
      - lisa-pack-personas
      - lisa-pack-recovery
routing: { from: { pack: lisa-pack-core, session: default } }
execution:
  max_session_tokens: 1000000
  max_completion_tokens: 200000
  max_capability_calls: 100
---
# E2E recovery session

This session exercises recovery from a deliberately malformed tool call. The
failure is intentional: report it factually, do not repair or retry the call,
and wait for the operator to inspect and resolve the recovery item.
