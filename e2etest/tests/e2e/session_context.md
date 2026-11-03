---
schema_version: 2
name: e2e-harness-smoke
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    required:
      - lisa-pack-core
providers:
  modes:
    default: lisa-pack-core
  tools:
    web_search: e2etest-offline
routing: { from: { pack: lisa-pack-core, session: default } }
execution:
  max_session_tokens: 200000
  max_completion_tokens: 50000
  max_capability_calls: 50
---
# Harness smoke session (offline scripted scenario)

The scripted model drives this session; no provider is contacted. The
bundled base-data overlay flips the core pack's ``web_search`` contract
stub ambient, so the default mode's tool surface carries it without a
provider pack. This session exists so `lisa-e2e run` works without
operator credentials or an external pack checkout.

The scripted model drives this session; no provider is contacted. This
session exists so `lisa-e2e run` works without operator credentials or an
external pack checkout, exercising the core pack's routing, tool surface,
and dispatch over the bundled base-data manifest.