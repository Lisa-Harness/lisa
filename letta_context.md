---
schema_version: 2
name: letta
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    include:
      - lisa-pack-core
      - lisa-pack-memory
      - lisa-pack-letta
    required:
      - lisa-pack-core
providers:
  capabilities:
    memory: lisa-pack-letta
  modes:
    default: lisa-pack-memory
  tools:
    memory: lisa-pack-letta
routing:
  start_mode: letta
context:
  history:
    compaction:
      enabled: true
      pressure_threshold: 0.65
      hard_threshold: 0.85
      min_interval_turns: 3
      retention_count: 4
---
# Letta Session

## User

The user is using the free open-source lisa_framework.
The lisa_framework includes "Lisa."
Lisa is a reference implementation of the lisa_framework.
You are a running instance of Lisa delegating to the Letta agent.
