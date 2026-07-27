---
schema_version: 2
name: testing
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    include:
      - lisa-pack-memory
    required:
      - lisa-pack-brave-search
      - lisa-pack-markdown
      - lisa-pack-search
    auto_install:
      lisa-pack-brave-search:
        source: ../lisa-packs/brave_search
      lisa-pack-markdown:
        source: ../lisa-packs/markdown
      lisa-pack-search:
        source: ../lisa-packs/search
providers:
  capabilities:
    memory: lisa-pack-memory
  modes:
    default: lisa-pack-memory
  tools:
    web_search: lisa-pack-brave-search
routing:
  start_mode: default
  # Sequential-connector vocabulary shared by the routing-hint generator
  # and the plan-mode planner. A query containing any of these phrases is
  # treated as a sequential chain (single handoff, step-by-step) instead of
  # a parallel fork fan-out. Add conservative entries here; the fallback
  # list (then / and then / after that / next / followed by) is used when
  # this list is empty, so existing sessions behave unchanged.
  sequential_connectors:
    - then
    - and then
    - after that
    - next
    - followed by
    - based on
    - once
context:
  lanes:
    - {lane_id: operational, max_tokens: 1200, render_order: 0}
    - {lane_id: general, max_tokens: 1800, render_order: 10}
    - {lane_id: history, max_tokens: 1200, render_order: 20}
  sources:
    - rule_id: alerts
      lane_id: operational
      evidence_types: [alert]
      priority: 100
    - rule_id: general
      lane_id: general
      fallback: true
  ranking:
    relevance_provider: lexical
  history:
    compaction:
      enabled: true
      pressure_threshold: 0.65
      hard_threshold: 0.85
      min_interval_turns: 3
      retention_count: 4
---
# Testing Session

## User

The user is using the free open-source lisa_framework.
The lisa_framework includes "Lisa."
Lisa is a reference implementation of the lisa_framework.
You are a running instance of Lisa.

## You (Lisa)

You help the user understand how you are functioning.
You reflect on how your context appears,
how you should work in practice,
and how it could be improved.
