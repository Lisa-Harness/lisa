---
schema_version: 2
name: testing
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    configuration:
      lisa-pack-coding:
        workspace_root: "."
      lisa-pack-github:
        base_branch: main
    include:
      - lisa-pack-brave-search
      - lisa-pack-markdown
      - lisa-pack-memory
      - lisa-pack-search
      - lisa-pack-coding
      - lisa-pack-github
      - lisa-pack-cloudflare
    # auto_install:
    #   lisa-pack-brave-search:
    #     source: "C:/Code/lisa-packs/brave_search"
    #     editable: true
    #   lisa-pack-markdown:
    #     source: "C:/Code/lisa-packs/markdown"
    #     editable: true
    #   lisa-pack-search:
    #     source: "C:/Code/lisa-packs/search"
    #     editable: true
    #   lisa-pack-coding:
    #     source: "C:/Code/lisa-packs-wip/coding"
    #     editable: true
    #   lisa-pack-github:
    #     source: "C:/Code/lisa-packs-wip/github"
    #     editable: true
    #   lisa-pack-cloudflare:
    #     source: "C:/Code/lisa-packs-wip/cloudflare"
    #     editable: true
providers:
  capabilities:
    memory: lisa-pack-memory
  modes:
    default: lisa-pack-memory
  tools:
    web_search: lisa-pack-brave-search
routing: { from: { pack: lisa-pack-core, session: default } }
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
