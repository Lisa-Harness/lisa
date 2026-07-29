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
        checks:
          - id: ruff-format
            argv: [uv, run, ruff, format, --check, kernel, sdk, mcp_server, hosts]
            timeout_seconds: 120
          - id: ruff-check
            argv: [uv, run, ruff, check, kernel, sdk, mcp_server, hosts]
            timeout_seconds: 120
          - id: mypy
            argv: [uv, run, mypy, kernel, sdk, mcp_server]
            timeout_seconds: 300
          - id: tests
            argv: [uv, run, pytest, -m, "not live", --tb=short, -q]
            timeout_seconds: 600
        required_check_ids: [ruff-format, ruff-check, mypy, tests]
        branch_prefix: "lisa/"
        protected_branches: [main, master, develop, prod, production]
        push_remote: origin
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
    - {lane_id: operational, max_tokens: 1000, render_order: 0}
    - {lane_id: general, max_tokens: 2000, render_order: 10}
    - {lane_id: history, max_tokens: 1500, render_order: 20}
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
