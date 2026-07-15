---
schema_version: 2
name: testing
surface:
  type: personal_dm
  id: host-console
  memory_scope: testing
resources:
  packs:
    include:
      - lisa-pack-core
      - lisa-pack-learning
      - lisa-pack-search
      - lisa-pack-github
      - lisa-pack-coding
    required:
      - lisa-pack-core
    configuration:
      lisa-pack-coding:
        workspace_root: "."
        checks:
          - id: focused-tests
            argv: [uv, run, pytest, kernel/tests, hosts/core/tests, sdk/packs/coding/tests, lisa_pack_github/tests, -q]
            timeout_seconds: 600
        required_check_ids: [focused-tests]
        branch_prefix: "lisa/"
        protected_branches: [main, master, develop, prod, production]
        push_remote: origin
      lisa-pack-github:
        base_branch: main
        allowed_base_branches: [main]
        branch_prefix: "lisa/"
        protected_branches: [main, master, develop, prod, production]
providers:
  modes:
    default: lisa-pack-core
  tools:
    web_search: lisa-pack-brave-search
  capabilities:
    memory: lisa-pack-learning
routing:
  start_mode: default
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
---
# Testing Session Context

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
