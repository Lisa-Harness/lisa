---
title: "Introducing Lisa: An Opinionated Agent Harness Built for the Agent Loop, Not Against It"
date: 2026-09-14
draft: false
author: "Lisa Harness Team"
tags: ["agent", "framework", "llm", "mcp", "context-engineering"]
categories: ["announcement"]
description: "Lisa is a runtime-neutral framework for building agent hosts and mode/operation packs. We built her explicitly to solve the failure modes the 2025-2026 agent-harness research surfaced—hallucinated success, read-loop burns, context decay—and to stay simple while the field converges on complexity."
---

Most agent frameworks help you build a demo and then leave you alone in production. Lisa is built for the second part: the moment when your agent claims it wrote a file it never called the tool for, or burns a hundred thousand tokens re-reading the same page, or cheerfully announces a task complete because the context window forgot what "complete" meant.

<!--more-->

## The problem we kept hitting

The 2025–2026 agent-engineering consensus is now well documented. Anthropic's posts on building effective agents, on long-running harnesses, and on demystifying evals; Manus's context-engineering write-up; and dozens of production retros all point at the same failure pattern:

- Agents hallucinate success when nothing checks it.
- Tool loops burn tokens reading the same thing until a cap fires.
- Mode routing costs an extra model call per turn and still hands off badly.
- Memory recall is retrieval theater—dropping RAG hits into the prompt is not state.
- Token budgets are aspirational because nobody models the 97% prompt-resend reality.

Lisa exists because we kept hitting every one of these, and the fix was never in the prompt. It was always in the harness.

## What Lisa actually is

Lisa is a layered Python framework (uv workspace) for building agent **hosts** and **mode/operation packs**. The stack is deliberately thin:

| Layer | What it owns |
| --- | --- |
| `kernel` | Runtime-neutral contracts, the IPSAR FSM, the operation registry, tool and storage protocols, and per-iteration model-router discovery |
| `sdk` | Author-facing helpers, rendering, pack composition, and a CEL-based model router |
| `sdk/packs` | Bundled reference packs: core, memory, introspect, recovery |
| `host` | Transport-neutral runtime—typed config, mutable state, provider dispatch, MCP pool compilation, command dispatch, context assembly, thread persistence |
| `mcp_server` | First-party MCP servers that expose Lisa's surface-scoped tools over stdio |

A real host injects concrete implementations for model completion, tool execution, storage, memory, transport, and scheduled work. The kernel and SDK stay neutral; you can plug in a Teams transport, a CLI REPL, or a headless scheduler without forking the framework.

## How Lisa is different

### 1. Routing is emergent, not a separate planner

Most frameworks burn a model call per turn just to ask "which mode should handle this?" Lisa removed the planner entirely. Natural-language turns go straight to `current_mode_id` and the model hands off mid-turn by calling `transition_stage`—the same mechanism that already handles workflow stage advancement and fork fan-out.

One mechanism, zero extra model calls, full conversation context, and—if you want it—proactive capability hints injected before the model call so it doesn't waste iterations discovering it lacks the tools it needs.

### 2. Hallucinated success is rejected at run time

Lisa's completion contract works from **effect receipts**, not tool-name heuristics. Every declared write mode must produce at least one `EffectReceipt` with an actual write effect to complete cleanly. Read-only work in write-capable modes is explicitly exempted and honestly labeled. This is what kills the "false success" failure mode that Anthropic's long-running research named as one of their original two patterns.

### 3. The invariant registry is the framework

Lisa ships a registry of named turn-loop invariants, each with a code site, an incident provenance link, a pinning test, and a behavioral matrix cell. Invariants include:

- **Forced-action enforcement**—when the kernel computes a deterministic transition, non-compliant dispatches get capped and the kernel synthesizes the tool call.
- **Freeze-over-truncate**—cap expiry returns a pause or terminate action; it never silently finishes with an empty answer.
- **Stalled-write and silent read-loop rescue**—write-announcement loops and crawl-until-cap behavior are pattern-matched and interrupted at iteration time.
- **Fail-loud exception policy**—broad `except: Exception` blocks must either log with context or fail closed. Silent `TypeError` inside a broad except is a regression.

This is the response to the compounding-error problem that eval research keeps pointing at. Most harnesses discover these in production. Lisa defines, names, and pins them.

### 4. Token budgets from evidence, not folklore

Lisa's budget defaults come from measured session data, not guessed round numbers. The current working cap (15M tokens/day, ~3–4M cumulative per turn) is derived from observed iteration counts, per-call context ceilings, and prompt-resend ratios recorded in the durable token-usage ledger. The design document is reproducible from the telemetry: if the spend profile changes, the numbers get re-derived, not re-argued.

### 5. Context is admitted, not pasted

The context routing system is a declarative admission engine. Telemetry, operational alerts, and domain knowledge are evaluated against rules and rendered into named, token-capped Markdown **lanes** in an explicit order. The framework does not paste context into the prompt; it admits it, budgets it, and renders it. That keeps the cognitive load and the silent drift problems that Manus called "lost-in-the-middle" visible and controllable.

### 6. Memory as external learning, not as a prompt trick

The bundled packs treat memory as a first-class capability: `memory` for store/recall operations with external backends, `recovery` for `/resume` and thread-based recovery, `introspect` for the runtime's operational self-model. The memory write and recall paths are scoped by `surface_type`, `surface_id`, and `memory_scope` in the contracts—so prompt wording is never a privacy control. Hosts enforce scope in storage keys, retrieval filters, and delivery routing.

### 7. The MCP server layer is first-class

Lisa's built-in MCP servers (`lisa-storage`, `lisa-context`, `lisa-content`, `lisa-catalog`) expose the toolkit over stdio with surface scoping intact. Tool surface expansion happens by composing packs, not by stuffing more tools into a monolithic system prompt.

## Who this is for

Lisa is for engineers building agents that need to survive contact with real workloads—long turns, multi-mode sessions, forked children, memory across time, and the messy reality that model capability and model behavior are not the same thing. It favors explicit contracts, evidence-driven defaults, and named failure-mode prevention over prompt magic and graph abstractions.

If you have ever shipped an agent that claimed success it never performed, or watched a debugging agent re-fetch the same page a dozen times, this is the framework built to stop that.

## Try it

```sh
git clone https://github.com/Lisa-Harness/lisa.git
cd lisa
uv sync --extra dev
uv run lisa verify
uv run lisa modes
uv run lisa models discover && uv run lisa models apply
uv run lisa once "summarize the current thread state"
uv run lisa
```

The CLI host lives in the sibling `lisa-hosts/cli` repository and installs with `uv tool install lisa-harness-cli`.
