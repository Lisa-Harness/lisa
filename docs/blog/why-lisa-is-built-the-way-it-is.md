---
title: "Why Lisa Is Built the Way She Is: Trade-offs and Principles Behind an Opinionated Agent Harness"
date: 2026-09-14
draft: false
author: "Lisa Framework Team"
tags: ["agent", "architecture", "design", "context-engineering", "tradeoffs"]
categories: ["engineering"]
description: "Lisa makes specific trade-offs: routing as emergent behavior instead of planner calls, effect receipts instead of transcript inspection, tokens as evidence instead of folklore, and simplicity over graph abstraction. This post explains why we chose each one and which principle made that choice non-negotiable."
---

Lisa's README describes *what* she is. This post covers *why*—the decisions that look opinionated from the outside and the principles that made each one non-negotiable. We did not arrive at this shape by preference. We arrived because the prompt-and-graph alternatives kept producing failures that no amount of tuning ever fixed.

<!--more-->

## Principle 1: The framework must never hide the model loop

The 2025–2026 consensus is blunt: the most common source of agent bugs is not the model; it is the harness making the model's behavior harder to inspect than it needed to be. Anthropic's guidance has been consistent since December 2024: complex frameworks obscure the prompt/response loop and cause incorrect assumptions about what is under the hood.

That is why Lisa's core is not a graph DSL, a planner node, or a hidden retry engine. The kernel expresses a single loop with explicit state—IPSAR (intake, plan, step, act, reflect)—and everything else is a protocol. If you cannot answer "what did the model see, what did it do, and what happened?" for any iteration, the harness has already failed you.

We traded **framework-feature breadth** for **observability depth**. That was the cost. The benefit is that every failure mode we have ever hit has been debuggable from the telemetry and the invariant registry, not from reading a hidden planner's reasoning.

## Principle 2: If the model can be trusted to hand off, it must hand off itself

A pre-turn router is a second model call that exists only because the first one cannot be trusted to route. The trade-off with that design is well documented: it costs an extra call per turn, it decides on truncated context, and it still hands off badly because routing decisions need the full conversation, not a summary.

Lisa's choice was **unified routing through the turn loop**: the IPSAR FSM's `transition_stage` handles workflow stage advancement, mode handoff, and fork fan-out through one primitive. The model starts in `current_mode_id`, sees the full chat history, and calls `transition_stage` with a `mode_id` target when a better-suited mode exists. The loop intercepts, swaps the system prompt and tool surface, and continues—with history intact.

If the model cannot be trusted to call `transition_stage`, Lisa does not pretend otherwise. The forced-action enforcer caps non-compliant dispatches and the kernel synthesizes the tool call when necessary. That is the honest version of "the model routes itself": emergent behavior with a floor, not emergent behavior with a hope.

We traded **predictability** (a pre-turn planner always looks deterministic) for **correctness** (a routing decision that sees full context and never pays a second call to be wrong). We are not going back.

## Principle 3: Hallucinated success is a harness bug, not a model quirk

The single most production-corroding failure pattern in current agents is the model claiming completion it never performed. Anthropic's long-running work identified it as one of the two original failure patterns. Manus's guidance was to keep wrong turns in context because erasing failure removes evidence. The eval literature's guidance is that you grade trajectories and outcomes, not transcripts by vibe.

Lisa's answer is the **effect-receipt completion contract**: declared write modes must produce at least one `EffectReceipt` with a write or external_write effect to complete cleanly. Read-only work is explicitly exempted and honestly labeled rather than blocked.

This is a deliberate trade-off. Heuristic success checks are cheaper to write and look more flexible; receipts are stricter and occasionally force you to declare what "done" means more precisely than you wanted to. We traded **authoring convenience** for **run-time correctness**. In practice, that trade has paid for itself: every "false success" incident in our history was caught by this contract, not by reading transcripts after the fact.

## Principle 4: Context is a budget, not a paste

The framework's job is not to get more context into the prompt; it is to decide *what earns the prompt's attention* and *how much of it gets admitted*. Manus called this context engineering; the token-budget literature called it "the KV-cache hit rate is the most important metric in production." The two observations are the same observation.

Lisa's context routing system is a declarative admission engine. Evidence items (telemetry, operational alerts, domain knowledge from packs) are evaluated against rules and rendered into named, token-capped Markdown lanes with explicit render order. Healthy "green" telemetry is silenced by default. The kernel does not own or understand pack context; it routes it.

We traded **uncontrolled context richness** (a bigger context window will absorb this) for **context-as-allocation** (a lane has a budget, an order, and an admission rule). The 97% prompt-token re-send rate in our measured baseline made "more context" a false economy. The allocation model made it obvious which items were being paid for and which were not.

## Principle 5: Token budgets must be evidence-derived, not round numbers

Every framework ships with "sensible defaults." Most of them are folklore. The measured reality from our token-usage ledger on a typical heavy session: 97% of all tokens burned were prompt re-sends, not model output; the largest real call was ~107K tokens; heavy turns are 30-iteration bursts with ~100K context per call; multi-hour sessions burn 15–19M tokens.

Lisa's working limits—the 15M/day cap, the 3–4M-per-turn rail, the ~128K physical per-call ceiling—are all derived from that evidence. The design document exists so others can re-derive from the same ledger when the workload changes.

We traded **arbitrary simplicity** (a round number that looks right) for **evidentiary accuracy** (a number that matches what the agent actually burns). The consequence is a budget model you can argue with, because the argument has to produce data, not vibes.

## Principle 6: Memory is scoped by contract, not by prompt

"Prompt wording is not a privacy control" is the line that governs Lisa's memory design. Shared surfaces must not receive personal context unless the host implements explicit, audited sharing semantics. The framework does not achieve this by telling the model to be careful; it achieves it by carrying `surface_type`, `surface_id`, and `memory_scope` through the contracts and enforcing scope in storage keys, retrieval filters, and delivery routing.

We traded **convenience** (letting the model decide what is safe to share) for **guarantee** (storage keys and retrieval filters that cannot be prompted around). This is the one trade-off that becomes *more* important the better the model gets, because better models are also better at finding the edges of a poorly scoped system.

## Principle 7: Hard cutover, no compat layer

The old Lisa had a separate router planner (`kernel/router.py:plan_route`) that spent a model call per turn on routing. The unified FSM replaced it. When the cutover shipped, the old code was removed. No shims, no dual-path, no compatibility switches.

This applies at every level: the same policy governs fork-child semantics, memory store/recall, and context admission rules. When we change the design, the old design stops existing.

We traded **migration convenience** (keeping old paths alive while you learn the new one) for **architectural clarity** (exactly one path exists, so every failure has exactly one explanation). Hard cutover is painful in the week it ships. It is unambiguous in every debugging session that follows.

## What we deliberately gave up

Being opinionated means refusing things:

- **Static graph DSLs.** The workflow engine is an FSM with authored modes and operations. If you need a full DAG builder with cycle detection and conditional branches authored graph-first, a graph-based framework is a better fit.
- **Hidden planner agents.** If you want a routing model that reasons over a truncated excerpt and picks a mode, that is available—but it is a model router (CEL rule or external entry point), not a replacement for turn-time dispatch.
- **Automatic recovery by prompt.** Recovery is a pack (`sdk/packs/recovery`) with explicit `/resume` semantics, not a prompt that hopes the model infers state from chat history.
- **Zero-config context.** The session context front matter is explicit. Lanes, sources, budgets, and order live in YAML you version, not in defaults you hope survive a refactor.

## The short version

Lisa is built for people who have been burnt by agent harnesses that were too clever to debug. The design principles are:

- The harness must never hide the model loop.
- Emergent behavior gets a deterministic floor, never a hopeful ceiling.
- Hallucinated success is rejected at run time, not discovered in transcripts.
- Context is a budget, not a paste.
- Defaults are evidence-derived.
- Privacy is scoped by contract, not prompted.
- Cutovers are hard; shims are prohibited.

Those choices cost flexibility. They buy correctness, debuggability, and—when it matters most—the ability to answer *why*.
