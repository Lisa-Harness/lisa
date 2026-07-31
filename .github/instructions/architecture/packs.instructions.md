---
name: Packs Architecture
applyTo: "sdk/packs/**"
description: Architecture boundaries for the packs layer.
---

# Packs architecture

- Packs provide capability extensions and should remain composable rather than entangled with host behavior.
- Packs may import from `kernel` and `sdk`, never hosts. Avoid cross-pack imports; promote shared helpers to the SDK.
- Discovery and entry points must be side-effect-free. Hosts inject credentials and resolved configuration.
- Prefer declarative mode and operation content. Runtime handlers are thin callables that accept `OperationExecutionContext` and return `OperationResult`; they do not own routing or framework state.

## Catalog and routing invariants

- Each operation ID is unique across content roots. Operations must remain importable and discoverable without side effects.
- When Markdown and Python define the same operation ID, the Markdown definition is authoritative.
- A routable mode's `satisfies_intents` normally matches its `routing_intent_tags`; otherwise the handoff gap cannot shrink.
- Keep `__init__.py` at every package level so catalog module-path inference remains importable.
- Duplicate tool IDs require an explicit provider selection. Treat this catalog error as an environment configuration issue before diagnosing unrelated turn failures.
- External sibling packs install as wheel copies. Re-run `uv sync --extra ext` after editing their source before testing the installed package.

## Provider, storage, and authored-content rules

- Structured-output recovery catches both `ValidationError` and `EmptyStructuredOutputError`, retries with `response_format=None`, and uses `completion_content()` rather than `str(completion)`.
- Preserve tool-call-only retry results for the dispatch loop. Calibrate reasoning-model token limits on the hardest prompt and log finish reasons plus completion tokens on failures.
- Normalize both directions for ambiguous scalar/list fields: `None` or an empty string becomes an empty list; a non-empty string becomes a one-item list.
- Pack writes go through `lisa-storage` and use sandbox-relative paths. Never write a model-supplied absolute path directly.
- Before emitting an OKF concept, detect duplicate resources across the bundle. Re-emitting the same path is an update; a duplicate at a new path is not.
- Source-aware document consolidation may retry once for insufficient density, then persists an explicit failed-density annotation rather than silently substituting a fallback.
