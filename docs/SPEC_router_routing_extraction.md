# SPEC: Router Routing Module Extraction

## Status

**Complete** — 2026-07-08. Module created (`routing.py`), lint-clean, integrated into `main.py`. C901 17+12 → resolved. Zero test regressions.

## Problem

`_plan_route` and its nested closure
`_complete_router_planner` (which contains its own nested closure
`_retry_without_response_format`) are the second-most complex area in
`main.py`:

| Metric | Current | Budget |
| --- | --- | --- |
| `_plan_route` C901 | 17 | ≤ 10 |
| `_plan_route` statements | 83 | ≤ 50 |
| `_complete_router_planner` C901 | 12 | ≤ 10 |
| `_complete_router_planner` statements | 53 | ≤ 50 |
| Nested closures | 2 | 0 |

The nested closures capture mutable parent state (`router_debug` dict),
making them untestable in isolation and violating the repo's "flatten
closures" structural rule.

## Goal

Extract all router routing logic into a dedicated `routing.py`
module with flattened, testable functions.

## Design

### New module: `routing.py`

| Function | Responsibility |
| --- | --- |
| `plan_route` | Thin orchestrator: resolve model, build debug dict, call planner, store debug. |
| `_build_router_debug_dict` | Initialize the debug payload dict. |
| `_resolve_planner_model` | Resolve the router model, returning `(model_def, debug_update)` or fallback. |
| `_check_is_default_executor` | Check if planner model == default model. |
| `_complete_router_planner` | Module-level async function (not a closure). Accepts `router_debug` as a parameter. |
| `_retry_without_response_format` | Module-level async function. Accepts `router_debug` and `model_def` as parameters. |
| `_record_router_completion` | Record one completion attempt into the debug dict. |
| `_capture_prompt_shape` | Capture system/user prompt shape into the debug dict. |
| `_apply_fallback` | Update the debug dict with fallback reason and return the fallback decision. |

### Parameter passing replaces closure capture

Instead of:

```python
async def _plan_route(...):
    router_debug: dict[str, Any] = {...}

    async def _complete_router_planner(**kwargs):
        # captures router_debug, model_def
        ...
        async def _retry_without_response_format(*, attempt):
            # captures router_debug, model_def, response_format, messages
            ...
```

Refactored to:

```python
async def plan_route(
    *,
    raw_message: str,
    conversation_excerpt_markdown: str,
    resolve_model: Callable[[str], StubModelDefinition],
    complete_live: Callable[..., Awaitable[StubCompletion]],
    router_model_alias: str,
) -> RouterRouteDecision:
    router_debug = _build_router_debug_dict(raw_message, conversation_excerpt_markdown)
    ...
```

The closures become module-level functions that accept `router_debug` and
`model_def` as explicit parameters. The caller (`main.py`) passes
`_resolve_model` and `_complete_live` as callables.

### main.py thin wrapper

```python
async def _plan_route(
    *,
    raw_message: str,
    conversation_excerpt_markdown: str,
) -> RouterRouteDecision:
    """Thin wrapper — delegates to routing.plan_route."""
    decision = await routing.plan_route(
        raw_message=raw_message,
        conversation_excerpt_markdown=conversation_excerpt_markdown,
        resolve_model=_resolve_model,
        complete_live=_complete_live,
        router_model_alias=_router_model_alias(),
    )
    _set_last_router_debug(routing.last_debug_dict())
    return decision
```

## Test seam preservation

- `_plan_route` stays on `main.py`'s surface as a thin wrapper.
- Tests using `monkeypatch.setattr(little_lisa, "_plan_route", fake)`
continue to work.
- `_set_last_router_debug` and `_last_router_debug` stay on `main.py`.

## Verification

1. `ruff check hosts/little_lisa/routing.py` — zero violations.
2. `ruff check hosts/little_lisa/main.py --select C901,PLR0915` — zero
   violations for `_plan_route`.
3. `pytest hosts/little_lisa/ -m "not live" -p no:randomly` — 99 passed,
   7 pre-existing failures unchanged.

## Out of scope

- Changing the router routing algorithm or fallback behavior.
- Moving `build_user_prompt` or `build_route_catalog_markdown`
  (these depend on runtime state accessors and stay in `main.py`).
