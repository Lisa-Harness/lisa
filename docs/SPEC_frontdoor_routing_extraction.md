# SPEC: Frontdoor Routing Module Extraction

## Status

**In progress** — 2026-07-08. Module created (`frontdoor_routing.py`), lint-clean. Integration into `main.py` pending.

## Problem

`_plan_frontdoor_route` and its nested closure
`_complete_frontdoor_planner` (which contains its own nested closure
`_retry_without_response_format`) are the second-most complex area in
`main.py`:

| Metric | Current | Budget |
| --- | --- | --- |
| `_plan_frontdoor_route` C901 | 17 | ≤ 10 |
| `_plan_frontdoor_route` statements | 83 | ≤ 50 |
| `_complete_frontdoor_planner` C901 | 12 | ≤ 10 |
| `_complete_frontdoor_planner` statements | 53 | ≤ 50 |
| Nested closures | 2 | 0 |

The nested closures capture mutable parent state (`frontdoor_debug` dict),
making them untestable in isolation and violating the repo's "flatten
closures" structural rule.

## Goal

Extract all frontdoor routing logic into a dedicated `frontdoor_routing.py`
module with flattened, testable functions.

## Design

### New module: `frontdoor_routing.py`

| Function | Responsibility |
| --- | --- |
| `plan_frontdoor_route` | Thin orchestrator: resolve model, build debug dict, call planner, store debug. |
| `_build_frontdoor_debug_dict` | Initialize the debug payload dict. |
| `_resolve_planner_model` | Resolve the front-door model, returning `(model_def, debug_update)` or fallback. |
| `_check_is_default_executor` | Check if planner model == default model. |
| `_complete_frontdoor_planner` | Module-level async function (not a closure). Accepts `frontdoor_debug` as a parameter. |
| `_retry_without_response_format` | Module-level async function. Accepts `frontdoor_debug` and `model_def` as parameters. |
| `_record_frontdoor_completion` | Record one completion attempt into the debug dict. |
| `_capture_prompt_shape` | Capture system/user prompt shape into the debug dict. |
| `_apply_fallback` | Update the debug dict with fallback reason and return the fallback decision. |

### Parameter passing replaces closure capture

Instead of:

```python
async def _plan_frontdoor_route(...):
    frontdoor_debug: dict[str, Any] = {...}

    async def _complete_frontdoor_planner(**kwargs):
        # captures frontdoor_debug, model_def
        ...
        async def _retry_without_response_format(*, attempt):
            # captures frontdoor_debug, model_def, response_format, messages
            ...
```

Refactored to:

```python
async def plan_frontdoor_route(
    *,
    raw_message: str,
    conversation_excerpt_markdown: str,
    resolve_model: Callable[[str], StubModelDefinition],
    complete_live: Callable[..., Awaitable[StubCompletion]],
    frontdoor_model_alias: str,
) -> FrontDoorRouteDecision:
    frontdoor_debug = _build_frontdoor_debug_dict(raw_message, conversation_excerpt_markdown)
    ...
```

The closures become module-level functions that accept `frontdoor_debug` and
`model_def` as explicit parameters. The caller (`main.py`) passes
`_resolve_model` and `_complete_live` as callables.

### main.py thin wrapper

```python
async def _plan_frontdoor_route(
    *,
    raw_message: str,
    conversation_excerpt_markdown: str,
) -> FrontDoorRouteDecision:
    """Thin wrapper — delegates to frontdoor_routing.plan_frontdoor_route."""
    decision = await frontdoor_routing.plan_frontdoor_route(
        raw_message=raw_message,
        conversation_excerpt_markdown=conversation_excerpt_markdown,
        resolve_model=_resolve_model,
        complete_live=_complete_live,
        frontdoor_model_alias=_frontdoor_model_alias(),
    )
    _set_last_frontdoor_debug(frontdoor_routing.last_debug_dict())
    return decision
```

## Test seam preservation

- `_plan_frontdoor_route` stays on `main.py`'s surface as a thin wrapper.
- Tests using `monkeypatch.setattr(little_lisa, "_plan_frontdoor_route", fake)`
continue to work.
- `_set_last_frontdoor_debug` and `_last_frontdoor_debug` stay on `main.py`.

## Verification

1. `ruff check examples/little_lisa/frontdoor_routing.py` — zero violations.
2. `ruff check examples/little_lisa/main.py --select C901,PLR0915` — zero
   violations for `_plan_frontdoor_route`.
3. `pytest examples/little_lisa/ -m "not live" -p no:randomly` — 99 passed,
   7 pre-existing failures unchanged.

## Out of scope

- Changing the frontdoor routing algorithm or fallback behavior.
- Moving `_build_frontdoor_user_prompt` or `_build_frontdoor_route_catalog_markdown`
  (these depend on runtime state accessors and stay in `main.py`).