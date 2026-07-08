# SPEC: _initialize_runtime Phase Extraction

## Status

Proposed — 2026-07-08

## Problem

`_initialize_runtime` in `main.py` has 60 statements (budget: ≤ 50). It
performs 7 distinct initialization phases in one function body:

1. Resolve and set debug/log-level.
2. Resolve and set data-dir, memory-db, thread-dir paths.
3. Resolve and set timeout profile and timezone.
4. Load state snapshot (if provided).
5. Resolve streaming mode and char delay.
6. Resolve session context (if provided).
7. Set runtime state fields and log the initialization.

## Goal

Split `_initialize_runtime` into focused phase functions so each phase is
independently testable and the orchestrator stays under 50 statements.

## Design

### Extracted phase functions

| Function | Responsibility |
| --- | --- |
| `_init_debug_and_log_level` | Resolve debug flag and log level, set in `_RUNTIME_STATE`. |
| `_init_paths` | Resolve data-dir, memory-db, thread-dir, set in `_RUNTIME_STATE`. |
| `_init_timeout_and_timezone` | Resolve timeout profile and timezone, set in `_RUNTIME_STATE`. |
| `_init_state_snapshot` | Load state snapshot from path, set in `_RUNTIME_STATE`. |
| `_init_streaming` | Resolve streaming mode and char delay, set in `_RUNTIME_STATE`. |
| `_init_session_context` | Resolve session context, set thread path and session flags. |

### Refactored _initialize_runtime

```python
def _initialize_runtime(
    *,
    debug: bool,
    log_level: str = "",
    log_path: str = "",
    no_log: bool = False,
    data_dir: str = "",
    memory_db: str = "",
    thread_dir: str = "",
    no_thread: bool = False,
    user_slug: str,
    timeout_profile: str,
    timezone: str = "",
    state_snapshot_path: str,
    streaming: str = "",
) -> None:
    """Initialize runtime state from CLI args and env vars."""
    _init_debug_and_log_level(debug=debug, log_level=log_level, log_path=log_path, no_log=no_log)
    _init_paths(data_dir=data_dir, memory_db=memory_db, thread_dir=thread_dir, no_thread=no_thread)
    _init_timeout_and_timezone(timeout_profile=timeout_profile, timezone=timezone)
    _init_state_snapshot(state_snapshot_path=state_snapshot_path)
    _init_streaming(streaming=streaming)

    _RUNTIME_STATE["user_slug"] = user_slug
    thread_name, context_ref, context_active = _resolve_session_context(user_slug)
    _init_session_context(
        thread_name=thread_name,
        context_ref=context_ref,
        context_active=context_active,
        thread_dir=thread_dir,
        no_thread=no_thread,
    )
    logger.info(
        "runtime initialized: user=%s timeout=%s tz=%s(%s) stream=%s",
        user_slug, _runtime_timeout_profile(), _runtime_timezone(),
        _runtime_timezone_source(), _runtime_streaming_mode(),
    )
```

## Test seam preservation

- `_initialize_runtime` stays on `main.py`'s surface with the same signature.
- Tests using `monkeypatch.setitem(little_lisa._RUNTIME_STATE, key, value)`
continue to work.
- Phase functions can be tested individually if needed.

## Verification

1. `ruff check examples/little_lisa/main.py --select PLR0915` — zero
   violations for `_initialize_runtime`.
2. `pytest examples/little_lisa/ -m "not live" -p no:randomly` — 99 passed,
   7 pre-existing failures unchanged.

## Out of scope

- Changing initialization behavior or resolution order.
- Moving `_resolve_runtime_thread_strategy` (already extracted, stays as-is).