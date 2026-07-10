# SPEC: handle_message Command Dispatch Table

## Status

**Complete** — 2026-07-08. `handle_message` decomposed into 9 focused functions with a dispatch pattern. C901=25 → resolved. Zero test regressions.

## Problem

`handle_message` in `hosts/little_lisa/main.py` is the single most complex
function in the codebase:

| Metric | Current | Budget |
| --- | --- | --- |
| C901 (cyclomatic complexity) | 25 | ≤ 10 |
| PLR0911 (returns) | 11+ | ≤ 6 |
| PLR0912 (branches) | 20+ | ≤ 12 |
| PLR0915 (statements) | 80+ | ≤ 50 |

The function is a giant if-elif chain that handles host shortcuts (`/help`,
`/tokens`, `/settings`, `/debug`, `/consolidate`), deep routing, catalog-mode
routing, front-door planning, direct-response optimization, clarification
flows, unsupported-mode reporting, and the final `run_turn` dispatch — all
in one function body.

## Goal

Replace the if-elif chain with a **command dispatch table** so adding a new
host command is a one-line registration, not an edit to a 200-line function.

## Design

### Command handler protocol

```python
@dataclass
class CommandResult:
    """Result of a host shortcut command."""
    response_text: str
    mode_id: str = "default"
    log_context: dict[str, Any] | None = None
```

```python
type CommandHandler = Callable[[str], Awaitable[CommandResult | None]]
```

A handler returns `None` to signal "not my command, try the next handler."
A handler returns `CommandResult` when it owns the command.

### Dispatch table

```python
_HOST_COMMANDS: list[tuple[str, CommandHandler]] = [
    ("/help", _handle_help_command),
    ("/tokens", _handle_tokens_command),
    ("/settings", _handle_settings_command),
    ("/debug", _handle_debug_command),
    ("/consolidate", _handle_consolidate_command),
    ("/deep", _handle_deep_command),
]
```

### Refactored handle_message

```python
async def handle_message(message: str) -> str:
    """Route one console message through the kernel turn runner."""
    _log_turn_received(message)
    lowered = message.strip().lower()

    # 1. Host shortcut commands
    for prefix, handler in _HOST_COMMANDS:
        if lowered.startswith(prefix) or lowered == prefix[1:]:
            result = await handler(message)
            if result is not None:
                _finalize_command_result(message, result)
                return result.response_text

    # 2. Model-backed routing (front-door + run_turn)
    return await _route_model_turn(message)
```

### Extracted functions

| Function | Responsibility |
| --- | --- |
| `_handle_help_command` | `/help` and `/help capabilities` |
| `_handle_tokens_command` | `/tokens` |
| `_handle_settings_command` | `/settings` |
| `_handle_debug_command` | `/debug context\|capabilities\|self\|frontdoor` |
| `_handle_consolidate_command` | `/consolidate` and `/slice` |
| `_handle_deep_command` | `/deep <query>` |
| `_route_model_turn` | Front-door planning + run_turn dispatch |
| `_finalize_command_result` | Set context, log, append thread, update state |
| `_log_turn_received` | Log the turn.received debug event |

### Module placement

All extracted functions stay in `main.py` for now (test compatibility). A
future pass can move them to `cli_handlers.py` once the dispatch table
pattern is proven.

## Test seam preservation

- `handle_message` stays on `main.py`'s surface with the same signature.
- Tests using `monkeypatch.setattr(little_lisa, "handle_message", fake)`
continue to work.
- Tests calling `await little_lisa.handle_message("/help")` continue to work.

## Verification

1. `ruff check hosts/little_lisa/main.py --select C901,PLR0911,PLR0912,PLR0915`
   — zero violations for `handle_message`.
2. `pytest hosts/little_lisa/ -m "not live" -p no:randomly` — 99 passed,
   7 pre-existing failures unchanged.
3. Manual: `/help`, `/tokens`, `/settings`, `/debug context`, `/consolidate`
   all produce the same output as before.

## Out of scope

- Moving handlers to a separate module (deferred to a later pass).
- Changing the command set or output format.
- Front-door routing refactoring (separate SPEC).
