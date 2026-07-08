# SPEC: _build_turn_hooks Closure Flattening

## Status

Proposed — 2026-07-08

## Problem

`_build_turn_hooks` in `main.py` contains 9 nested async closures that
capture mutable parent state (`memory_store`, `state_provider`). This makes
the hooks untestable in isolation and violates the repo's "flatten closures"
structural rule.

| Metric | Current | Budget |
| --- | --- | --- |
| C901 (cyclomatic complexity) | 18 | ≤ 10 |
| PLR0915 (statements) | 51 | ≤ 50 |
| Nested closures | 9 | 0 |

The nested closures:

1. `_post_turn_slice` — delegates to `state_provider.slice`
2. `_pre_turn_slice` — delegates to `state_provider.slice`
3. `_explicit_slice` — delegates to `state_provider.slice`
4. `_answer_memory_recall` — queries `memory_store`, ranks records
5. `_write_memory` — writes to `memory_store`
6. `_list_memory_records` — lists from `memory_store`
7. `_complete` — delegates to `_complete_live`
8. `_resolve_model_hook` — delegates to `_resolve_model`
9. `_execute_tool` — delegates to `_execute_tool`

## Goal

Flatten all closures into a `TurnHookProviders` class that holds the shared
state (`memory_store`, `state_provider`) as instance attributes, with the
hook callbacks as methods.

## Design

### New class: `LittleLisaTurnHooks`

```python
class LittleLisaTurnHooks:
    """Holds shared state and provides TurnHooks callback methods."""

    def __init__(
        self,
        *,
        memory_store: MCPMemoryClient,
        state_provider: LittleLisaStateProvider,
        complete_live: Callable[..., Awaitable[StubCompletion]],
        resolve_model: Callable[[str], StubModelDefinition],
        execute_tool: Callable[[str, Any], Awaitable[ToolResult]],
    ) -> None:
        self._memory_store = memory_store
        self._state_provider = state_provider
        self._complete_live = complete_live
        self._resolve_model = resolve_model
        self._execute_tool = execute_tool

    async def post_turn_slice(self, policy: SlicePolicy) -> Any:
        return await self._state_provider.slice(ctx=None, policy=policy)

    async def pre_turn_slice(self, policy: SlicePolicy) -> Any:
        return await self._state_provider.slice(ctx=None, policy=policy)

    async def explicit_slice(self, policy: SlicePolicy) -> Any:
        return await self._state_provider.slice(ctx=None, policy=policy)

    async def answer_memory_recall(
        self, *, query: str, user_slug: str, workspace_id: str,
        memory_scope: str = "personal", surface_id: str = "",
        limit: int = 5, **_kwargs: Any,
    ) -> MemoryRecallAnswer:
        ...  # moved from closure

    async def write_memory(self, request: MemoryWriteRequest) -> MemoryRecord:
        ...  # moved from closure

    async def list_memory_records(self, ...) -> list[MemoryRecord]:
        ...  # moved from closure

    async def complete(self, **kwargs: Any) -> StubCompletion:
        return await self._complete_live(**kwargs)

    def resolve_model_hook(self, model_id: str) -> StubModelDefinition:
        return self._resolve_model(model_id)

    async def execute_tool_hook(self, tool_id: str, ctx: Any) -> ToolResult:
        return await self._execute_tool(tool_id, ctx)

    def to_turn_hooks(self, context: dict[str, Any] | None = None) -> TurnHooks:
        """Build the TurnHooks dataclass from this provider's methods."""
        ...
```

### Refactored _build_turn_hooks

```python
def _build_turn_hooks(context: dict[str, Any] | None = None) -> TurnHooks:
    """Build TurnHooks wired to Little Lisa callbacks and memory operations."""
    memory_store, state_provider = _build_memory_and_state_provider()
    provider = LittleLisaTurnHooks(
        memory_store=memory_store,
        state_provider=state_provider,
        complete_live=_complete_live,
        resolve_model=_resolve_model,
        execute_tool=_execute_tool,
    )
    return provider.to_turn_hooks(context=context)
```

### Helper extraction

```python
def _build_memory_and_state_provider() -> tuple[MCPMemoryClient, LittleLisaStateProvider]:
    """Construct the MCP memory client and state provider for turn hooks."""
    ...
```

### Module placement

`LittleLisaTurnHooks` class goes in a new `turn_hooks.py` module.
`_build_turn_hooks` stays in `main.py` as a thin wrapper for test
compatibility.

## Test seam preservation

- `_build_turn_hooks` stays on `main.py`'s surface with the same signature.
- Tests using `monkeypatch.setattr(little_lisa, "_build_turn_hooks", fake)`
continue to work.
- `LittleLisaStateProvider` can also move to `turn_hooks.py` or stay in
`main.py` — decide during implementation.

## Verification

1. `ruff check examples/little_lisa/turn_hooks.py` — zero violations.
2. `ruff check examples/little_lisa/main.py --select C901,PLR0915` — zero
   violations for `_build_turn_hooks`.
3. `pytest examples/little_lisa/ -m "not live" -p no:randomly` — 99 passed,
   7 pre-existing failures unchanged.

## Out of scope

- Changing hook behavior or memory recall ranking algorithm.
- Moving `LittleLisaStateProvider` if it adds complexity (assess during impl).