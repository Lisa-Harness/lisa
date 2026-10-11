"""Session token gate counts processed tokens, not cached prefix reads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from runtime.handle_message import _session_budget_blocked
from runtime.runtime import TokenCounters


def _runtime(cap: int, total: int, cached: int) -> Any:
    tokens = TokenCounters(session_total_tokens=total, session_cached_tokens=cached)
    execution = SimpleNamespace(max_session_tokens=cap)
    return cast(
        Any,
        SimpleNamespace(
            session_plan=SimpleNamespace(manifest=SimpleNamespace(execution=execution)),
            session=SimpleNamespace(tokens=tokens),
        ),
    )


def test_cached_prefix_reads_do_not_exhaust_session_budget() -> None:
    assert _session_budget_blocked(_runtime(cap=1000, total=5000, cached=4500)) is None


def test_processed_tokens_exhaust_session_budget() -> None:
    message = _session_budget_blocked(_runtime(cap=1000, total=5000, cached=3900))
    assert message is not None
    assert "1,100 / 1,000" in message
