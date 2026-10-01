"""plan_workflow's planning budget comes from lisa-pack-core session configuration."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest
from sdk.packs.core.operations.planning.plan_workflow import _PLAN_MAX_TOKENS, _plan_max_tokens


def _ctx(config: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(pack_config=MappingProxyType(config))


def test_plan_max_tokens_defaults_when_session_omits_it() -> None:
    assert _plan_max_tokens(_ctx({})) == _PLAN_MAX_TOKENS  # type: ignore[arg-type]


def test_plan_max_tokens_reads_session_configuration() -> None:
    assert _plan_max_tokens(_ctx({"plan_max_tokens": 65536})) == 65536  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, "big", True, 1.5])
def test_plan_max_tokens_rejects_malformed_values(value: object) -> None:
    with pytest.raises(ValueError, match="plan_max_tokens"):
        _plan_max_tokens(_ctx({"plan_max_tokens": value}))  # type: ignore[arg-type]
