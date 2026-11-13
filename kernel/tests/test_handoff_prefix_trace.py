"""E2E prefix trace for the safe-handoff prompt layout (SPEC §6).

Drives the layouts through the real ``CacheOracle``:

- baseline (today): the merged stable base means a mid-turn handoff
  moves the first divergent canonical entry to 0 — the invariant head
  is lost with the mode tail;
- target (factored S0/S1): the invariant head survives the handoff;
  the first divergence lands on the mode segment, and the carried
  digest equals the shape digest of the invariant segment alone.

The target assertions previously simulated the factored layout with
hand-built segment tuples; they are now driven end-to-end through
``run_turn`` with ``ContextHooks.assemble_system_prompt_parts`` wired
and a real ``transition_stage`` tool call, so the acceptance gate for
the implementation slice exercises the production handoff path. Tool
authorization changes stay observable as tool-digest changes in both
layouts; nothing unions or reorders schemas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from kernel.cache_oracle import (
    BREAK_IDENTICAL,
    BREAK_NO_PRIOR,
    BREAK_PREFIX_CHANGED,
    BREAK_TOOLS_CHANGED,
    CacheOracle,
    shape_digest,
    tools_digest,
)
from kernel.provider_contracts import CompletionRequest, ModelDescriptor, SystemPromptSegment
from kernel.testing.gateway_stub import StubGatewayProvider
from kernel.turn_hooks import ContextHooks, ModelHooks, OperationHooks, TurnHooks
from kernel.turn_runner import ContextSeams, run_turn


class _StubCompletion:
    def __init__(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


async def _stub_execute_operation(*_: Any, **__: Any) -> Any:
    class _R:
        output_text = ""
        output_json: ClassVar[dict[str, Any]] = {}

    return _R()


def _make_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _transition_call(call_id: str, target_stage_id: str, **extra: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"target_stage_id": target_stage_id}
    args.update(extra)
    return _make_tool_call(call_id, "transition_stage", args)


HARNESS = "harness rules"
SESSION = "session instruction"
COPILOT = "copilot customizations"
MODE_A = "mode a instructions"
MODE_B = "mode b instructions"
S0_TEXT = "\n\n".join([HARNESS, SESSION, COPILOT])
ALL_A = "\n\n".join([HARNESS, SESSION, COPILOT, MODE_A])
ALL_B = "\n\n".join([HARNESS, SESSION, COPILOT, MODE_B])

READ_TOOL = {
    "type": "function",
    "function": {"name": "read_file", "parameters": {"type": "object"}},
}
WRITE_TOOL = {
    "type": "function",
    "function": {"name": "write_file", "parameters": {"type": "object"}},
}


@dataclass(frozen=True)
class _Descriptor(ModelDescriptor):
    """Concrete stand-in for the ``ModelDescriptor`` protocol."""

    provider: str
    model_id: str
    display_name: str
    reasoning_model: bool
    thinking_model: bool


MODEL = _Descriptor(
    provider="openai", model_id="m1", display_name="M1", reasoning_model=False, thinking_model=False
)


def _request(segments, tools, transcript=None):
    return CompletionRequest(
        messages=[{"role": "user", "content": "transcript body"}]
        if transcript is None
        else list(transcript),
        model_def=MODEL,
        max_tokens=512,
        temperature=0.0,
        tools=list(tools),
        system_prompt_segments=tuple(
            SystemPromptSegment(text, stability) for text, stability in segments
        ),
    )


def test_baseline_trace_merged_base_loses_invariant_head_on_handoff():
    """Today's layout: handoff diverges at entry 0 — nothing survives."""
    oracle = CacheOracle()
    first = oracle.observe(_request([(ALL_A, "stable")], [READ_TOOL]), label="mode-a")
    assert first.break_reason == BREAK_NO_PRIOR

    handoff = oracle.observe(_request([(ALL_B, "stable")], [READ_TOOL]), label="mode-b")
    assert handoff.break_reason == BREAK_PREFIX_CHANGED
    assert handoff.first_divergence == 0
    # The tool surface is byte-identical here, so the whole break is the
    # merged base — proof the layout, not authorization, causes the loss.
    assert handoff.tools_digest == first.tools_digest


def test_target_trace_factored_base_keeps_invariant_head_on_handoff():
    """Factored S0/S1: handoff diverges at the mode segment; S0 survives."""
    oracle = CacheOracle()
    oracle.observe(_request([(S0_TEXT, "stable"), (MODE_A, "stable")], [READ_TOOL]), label="mode-a")

    handoff = oracle.observe(
        _request([(S0_TEXT, "stable"), (MODE_B, "stable")], [READ_TOOL]), label="mode-b"
    )
    assert handoff.break_reason == BREAK_PREFIX_CHANGED
    assert handoff.first_divergence == 1
    # The carried region is exactly the invariant head: its digest equals
    # the shape digest of the S0 segment alone — the same segment with an
    # empty transcript, so no other canonical entry joins the digest.
    s0_only = _request([(S0_TEXT, "stable")], [READ_TOOL], transcript=[])
    assert handoff.carried_digest == shape_digest(s0_only)


@pytest.mark.asyncio
async def test_real_handoff_swaps_only_the_mode_tail():
    """Through ``run_turn``: the handoff rewrites S1; S0 bytes survive.

    This is the production-path form of the acceptance gate above — no
    hand-built segment tuples. The second call's stable segments are
    exactly ``[S0, MODE_B]``, which makes the oracle's first divergence
    land on the mode entry with the carried digest equal to the
    invariant-head-only shape digest.
    """
    captured: list[list[tuple[str, str]]] = []
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        provider = _provider_ref[0]
        assert provider is not None and provider.last_gateway is not None
        request = provider.last_gateway.last_request
        assert request is not None
        captured.append([(s.text, s.stability) for s in request.system_prompt_segments])
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_transition_call("call_1", "markdown")],
            )
        return _StubCompletion(content="Done.")

    _provider_ref: list[StubGatewayProvider | None] = [None]
    provider = StubGatewayProvider(two_step)
    _provider_ref[0] = provider

    hook_calls = 0

    def parts(_mode_instruction: str, _stage_instruction: str | None) -> tuple[str, str]:
        nonlocal hook_calls
        hook_calls += 1
        # First call seeds the entry mode's tail; later calls are the
        # handoff re-deriving the target mode's tail. The kernel passes
        # the real catalog intros, so the test models the tail change by
        # call order rather than matching instruction text.
        return S0_TEXT, (MODE_A if hook_calls == 1 else MODE_B)

    hooks = TurnHooks(
        models=ModelHooks(complete=two_step),
        operations=OperationHooks(execute_operation=_stub_execute_operation),
        context=ContextHooks(assemble_system_prompt_parts=parts),
        context_seams=ContextSeams(provider_gateways=provider),
    )
    result = await run_turn(mode_id="default", query="write", hooks=hooks, max_tool_iterations=5)
    assert result.handoff_to == "markdown"
    assert len(captured) >= 2

    first, second = captured[0], captured[-1]
    assert first[:2] == [(S0_TEXT, "stable"), (MODE_A, "stable")]
    assert second[:2] == [(S0_TEXT, "stable"), (MODE_B, "stable")]

    oracle = CacheOracle()
    # Script only the stable entries: real requests carry ephemeral
    # dynamics (routing hint, handoff context) that belong after the
    # cache boundary and would shift the divergence index. The tool
    # surface is held fixed to isolate the layout claim (the
    # tool-authorization case has its own test below).
    oracle.observe(_request(first[:2], [READ_TOOL], transcript=[]), label="pre")
    handoff = oracle.observe(_request(second[:2], [READ_TOOL], transcript=[]), label="post")
    assert handoff.break_reason == BREAK_PREFIX_CHANGED
    assert handoff.first_divergence == 1
    s0_only = _request([(S0_TEXT, "stable")], [READ_TOOL], transcript=[])
    assert handoff.carried_digest == shape_digest(s0_only)


def test_target_trace_identical_base_is_identical_request():
    """Mode re-entry with unchanged bytes causes no break at all."""
    oracle = CacheOracle()
    oracle.observe(_request([(S0_TEXT, "stable"), (MODE_A, "stable")], [READ_TOOL]), label="t0")
    again = oracle.observe(
        _request([(S0_TEXT, "stable"), (MODE_A, "stable")], [READ_TOOL]), label="t1"
    )
    assert again.break_reason == BREAK_IDENTICAL


def test_trace_tool_authorization_change_stays_observable():
    """No layout work masks a widening: the tool digest declares the break."""
    oracle = CacheOracle()
    oracle.observe(_request([(S0_TEXT, "stable"), (MODE_A, "stable")], [READ_TOOL]), label="a")
    widened = oracle.observe(
        _request([(S0_TEXT, "stable"), (MODE_A, "stable")], [READ_TOOL, WRITE_TOOL]),
        label="widened",
    )
    assert widened.break_reason == BREAK_TOOLS_CHANGED
    assert widened.tools_digest != tools_digest(_request([], [READ_TOOL]))
