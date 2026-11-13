"""E2E prefix trace for the safe-handoff prompt layout (SPEC §6).

The design deliverable requires a prefix trace **before** implementation.
This module scripts both layouts through the real ``CacheOracle``:

- baseline (today): the merged stable base means a mid-turn handoff
  moves the first divergent canonical entry to 0 — the invariant head
  is lost with the mode tail;
- target (factored S0/S1): the invariant head survives the handoff;
  the first divergence lands on the mode segment, and the carried
  digest equals the shape digest of the invariant segment alone.

Tool authorization changes stay observable as tool-digest changes in
both layouts; nothing unions or reorders schemas. The target-layout
assertions are the acceptance gate for the implementation slice.
"""

from __future__ import annotations

from dataclasses import dataclass

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
