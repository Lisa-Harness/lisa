"""Cache-prefix oracle tests (cache-efficiency SPEC §7 criterion 2).

Scripted scenarios replay model iterations through ``CacheOracle`` and
assert the oracle properties: appends keep the carried digest, folds
move the digest only at their boundary, a fold plus a same-boundary
compaction causes no duplicate rewrite, mode handoff preserves tool
authorization and schema order, and a known-expired cache is
distinguishable from a changed prefix. Reports stay aggregate-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernel.cache_oracle import (
    BREAK_CACHE_EXPIRED,
    BREAK_IDENTICAL,
    BREAK_NO_PRIOR,
    BREAK_PREFIX_CHANGED,
    BREAK_PROVIDER_CHANGED,
    BREAK_TAIL_APPEND,
    BREAK_TOOLS_CHANGED,
    CacheOracle,
    apply_folds,
    explain,
    shape_digest,
    tools_digest,
)
from kernel.provider_contracts import CompletionRequest, ModelDescriptor, SystemPromptSegment

BASE_PROMPT = "invariant base instructions"
MODE_A_PROMPT = "mode a instructions"
MODE_B_PROMPT = "mode b instructions"
SECRET_PROMPT = "SECRET-BASE-INSTRUCTIONS-do-not-emit"
SECRET_QUESTION = "SECRET-USER-QUESTION-do-not-emit"


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
MODEL_B = _Descriptor(
    provider="anthropic",
    model_id="m2",
    display_name="M2",
    reasoning_model=False,
    thinking_model=False,
)


def request(
    messages,
    *,
    model=MODEL,
    tools=None,
    segments=((BASE_PROMPT, "stable"), (MODE_A_PROMPT, "ephemeral")),
):
    return CompletionRequest(
        messages=list(messages),
        model_def=model,
        max_tokens=512,
        temperature=0.0,
        tools=list(tools) if tools is not None else None,
        system_prompt_segments=tuple(
            SystemPromptSegment(text, stability) for text, stability in segments
        ),
    )


def turn(user, call_id, result):
    return [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "function": {"name": "read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def base_transcript(n_results):
    messages = []
    for i in range(n_results):
        messages.extend(
            turn(f"question {i}", f"call-{i}", f"tool result {i} " + "x" * (40 * (i + 1)))
        )
    return messages


def fold_entries():
    """Real supersession records: call-1's result is superseded by call-1b."""
    record = {
        "call_id": "call-1",
        "tool_id": "read",
        "scope": "task",
        "keys": ["r:a@1"],
        "chars": 1200,
        "ref": "r" + "a" * 64,
    }
    supersede = dict(record, call_id="call-1b", ref="r" + "b" * 64)
    return [("", {"records": [record, supersede], "snapshots": {}})]


def test_appending_tool_results_preserves_carried_digest():
    oracle = CacheOracle()
    obs1 = oracle.observe(request(base_transcript(2)), label="iter1")
    assert obs1.break_reason == BREAK_NO_PRIOR

    appended = base_transcript(2) + turn("follow up", "call-new", "fresh result")
    obs2 = oracle.observe(request(appended), label="iter2")
    assert obs2.break_reason == BREAK_TAIL_APPEND
    assert obs2.appended_entries == 3
    assert obs2.shape_digest != obs1.shape_digest
    # Everything before the append is exactly the previously seen request.
    assert obs2.carried_digest == obs1.shape_digest


def test_fold_moves_digest_only_at_boundary():
    oracle = CacheOracle()
    trans = base_transcript(3)
    oracle.observe(request(trans), label="pre-fold")

    folded, replacements = apply_folds(trans, fold_entries())
    # Real fold machinery produced a real stub.
    assert set(replacements) == {"call-1"}
    stub = replacements["call-1"]
    assert stub.endswith("with thread_read_content]")
    assert "1200 chars" in stub
    assert any(
        m.get("role") == "tool" and m.get("tool_call_id") == "call-1" and m["content"] == stub
        for m in folded
    )

    obs = oracle.observe(request(folded), label="post-fold")
    assert obs.break_reason == BREAK_PREFIX_CHANGED
    # The digest moved at the fold boundary, not at the head of the prompt:
    # entries 0-6 (2 system segments + turn 0 + turn 1's user/assistant rows)
    # are unchanged; the folded tool result is the first rewritten entry.
    assert obs.first_divergence == 7


def test_fold_and_compaction_same_boundary_no_duplicate_rewrite():
    oracle = CacheOracle()
    trans = base_transcript(3)
    oracle.observe(request(trans), label="iter")

    folded, _ = apply_folds(trans, fold_entries())
    obs_fold = oracle.observe(request(folded), label="folded")
    assert obs_fold.break_reason == BREAK_PREFIX_CHANGED

    # A compaction targeting the same boundary rewrites nothing further.
    refolded, _replacements_again = apply_folds(folded, fold_entries())
    assert refolded == folded
    obs_both = oracle.observe(request(refolded), label="folded+compacted")
    assert obs_both.break_reason == BREAK_IDENTICAL
    assert obs_both.first_divergence is None


def test_mode_handoff_preserves_tools_and_stable_prefix():
    oracle = CacheOracle()
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
    ]
    first = oracle.observe(request(base_transcript(1), tools=tools), label="mode-a")

    handoff = request(
        base_transcript(1),
        tools=tools,
        segments=((BASE_PROMPT, "stable"), (MODE_B_PROMPT, "ephemeral")),
    )
    obs = oracle.observe(handoff, label="mode-b")
    # Only the ephemeral mode segment moved; the stable base prefix held.
    assert obs.break_reason == BREAK_PREFIX_CHANGED
    assert obs.first_divergence == 1
    assert obs.tools_digest == first.tools_digest

    # Widening or reordering schemas is a distinct, detected break.
    widened = request(
        base_transcript(1),
        tools=[*tools, {"type": "function", "function": {"name": "write_file"}}],
        segments=((BASE_PROMPT, "stable"), (MODE_B_PROMPT, "ephemeral")),
    )
    obs_wide = oracle.observe(widened, label="widened")
    assert obs_wide.break_reason == BREAK_TOOLS_CHANGED
    reordered = request(
        base_transcript(1),
        tools=[{"type": "function", "function": {"name": "write_file"}}, *tools],
        segments=((BASE_PROMPT, "stable"), (MODE_B_PROMPT, "ephemeral")),
    )
    obs_reordered = oracle.observe(reordered, label="reordered")
    assert obs_reordered.break_reason == BREAK_TOOLS_CHANGED


def test_provider_change_is_reported_separately():
    oracle = CacheOracle()
    oracle.observe(request(base_transcript(1)), label="m1")
    obs = oracle.observe(request(base_transcript(1), model=MODEL_B), label="m2")
    assert obs.break_reason == BREAK_PROVIDER_CHANGED
    assert obs.break_detail == "shape_changed=False"


def test_expired_cache_is_distinguishable_from_prefix_change():
    clock = iter([0.0, 10.0, 11.0])
    oracle = CacheOracle(ttl_seconds=5.0, clock=lambda: next(clock))
    trans = base_transcript(1)
    obs1 = oracle.observe(request(trans), label="t0")
    assert obs1.break_reason == BREAK_NO_PRIOR
    assert obs1.cache_ttl_state == "n/a"

    # Same shape past the TTL: expired cache, not a changed prefix.
    obs2 = oracle.observe(request(trans), label="t10")
    assert obs2.break_reason == BREAK_CACHE_EXPIRED
    assert obs2.cache_ttl_state == "expired"
    assert obs2.first_divergence is None
    assert obs2.carried_digest == obs1.shape_digest

    # A genuinely changed prefix still reports the change; this request
    # arrives within the previous observation's TTL window, so it is fresh.
    obs3 = oracle.observe(request(trans + turn("more", "call-x", "result")), label="t11")
    assert obs3.break_reason == BREAK_TAIL_APPEND
    assert obs3.cache_ttl_state == "fresh"


def test_identical_request_within_ttl_is_fresh():
    clock = iter([0.0, 1.0])
    oracle = CacheOracle(ttl_seconds=5.0, clock=lambda: next(clock))
    oracle.observe(request(base_transcript(1)), label="t0")
    obs = oracle.observe(request(base_transcript(1)), label="t1")
    assert obs.break_reason == BREAK_IDENTICAL
    assert obs.cache_ttl_state == "fresh"


def test_shape_digest_is_deterministic_and_typed():
    assert shape_digest(request(base_transcript(1))) == shape_digest(request(base_transcript(1)))
    # Same rendered text, different cache typing → different canonical shape.
    as_stable = request(base_transcript(1), segments=((MODE_A_PROMPT, "stable"),))
    as_ephemeral = request(base_transcript(1), segments=((MODE_A_PROMPT, "ephemeral"),))
    assert shape_digest(as_stable) != shape_digest(as_ephemeral)
    # Absent and empty tool lists are the same canonical shape.
    assert tools_digest(request([], tools=None)) == tools_digest(request([], tools=[]))
    # Non-cacheable roles are not part of the carried shape.
    with_extra_role = request([{"role": "developer", "content": "note"}, *base_transcript(1)])
    assert shape_digest(with_extra_role) == shape_digest(request(base_transcript(1)))


def test_report_carries_digests_not_prompt_text():
    oracle = CacheOracle()
    obs = oracle.observe(
        request(
            [{"role": "user", "content": SECRET_QUESTION}],
            segments=((SECRET_PROMPT, "stable"),),
        ),
        label="secret-iteration",
    )
    report = "\n".join(oracle.render_report())
    assert SECRET_PROMPT not in report
    assert SECRET_QUESTION not in report
    assert "carried=" in report and "shape=" in report
    assert SECRET_PROMPT not in explain(obs)
