"""Scripted model for the memory pack's dedup-recall e2e scenario.

Drives the real memory-pack operation surface with zero credentials: the
model calls the ``synthesize_and_store_learning`` operation (the canonical
capture surface) with content, the operation's distill call comes back to
this script as an auxiliary request (response_format=DistilledLearning)
which answers with a create decision, and the second turn recalls via the
``recall_memory`` operation. Everything below the model — routing, tool
surface, dispatch, dedup-candidate search, memory.add, receipts,
persistence — is production code.

See ``e2etest.scripted`` for the ScriptTurn/Reply contract.
"""

from __future__ import annotations

import json

from e2etest.scripted import Reply, ScriptError, ScriptTurn

LEARNING = "The lisa harness memory pack dedups facts at write time."


def auxiliary(turn: ScriptTurn) -> Reply:
    """Tool-less requests: structured calls get a dedup-decided payload."""
    fmt = turn.request.response_format
    if fmt is None:
        return Reply.text("")
    name = getattr(fmt, "__name__", "") or str(fmt)
    if "DistilledLearning" in name:
        return Reply.text(
            json.dumps(
                {"facts": [{"action": "create", "fact": LEARNING, "reason": "new preference"}]}
            )
        )
    if "MinedFacts" in name:
        return Reply.text(json.dumps({"facts": []}))
    return Reply.text("{}")


def respond(turn: ScriptTurn) -> Reply:
    if "synthesize_and_store_learning" not in turn.tool_names:
        raise ScriptError(
            f"synthesize_and_store_learning not offered; tools={sorted(turn.tool_names)}"
        )

    if turn.count("synthesize_and_store_learning") == 0:
        return Reply.call(
            "synthesize_and_store_learning",
            content=f"User asked:\nremember the dedup behavior\n\nLisa answered:\n{LEARNING}",
            memory_tier="long_term",
        )

    if turn.count("recall_memory") == 0:
        return Reply.call("recall_memory", query="what does the harness dedup at write time")

    last = turn.last("recall_memory")
    if last is None or LEARNING not in last.text:
        raise ScriptError(
            f"recall_memory did not surface the stored fact: {last.text if last else None}"
        )
    return Reply.text(
        "Recall confirmed: the stored learning about write-time dedup surfaced "
        "from memory with lexical overlap."
    )
