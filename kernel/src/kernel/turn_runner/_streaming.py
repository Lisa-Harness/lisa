"""StreamedCompletionAdapter — makes a StreamedCompletion look like a CompletionResult."""

from __future__ import annotations

from kernel.provider_contracts import StreamedCompletion

# ── Streaming adapter ──────────────────────────────────────────────────────


class StreamedCompletionAdapter:
    """Adapter that makes a ``StreamedCompletion`` look like a ``CompletionResult``.

    The turn runner's tool loop expects ``completion.content``,
    ``completion.tool_calls``, and ``completion.usage``. When the streaming
    path produces a ``StreamedCompletion`` (no tool calls in v1), this adapter
    bridges the two shapes without forcing the streaming code to construct a
    full ``CompletionResult``.

    For gpt-5.x reasoning models the visible ``content`` can be empty while
    the model's substantive answer lives in ``reasoning_content``. This
    adapter keeps the two SEPARATE: ``content`` stays the visible answer and
    ``reasoning_content`` carries the thinking. Folding reasoning into
    ``content`` here would defeat the kernel's user-facing guards, which rely
    on distinguishing a real answer from leaked chain-of-thought so they can
    bound it (the 2026-10-04 non-streaming 119KB monologue incident). Callers
    that need the folded view (structured-output parsers) use
    ``completion_content``, which folds ``reasoning_content`` when ``content``
    is empty.
    """

    __slots__ = (
        "_aggregated",
        "content",
        "duration_ms",
        "finish_reason",
        "reasoning_content",
        "tool_calls",
        "usage",
    )

    def __init__(self, aggregated: StreamedCompletion) -> None:
        self._aggregated = aggregated
        self.reasoning_content = aggregated.reasoning_content
        # Keep the visible answer and the reasoning SEPARATE — the kernel's
        # display sites bound leaked reasoning via ``reasoning_tail_excerpt``,
        # which only fires when ``content`` is empty for a reasoning-only turn.
        self.content = aggregated.content
        self.tool_calls = aggregated.tool_calls
        self.usage = aggregated.usage
        # Surfaced flat for the same diagnostics the non-streaming
        # ``CompletionResult`` carries: the reasoning-tail caveat's
        # truncation clause and the turn-loop empty-completion warnings read
        # ``finish_reason`` via ``getattr``. The 2026-10-13 turn 789 streamed
        # burnout caveat reported ``finish_reason=unknown`` because the
        # adapter hid the aggregate's termination signal.
        self.finish_reason = aggregated.finish_reason
        # Surfaced flat for the turn-runner's duration extraction; the stream
        # carries no per-chunk duration, so the caller measures and stamps it.
        self.duration_ms = 0.0

    def __repr__(self) -> str:
        return (
            f"StreamedCompletionAdapter(content={self.content!r}, "
            f"reasoning_content={self.reasoning_content!r}, "
            f"finish_reason={self._aggregated.finish_reason!r})"
        )
