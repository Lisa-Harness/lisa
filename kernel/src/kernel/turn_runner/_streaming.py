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
    the model's substantive answer lives in ``reasoning_content``. We fold
    reasoning into ``content`` here (when content is empty) so the rest of
    the turn runner — which reads ``completion.content`` — sees the model's
    actual answer instead of falling through to "did not produce a final
    response." The raw reasoning is still available via
    ``self.reasoning_content`` for callers that want to distinguish the two.
    """

    __slots__ = ("_aggregated", "content", "duration_ms", "reasoning_content", "tool_calls", "usage")

    def __init__(self, aggregated: StreamedCompletion) -> None:
        self._aggregated = aggregated
        self.reasoning_content = aggregated.reasoning_content
        # Fold reasoning into content when the visible answer is empty.
        self.content = aggregated.content or aggregated.reasoning_content
        self.tool_calls = aggregated.tool_calls
        self.usage = aggregated.usage
        # Surfaced flat for the turn-runner's duration extraction; the stream
        # carries no per-chunk duration, so the caller measures and stamps it.
        self.duration_ms = 0.0

    def __repr__(self) -> str:
        return (
            f"StreamedCompletionAdapter(content={self.content!r}, "
            f"reasoning_content={self.reasoning_content!r}, "
            f"finish_reason={self._aggregated.finish_reason!r})"
        )
