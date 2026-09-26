"""Representative session routing vocabulary shared by kernel and SDK tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from kernel.routing_vocabulary import register_routing_vocabulary, reset_routing_vocabulary

__all__ = ["representative_routing_vocabulary"]


@contextmanager
def representative_routing_vocabulary() -> Iterator[None]:
    """Register a session-shaped vocabulary for the block, then clear it."""
    reset_routing_vocabulary()
    register_routing_vocabulary(
        sequential_connectors=["then", "and then", "after that", "next", "followed by"],
        producer_intents=["research", "gather", "learn", "read", "analyze", "review", "architect"],
        consumer_intents=["code", "write", "implement", "review"],
        resume_phrases=[
            "proceed",
            "continue",
            "go ahead",
            "go for it",
            "do it",
            "carry on",
            "pick up",
            "resume",
            "next step",
            "keep going",
        ],
        topic_abandonment_markers=[
            "side quest",
            "rewind",
            "never mind",
            "forget",
            "different topic",
            "not about",
        ],
    )
    try:
        yield
    finally:
        reset_routing_vocabulary()
