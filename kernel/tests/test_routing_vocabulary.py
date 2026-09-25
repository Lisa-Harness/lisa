"""Tests for the session-injected routing-vocabulary registry.

Hard cutover (2026-08-11): the kernel has no hardcoded fallbacks. A
reader raises ``RuntimeError`` when no session vocabulary has been
wired. The session defines the vocabulary or boot fails.
"""

from __future__ import annotations

import pytest
from kernel.routing_vocabulary import (
    get_consumer_intents,
    get_producer_intents,
    get_resume_phrases,
    get_sequential_connectors,
    get_topic_abandonment_markers,
    register_routing_vocabulary,
    reset_routing_vocabulary,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Reset the registry before and after each test."""
    reset_routing_vocabulary()
    yield
    reset_routing_vocabulary()


# ── No fallback: unwired readers raise ──────────────────────────────────


def test_get_sequential_connectors_raises_when_unwired() -> None:
    """With no session vocabulary registered, the reader raises."""
    with pytest.raises(RuntimeError, match="sequential connectors not wired"):
        get_sequential_connectors()


def test_get_producer_intents_raises_when_unwired() -> None:
    """With no session vocabulary registered, the reader raises."""
    with pytest.raises(RuntimeError, match="producer intents not wired"):
        get_producer_intents()


def test_get_consumer_intents_raises_when_unwired() -> None:
    """With no session vocabulary registered, the reader raises."""
    with pytest.raises(RuntimeError, match="consumer intents not wired"):
        get_consumer_intents()


def test_get_resume_phrases_raises_when_unwired() -> None:
    """With no session vocabulary registered, the reader raises."""
    with pytest.raises(RuntimeError, match="resume phrases not wired"):
        get_resume_phrases()


# ── Session registration is authoritative ────────────────────────────────


def test_register_sequential_connectors_replaces_registry() -> None:
    """A registered session vocabulary is the sole source — no fallback mix-in."""
    register_routing_vocabulary(sequential_connectors=["then", "based on"])
    connectors = get_sequential_connectors()
    assert " then " in connectors
    assert " based on " in connectors


def test_register_sequential_connectors_normalizes_entries() -> None:
    """Entries are stripped, lowercased, padded, de-duplicated."""
    register_routing_vocabulary(sequential_connectors=["  Then  ", "BASED ON", "then", "", "  "])
    assert get_sequential_connectors() == (" then ", " based on ")


def test_register_sequential_connectors_none_leaves_unwired() -> None:
    """Passing ``None`` leaves the vocabulary unset — the reader raises."""
    register_routing_vocabulary(sequential_connectors=["based on"])
    assert " based on " in get_sequential_connectors()

    register_routing_vocabulary(sequential_connectors=None)
    with pytest.raises(RuntimeError):
        get_sequential_connectors()

    register_routing_vocabulary(sequential_connectors=["based on"])
    register_routing_vocabulary(sequential_connectors=[])
    with pytest.raises(RuntimeError):
        get_sequential_connectors()


def test_reset_routing_vocabulary_clears_session_list() -> None:
    """``reset_routing_vocabulary`` returns to the unwired state (raises)."""
    register_routing_vocabulary(sequential_connectors=["based on"])
    reset_routing_vocabulary()
    with pytest.raises(RuntimeError):
        get_sequential_connectors()


# ── Resume-phrase vocabulary ───────────────────────────────────────────


def test_register_resume_phrases_replaces_registry() -> None:
    """A registered resume-phrase list is the sole source."""
    register_routing_vocabulary(resume_phrases=["proceed", "keep going"])
    phrases = get_resume_phrases()
    assert "proceed" in phrases
    assert "keep going" in phrases


def test_register_resume_phrases_normalizes_entries() -> None:
    """Entries are stripped, lowercased, de-duplicated (no space padding)."""
    register_routing_vocabulary(resume_phrases=["  Proceed  ", "KEEP GOING", "proceed", "", "  "])
    assert get_resume_phrases() == ("proceed", "keep going")


def test_register_resume_phrases_none_leaves_unwired() -> None:
    """Passing ``None`` leaves the vocabulary unset — the reader raises."""
    register_routing_vocabulary(resume_phrases=["keep going"])
    assert "keep going" in get_resume_phrases()

    register_routing_vocabulary(resume_phrases=None)
    with pytest.raises(RuntimeError):
        get_resume_phrases()

    register_routing_vocabulary(resume_phrases=["keep going"])
    register_routing_vocabulary(resume_phrases=[])
    with pytest.raises(RuntimeError):
        get_resume_phrases()


def test_reset_routing_vocabulary_clears_resume_phrases() -> None:
    """``reset_routing_vocabulary`` returns resume phrases to unwired (raises)."""
    register_routing_vocabulary(resume_phrases=["keep going"])
    reset_routing_vocabulary()
    with pytest.raises(RuntimeError):
        get_resume_phrases()


# ── Producer/consumer intent roles ──────────────────────────────────────


def test_register_producer_intents_replaces_registry() -> None:
    """A registered producer-intent set is the sole source."""
    register_routing_vocabulary(producer_intents=["research", "review"])
    assert get_producer_intents() == frozenset({"research", "review"})


def test_register_consumer_intents_replaces_registry() -> None:
    """A registered consumer-intent set is the sole source."""
    register_routing_vocabulary(consumer_intents=["code", "write", "review"])
    assert get_consumer_intents() == frozenset({"code", "write", "review"})


def test_register_intent_roles_none_leaves_unwired() -> None:
    """Passing ``None`` leaves the vocabulary unset — the reader raises."""
    register_routing_vocabulary(producer_intents=["research"])
    assert get_producer_intents() == frozenset({"research"})

    register_routing_vocabulary(producer_intents=None)
    with pytest.raises(RuntimeError):
        get_producer_intents()

    register_routing_vocabulary(consumer_intents=["code"])
    assert get_consumer_intents() == frozenset({"code"})

    register_routing_vocabulary(consumer_intents=[])
    with pytest.raises(RuntimeError):
        get_consumer_intents()


# ── Topic-abandonment markers ──────────────────────────────────────────


def test_get_topic_abandonment_markers_raises_when_unwired() -> None:
    """With no session vocabulary registered, the reader raises."""
    with pytest.raises(RuntimeError, match="topic abandonment markers not wired"):
        get_topic_abandonment_markers()


def test_register_topic_abandonment_markers_replaces_registry() -> None:
    """A registered marker list is the sole source — no fallback mix-in."""
    register_routing_vocabulary(topic_abandonment_markers=["side quest", "rewind"])
    markers = get_topic_abandonment_markers()
    assert "side quest" in markers
    assert "rewind" in markers


def test_register_topic_abandonment_markers_normalizes_entries() -> None:
    """Entries are stripped, lowercased, de-duplicated (no space padding)."""
    register_routing_vocabulary(
        topic_abandonment_markers=["  Side Quest  ", "REWIND", "side quest", "", "  "]
    )
    assert get_topic_abandonment_markers() == ("side quest", "rewind")


def test_register_topic_abandonment_markers_none_leaves_unwired() -> None:
    """Passing ``None`` leaves the vocabulary unset — the reader raises."""
    register_routing_vocabulary(topic_abandonment_markers=["rewind"])
    assert "rewind" in get_topic_abandonment_markers()

    register_routing_vocabulary(topic_abandonment_markers=None)
    with pytest.raises(RuntimeError):
        get_topic_abandonment_markers()

    register_routing_vocabulary(topic_abandonment_markers=["rewind"])
    register_routing_vocabulary(topic_abandonment_markers=[])
    with pytest.raises(RuntimeError):
        get_topic_abandonment_markers()


def test_reset_routing_vocabulary_clears_topic_abandonment_markers() -> None:
    """``reset_routing_vocabulary`` returns markers to unwired (raises)."""
    register_routing_vocabulary(topic_abandonment_markers=["rewind"])
    reset_routing_vocabulary()
    with pytest.raises(RuntimeError):
        get_topic_abandonment_markers()
