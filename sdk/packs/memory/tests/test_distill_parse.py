"""Regression tests for `_parse_distilled_learning`.

Covers the incident where a local OpenAI-compatible model (Ollama) returned
fenced JSON that failed structured validation, and the prose/bullet
fallbacks misinterpreted JSON syntax lines as facts — producing garbage
memory records like ``json | { | facts": [``.
"""

from __future__ import annotations

import pytest

from sdk.packs.memory.operations.memory.synthesize_learning import (
    DistilledLearning,
    _parse_distilled_learning,
)


def test_parse_valid_json_object() -> None:
    """A clean JSON object parses into facts directly."""
    payload = '{"facts": ["A is B", "C is D"]}'
    result = _parse_distilled_learning(payload)
    assert isinstance(result, DistilledLearning)
    assert result.facts == ["A is B", "C is D"]


def test_parse_fenced_json_object_strips_fence() -> None:
    """Fenced JSON (```json ... ```) is extracted and validated."""
    payload = '```json\n{"facts": ["one fact", "two fact"]}\n```'
    result = _parse_distilled_learning(payload)
    assert result.facts == ["one fact", "two fact"]


def test_parse_bullet_list_fallback() -> None:
    """When the model returns bullets (not JSON), the bullet fallback works."""
    payload = "- First durable fact\n- Second durable fact"
    result = _parse_distilled_learning(payload)
    assert result.facts == ["First durable fact", "Second durable fact"]


def test_parse_prose_fallback_for_plain_sentences() -> None:
    """Plain prose sentences (no JSON, no bullets) fall back to sentence split."""
    payload = "Python dominates the AI agent ecosystem. LangChain is the most used."
    result = _parse_distilled_learning(payload)
    assert len(result.facts) == 2
    assert "Python dominates" in result.facts[0]


def test_parse_rejects_fenced_json_that_failed_validation() -> None:
    """Regression: fenced JSON that fails validation must NOT fall through to
    the prose fallback, which would store JSON syntax fragments as facts.

    Before the fix, this input produced facts ``['json', '{', 'facts": [']``
    joined as ``json | { | facts": [`` — the exact garbage seen in the
    memory database on 2026-07-17.
    """
    # Fenced JSON truncated mid-array — will fail DistilledLearning validation
    # because `facts` is not a complete value. Looks like JSON → must raise.
    payload = '```json\n{\n  "facts": [\n```\n'
    with pytest.raises(RuntimeError, match="JSON-shaped"):
        _parse_distilled_learning(payload)


def test_parse_rejects_bare_broken_json() -> None:
    """Bare (unfenced) broken JSON must also raise, not fall back to prose."""
    payload = '{"facts": ['
    with pytest.raises(RuntimeError, match="JSON-shaped"):
        _parse_distilled_learning(payload)


def test_parse_empty_raises() -> None:
    """Empty distillation output raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        _parse_distilled_learning("")
