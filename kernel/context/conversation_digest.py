"""Rolling conversation digest for every-turn situational awareness.

This module maintains the medium-short thread layer that sits between raw
recent turns and durable IPSAR memories. It is called by ``app.agent`` after a
turn is persisted and stores the result on ``ThreadState`` so privacy isolation
uses the same scoped thread keys as the raw conversation history.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from kernel.models import ConversationDigest, LoadedContextSnapshot, ThreadMessage
from pydantic import BaseModel, Field

import app.model_policy as model_policy
import app.providers as providers
from app.config import settings
from app.content_registry import read_authored_text
from app.context_keys import DIGEST_HYGIENE_CONTEXT_KEYS

logger = logging.getLogger(__name__)

_DIGEST_TOKEN_RE = re.compile(r"[a-z0-9']+")
_DIGEST_SEGMENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_DIGEST_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "you",
    "your",
}


class _ConversationDigestSynthesis(BaseModel):
    """Strict model output for one rolling digest refresh."""

    summary: str = Field(default="")
    open_loops: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def _load_digest_system_prompt() -> str:
    """Load the authored system prompt for conversation digest refresh."""
    return read_authored_text("prompts/conversation_digest/system.md").strip()


@lru_cache(maxsize=1)
def _load_digest_user_prompt_template() -> str:
    """Load the authored user prompt template for conversation digest refresh."""
    return read_authored_text("prompts/conversation_digest/user.md").strip()


def _turn_timestamp(turn: ThreadMessage) -> str:
    created_at = turn.created_at
    if isinstance(created_at, datetime):
        value = created_at
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return str(created_at or "")


def _tokenize_digest_text(text: str) -> set[str]:
    """Return content tokens for digest-side overlap detection."""

    return {
        match.group(0).lower()
        for match in _DIGEST_TOKEN_RE.finditer(text.lower())
        if match.group(0).lower() not in _DIGEST_STOPWORDS
    }


def _normalize_conditioning_text(text: str) -> str:
    """Strip frame markup so digest hygiene compares semantic conditioning text."""

    normalized_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        line = re.sub(r"^[\-*\d\.\)\s]+", "", line).strip()
        if line:
            normalized_lines.append(line)
    return " ".join(normalized_lines)


def _coerce_loaded_context_snapshot(turn: ThreadMessage) -> LoadedContextSnapshot | None:
    """Return a loaded-context snapshot attached to one assistant turn when present."""

    payload = turn.extra.get("loaded_context_snapshot")
    if isinstance(payload, LoadedContextSnapshot):
        return payload
    if isinstance(payload, dict):
        try:
            return LoadedContextSnapshot.model_validate(payload)
        except Exception:  # pylint: disable=broad-except
            return None
    return None


def _conditioning_keys_and_tokens(turn: ThreadMessage) -> tuple[list[str], set[str]]:
    """Return disambiguated conditioning keys plus comparison tokens for digest hygiene."""

    snapshot = _coerce_loaded_context_snapshot(turn)
    if snapshot is None:
        return [], set()

    keys: list[str] = []
    tokens: set[str] = set()
    for part in snapshot.parts:
        if part.key not in DIGEST_HYGIENE_CONTEXT_KEYS:
            continue
        keys.append(part.key)
        tokens.update(_tokenize_digest_text(_normalize_conditioning_text(part.content)))
    return list(dict.fromkeys(keys)), tokens


def _filter_conditioning_echo(
    content: str,
    conditioning_tokens: set[str],
) -> tuple[str, bool]:
    """Remove assistant sentences that mostly restate runtime conditioning context."""

    normalized = content.strip()
    if not normalized or not conditioning_tokens:
        return normalized, False

    segments = [
        seg.strip()
        for seg in _DIGEST_SEGMENT_SPLIT_RE.split(normalized)
        if seg.strip()
    ]
    if not segments:
        return normalized, False

    kept_segments: list[str] = []
    filtered = False
    for segment in segments:
        segment_tokens = _tokenize_digest_text(segment)
        overlap_count = len(segment_tokens & conditioning_tokens)
        overlap_ratio = overlap_count / max(len(segment_tokens), 1)
        if segment_tokens and len(segment_tokens) >= 2 and segment_tokens <= conditioning_tokens:
            filtered = True
            continue
        if len(segment_tokens) >= 4 and overlap_count >= 3 and overlap_ratio >= 0.5:
            filtered = True
            continue
        kept_segments.append(segment)

    if not kept_segments and filtered:
        return "", True
    if not filtered:
        return normalized, False
    return " ".join(kept_segments).strip(), True


def _build_digest_turn_payload(turn: ThreadMessage) -> dict[str, object] | None:
    """Render one digest-safe thread turn payload with conditioning hygiene metadata."""

    content = turn.content.strip()
    if not content:
        return None

    digest_hygiene: dict[str, object] = {}
    if turn.role == "assistant":
        conditioning_keys, conditioning_tokens = _conditioning_keys_and_tokens(turn)
        if conditioning_keys:
            content, filtered = _filter_conditioning_echo(content, conditioning_tokens)
            digest_hygiene = {
                "conditioning_context_keys": conditioning_keys,
                "conditioning_echo_filtered": filtered,
            }

    normalized = content.strip()
    if not normalized:
        return None

    payload: dict[str, object] = {
        "role": turn.role,
        "created_at": _turn_timestamp(turn),
        "content": normalized,
    }
    if digest_hygiene:
        payload["digest_hygiene"] = digest_hygiene
    return payload


def _format_turns_for_prompt(turns: list[ThreadMessage]) -> str:
    """Render scoped thread turns as bounded JSON for digest synthesis."""
    max_chars = max(1000, settings.conversation_digest_max_prompt_chars)
    source_turns = turns[-max(1, settings.conversation_digest_source_turns) :]
    rendered: list[dict[str, object]] = []
    used_chars = 0
    for turn in source_turns:
        payload = _build_digest_turn_payload(turn)
        if payload is None:
            continue
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        content = str(payload["content"])
        clipped = content[: min(len(content), remaining)]
        payload["content"] = clipped
        rendered.append(payload)
        used_chars += len(clipped)
    return json.dumps(rendered, indent=2)


def _build_digest_user_prompt(
    *,
    current: ConversationDigest,
    turns: list[ThreadMessage],
) -> str:
    """Render the authored digest prompt with previous digest and recent turns."""
    return _load_digest_user_prompt_template().format(
        max_summary_chars=max(400, settings.conversation_digest_max_summary_chars),
        recent_turn_count=max(1, settings.conversation_digest_recent_turns),
        previous_digest=json.dumps(current.model_dump(mode="json"), indent=2),
        scoped_turns=_format_turns_for_prompt(turns),
    )


def _bounded_items(values: list[str], *, limit: int = 6) -> list[str]:
    """Return non-empty, de-duplicated digest items within a small display cap."""
    selected: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(normalized[:240])
        if len(selected) >= limit:
            break
    return selected


def _fallback_digest(
    *,
    current: ConversationDigest,
    turns: list[ThreadMessage],
) -> ConversationDigest:
    """Return a deterministic digest when model synthesis is unavailable."""
    if current.summary.strip():
        return current.model_copy(
            update={
                "source_turn_count": len(turns),
                "last_message_at": _turn_timestamp(turns[-1]) if turns else "",
            }
        )

    recent_payloads = [
        payload for payload in (_build_digest_turn_payload(turn) for turn in turns[-2:]) if payload
    ]
    summary = " ".join(
        f"{str(payload['role'])}: {str(payload['content']).strip()[:180]}"
        for payload in recent_payloads
    )
    return ConversationDigest(
        updated_at=datetime.now(UTC).isoformat(),
        source_turn_count=len(turns),
        last_message_at=_turn_timestamp(turns[-1]) if turns else "",
        summary=summary[: settings.conversation_digest_max_summary_chars],
    )


async def update_conversation_digest(
    *,
    current: ConversationDigest,
    turns: list[ThreadMessage],
    complete_fn: Any = providers.complete_with_details,
) -> ConversationDigest:
    """Refresh one scoped rolling digest from the latest thread turns.

    Called after a user+assistant turn has been persisted. Provider failures are
    non-fatal because raw thread state and existing digest content remain valid.
    """
    if not settings.conversation_digest_enabled or not turns:
        return current
    if current.updated_at and current.source_turn_count == len(turns):
        return current
    if complete_fn is None:
        return _fallback_digest(current=current, turns=turns)

    model_def = model_policy.resolve_model_role(
        "distiller",
        preferred_alias=settings.conversation_digest_model,
    )
    try:
        raw = await complete_fn(
            messages=[
                {"role": "system", "content": _load_digest_system_prompt()},
                {
                    "role": "user",
                    "content": _build_digest_user_prompt(current=current, turns=turns),
                },
            ],
            model_def=model_def,
            max_tokens=1100 if model_def.reasoning_model else 900,
            temperature=0.1,
            response_format=_ConversationDigestSynthesis,
        )
        content = str(getattr(raw, "content", raw) or "")
        synthesis = _ConversationDigestSynthesis.model_validate_json(content)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("conversation digest refresh failed: %s", exc)
        return _fallback_digest(current=current, turns=turns)

    summary = synthesis.summary.strip()[: settings.conversation_digest_max_summary_chars]
    return ConversationDigest(
        updated_at=datetime.now(UTC).isoformat(),
        source_turn_count=len(turns),
        last_message_at=_turn_timestamp(turns[-1]) if turns else "",
        summary=summary,
        open_loops=_bounded_items(synthesis.open_loops),
        decisions=_bounded_items(synthesis.decisions),
        commitments=_bounded_items(synthesis.commitments),
        unresolved_references=_bounded_items(synthesis.unresolved_references),
    )


def render_conversation_digest(
    digest: ConversationDigest,
    *,
    max_chars: int | None = None,
) -> str:
    """Render a prompt-safe digest block for default-chat history injection."""
    if not digest.summary.strip() and not any(
        [digest.open_loops, digest.decisions, digest.commitments, digest.unresolved_references]
    ):
        return ""

    lines = ["## Conversation Digest"]
    if digest.summary.strip():
        lines.append(digest.summary.strip())

    sections = [
        ("Open loops", digest.open_loops),
        ("Decisions", digest.decisions),
        ("Commitments", digest.commitments),
        ("Unresolved references", digest.unresolved_references),
    ]
    for title, items in sections:
        selected = [item.strip() for item in items if item.strip()]
        if not selected:
            continue
        lines.append(f"### {title}")
        lines.extend(f"- {item}" for item in selected[:6])

    rendered = "\n".join(lines)
    limit = max_chars or settings.conversation_digest_max_summary_chars
    if len(rendered) > limit:
        rendered = rendered[:limit].rsplit("\n", 1)[0] + "\n- Conversation digest truncated."
    return rendered
