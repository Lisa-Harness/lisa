"""File-backed memory for local operation testing. NOT for production use.

`LiteMemoryStore` persists canonical memory records to disk so local hosts can
exercise operation contracts (`record_memory`, `list_memories`, etc.) without
bootstrapping the full app memory subsystem.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kernel.models import MemoryRecord, MemoryWriteRequest

logger = logging.getLogger(__name__)


class LiteMemoryStore:
    """File-backed memory store implementing the operation memory contract."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[MemoryRecord] = []
        self._next_id = 1
        self._load()

    # ── Public API ─────────────────────────────────────────────────────

    async def record_memory(self, request: MemoryWriteRequest | Any) -> MemoryRecord:
        """Persist one canonical memory record and return the created row."""

        if isinstance(request, MemoryWriteRequest):
            write = request
        else:
            write = MemoryWriteRequest.model_validate(request)

        created = datetime.now(UTC).isoformat()
        record = MemoryRecord(
            id=self._next_id,
            workspace_id=write.workspace_id,
            user_id=write.user_id,
            user_slug=write.user_slug,
            memory_scope=write.memory_scope,
            surface_id=write.surface_id,
            source_user_id=write.source_user_id,
            source_user_slug=write.source_user_slug,
            agent_id=write.agent_id,
            operation_id=write.operation_id,
            instruction_id=write.instruction_id,
            memory_tier=write.memory_tier,
            content=write.content,
            content_json=write.content_json,
            salience=write.salience,
            memory_kind=write.memory_kind,
            confidence=write.confidence,
            evidence_ids=write.evidence_ids,
            supersedes_id=write.supersedes_id,
            valid_until=write.valid_until,
            created_at=created,
        )
        self._records.append(record)
        self._next_id += 1
        self._save()
        return record

    async def list_memories(
        self,
        *,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """Return recent memories for one user and optional scope filters."""

        filtered = [
            record
            for record in self._records
            if record.user_slug == user_slug
            and record.workspace_id == workspace_id
            and record.memory_scope == memory_scope
            and (not surface_id or record.surface_id == surface_id)
        ]
        filtered.sort(key=lambda item: item.id, reverse=True)
        return filtered[: max(1, int(limit or 1))]

    async def list_recent_memories(
        self,
        *,
        user_slug: str,
        operation_id: str = "",
        limit: int = 20,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
    ) -> list[MemoryRecord]:
        """Return recent memories, optionally narrowed by operation id."""

        records = await self.list_memories(
            user_slug=user_slug,
            workspace_id=workspace_id,
            memory_scope=memory_scope,
            surface_id=surface_id,
            limit=limit,
        )
        if not operation_id:
            return records
        return [record for record in records if record.operation_id == operation_id]

    async def recall(self, query: str, **kwargs: Any) -> str:
        """Return a compact lexical recall snippet from persisted records."""

        user_slug = str(kwargs.get("user_slug") or "").strip() or "default"
        workspace_id = str(kwargs.get("workspace_id") or "default").strip() or "default"
        memory_scope = str(kwargs.get("memory_scope") or "personal").strip() or "personal"
        surface_id = str(kwargs.get("surface_id") or "").strip()
        limit = int(kwargs.get("limit", 5) or 5)
        keywords = [word.lower() for word in query.split() if len(word) > 2]
        if not keywords:
            return ""

        records = await self.list_memories(
            user_slug=user_slug,
            workspace_id=workspace_id,
            memory_scope=memory_scope,
            surface_id=surface_id,
            limit=max(5, limit * 2),
        )
        matched: list[MemoryRecord] = []
        for record in records:
            haystack = f"{record.content} {json.dumps(record.content_json, ensure_ascii=True)}".lower()
            if any(keyword in haystack for keyword in keywords):
                matched.append(record)
        if not matched:
            return ""
        lines = [f"- {record.content}" for record in matched[: max(1, limit)] if record.content]
        return "\n".join(lines)

    async def store(
        self,
        query: str,
        response: str,
    ) -> None:
        """Compatibility helper used only by lightweight tests/examples."""

        await self.record_memory(
            MemoryWriteRequest(
                user_slug="default",
                memory_scope="personal",
                operation_id="lite_store",
                instruction_id="lite_store.v1",
                memory_tier="daily",
                content=_synthesize_content(query, response),
                content_json={"query": query},
                salience=0.2,
            )
        )

    # ── Internal ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            self._records = []
            self._next_id = 1
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                self._records = []
                self._next_id = 1
                return
            raw_records = data.get("records", [])
            if not isinstance(raw_records, list):
                self._records = []
                self._next_id = 1
                return
            self._records = [MemoryRecord.model_validate(item) for item in raw_records]
            max_id = max((record.id for record in self._records), default=0)
            self._next_id = max_id + 1
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load lite memory: %s", exc)
            self._records = []
            self._next_id = 1

    def _save(self) -> None:
        try:
            payload = {
                "schema_version": 2,
                "records": [record.model_dump(mode="json") for record in self._records],
            }
            self._path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save lite memory: %s", exc)


def _make_entry_id() -> str:
    import uuid

    return str(uuid.uuid4())[:8]


def _synthesize_content(query: str, response: str) -> str:
    """Produce a compact memory entry from a turn's query and response."""
    clean_query = query.strip()
    clean_response = response.strip()
    if len(clean_response) <= 200:
        return clean_response
    # Truncate long responses
    snippet = clean_response[:200].rsplit(" ", 1)[0]
    return f"{clean_query}: {snippet}..."