"""Safe child-workflow execution for workshop fan-out.

This module runs multiple child workflow turns with isolated session keys and
structured concurrency. It stays below the app layer so pack-owned workshop
operations can reuse it without importing host orchestration modules.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Protocol

from kernel.models import (
    SurfaceContext,
    WorkflowSessionState,
    WorkshopChildResult,
    WorkshopExecutionSummary,
    WorkshopPanelContext,
)


class TextResponse(Protocol):
    """Minimal response contract needed from a workflow-turn runner."""

    text: str


class WorkflowTurnRunner(Protocol):
    """Public callback contract for one workflow turn execution."""

    async def __call__(
        self,
        *,
        session: WorkflowSessionState,
        slug: str,
        clean_message: str,
        model_def: object,
        user_block: str,
        user_id: str,
        user_name: str,
        principal_key: str,
        thread_user_id: str,
        session_key: str,
        conversation_id: str,
        workspace_id: str,
        surface_context: SurfaceContext,
        event_bus: Any = None,
        persist_session_state: bool = True,
        run_post_turn_operations: bool = True,
        include_model_badge: bool = True,
        delegation_depth: int = 0,
    ) -> TextResponse:
        ...


@dataclass(slots=True)
class WorkshopExecutionParentContext:
    """Stable parent-turn inputs shared by all workshop child runs."""

    user_id: str
    user_slug: str
    user_name: str
    principal_key: str
    thread_user_id: str
    parent_session_key: str
    conversation_id: str
    workspace_id: str
    surface_context: SurfaceContext
    event_bus: Any = None


@dataclass(slots=True)
class WorkshopChildExecutionRequest:
    """One child workflow request derived from a panel context bundle."""

    panel_context: WorkshopPanelContext
    model_def: object
    user_block: str = ""
    clean_message: str = ""
    timeout_ms: int = 30_000
    persist_session_state: bool = False
    run_post_turn_operations: bool = False
    include_model_badge: bool = False
    delegation_depth: int = 1


class WorkshopExecutionService:
    """Run workshop children with isolated session keys and bounded cleanup."""

    def __init__(
        self,
        *,
        run_workflow_turn: WorkflowTurnRunner,
        create_session_fn: Callable[[str, str, str], WorkflowSessionState] | None = None,
        perf_clock: Callable[[], float] = perf_counter,
        max_parallel_children: int | None = None,
    ) -> None:
        self._run_workflow_turn = run_workflow_turn
        self._create_session = create_session_fn or _default_create_session
        self._perf_clock = perf_clock
        self._max_parallel_children = max_parallel_children

    @staticmethod
    def build_child_session_key(parent_session_key: str, panel_id: str) -> str:
        """Return one deterministic child-scoped session key."""

        return f"{parent_session_key}::workshop::{panel_id}"

    async def execute_children(
        self,
        *,
        workshop_id: str,
        parent_context: WorkshopExecutionParentContext,
        child_requests: list[WorkshopChildExecutionRequest],
    ) -> WorkshopExecutionSummary:
        """Execute workshop child workflows concurrently."""

        panel_ids = [request.panel_context.panel_id for request in child_requests]
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError("Workshop child requests must use unique panel_id values.")

        max_parallel_children = self._resolve_max_parallel_children(child_requests)
        results: list[WorkshopChildResult] = []
        for batch_start in range(0, len(child_requests), max_parallel_children):
            batch_requests = child_requests[batch_start : batch_start + max_parallel_children]
            results.extend(
                await self._execute_batch(
                    parent_context=parent_context,
                    child_requests=batch_requests,
                )
            )

        return WorkshopExecutionSummary(
            workshop_id=workshop_id,
            total_children=len(results),
            succeeded_children=sum(1 for result in results if result.status == "ok"),
            timed_out_children=sum(1 for result in results if result.status == "timeout"),
            cancelled_children=sum(1 for result in results if result.status == "cancelled"),
            failed_children=sum(1 for result in results if result.status == "error"),
            results=results,
        )

    async def _execute_batch(
        self,
        *,
        parent_context: WorkshopExecutionParentContext,
        child_requests: list[WorkshopChildExecutionRequest],
    ) -> list[WorkshopChildResult]:
        tasks: dict[str, asyncio.Task[WorkshopChildResult]] = {}
        async with asyncio.TaskGroup() as task_group:
            for request in child_requests:
                panel_id = request.panel_context.panel_id
                tasks[panel_id] = task_group.create_task(
                    self._execute_child(parent_context=parent_context, request=request),
                    name=f"workshop-child-{panel_id}",
                )

        return [tasks[request.panel_context.panel_id].result() for request in child_requests]

    def _resolve_max_parallel_children(
        self,
        child_requests: list[WorkshopChildExecutionRequest],
    ) -> int:
        if not child_requests:
            return 1
        if self._max_parallel_children is None:
            return len(child_requests)
        return max(1, min(self._max_parallel_children, len(child_requests)))

    async def _execute_child(
        self,
        *,
        parent_context: WorkshopExecutionParentContext,
        request: WorkshopChildExecutionRequest,
    ) -> WorkshopChildResult:
        """Execute one child workflow request with a bounded timeout."""

        panel_context = request.panel_context
        session_key = self.build_child_session_key(
            parent_context.parent_session_key,
            panel_context.panel_id,
        )
        session = self._create_session(
            parent_context.user_id,
            parent_context.user_slug,
            panel_context.mode_id,
        )
        started_at = self._now_iso()
        started_perf = self._perf_clock()
        clean_message = request.clean_message or panel_context.query

        try:
            async with asyncio.timeout(max(request.timeout_ms, 1) / 1000):
                response = await self._run_workflow_turn(
                    session=session,
                    slug=parent_context.user_slug,
                    clean_message=clean_message,
                    model_def=request.model_def,
                    user_block=request.user_block,
                    user_id=parent_context.user_id,
                    user_name=parent_context.user_name,
                    principal_key=parent_context.principal_key,
                    thread_user_id=parent_context.thread_user_id,
                    session_key=session_key,
                    conversation_id=parent_context.conversation_id,
                    workspace_id=parent_context.workspace_id,
                    surface_context=parent_context.surface_context,
                    event_bus=parent_context.event_bus,
                    persist_session_state=request.persist_session_state,
                    run_post_turn_operations=request.run_post_turn_operations,
                    include_model_badge=request.include_model_badge,
                    delegation_depth=request.delegation_depth,
                )
        except TimeoutError:
            return WorkshopChildResult(
                panel_id=panel_context.panel_id,
                mode_id=panel_context.mode_id,
                child_session_key=session_key,
                status="timeout",
                error_message=f"Child workflow timed out after {request.timeout_ms}ms.",
                started_at=started_at,
                finished_at=self._now_iso(),
                duration_ms=self._elapsed_ms(started_perf),
            )
        except Exception as exc:  # pylint: disable=broad-except
            return WorkshopChildResult(
                panel_id=panel_context.panel_id,
                mode_id=panel_context.mode_id,
                child_session_key=session_key,
                status="error",
                error_message=f"{type(exc).__name__}: {exc}",
                started_at=started_at,
                finished_at=self._now_iso(),
                duration_ms=self._elapsed_ms(started_perf),
            )

        return WorkshopChildResult(
            panel_id=panel_context.panel_id,
            mode_id=panel_context.mode_id,
            child_session_key=session_key,
            status="ok",
            output_text=response.text,
            turn_count=session.turn_count,
            started_at=started_at,
            finished_at=self._now_iso(),
            duration_ms=self._elapsed_ms(started_perf),
        )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def _elapsed_ms(self, started_perf: float) -> int:
        return int((self._perf_clock() - started_perf) * 1000)


def _default_create_session(user_id: str, user_slug: str, mode_id: str) -> WorkflowSessionState:
    """Create one workflow session via the live IPSAR catalog only when needed."""

    from kernel.ipsar_fsm import create_workflow_session  # noqa: PLC0415

    return create_workflow_session(user_id, user_slug, mode_id)
