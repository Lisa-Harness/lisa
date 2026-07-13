"""``gmail_draft`` operation — draft an email and create a Gmail draft.

The operation asks the model to produce structured draft fields (to, subject,
body) from the user's natural-language request, then calls the
``gmail_create_draft`` tool to persist the draft via the Gmail REST API.
"""

from __future__ import annotations

import json

from kernel.models import (
    OperationExecutionContext,
    OperationResult,
    ToolExecutionContext,
)

__all__ = ["execute"]


_SYSTEM_MESSAGE = (
    "You draft concise, professional emails. Return ONLY a JSON object with "
    "keys: to, subject, body. Infer the recipient from context if not "
    "explicit. Keep the body under 300 words."
)


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Draft an email from the user's request and create it as a Gmail draft."""

    query = (ctx.params.get("query") or ctx.query or "").strip()
    if not query:
        return OperationResult()

    if ctx.complete_fn is None or ctx.execute_tool_fn is None:
        return OperationResult(
            output_text="⚠️ Gmail drafting needs model + tool callbacks wired.",
        )

    # Ask the model to produce structured draft fields.
    completion = await ctx.complete_fn(
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=600,
    )

    content = getattr(completion, "content", "") or ""
    try:
        draft = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return OperationResult(
            output_text=f"⚠️ Could not parse draft: {content[:200]}",
        )

    to = str(draft.get("to", "")).strip()
    subject = str(draft.get("subject", "")).strip()
    body = str(draft.get("body", "")).strip()
    if not to or not subject or not body:
        return OperationResult(
            output_text="⚠️ Model returned an incomplete draft (need to, subject, body).",
        )

    # Persist via the tool.
    tool_result = await ctx.execute_tool_fn(
        "gmail_create_draft",
        ToolExecutionContext(
            tool_id="gmail_create_draft",
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            query=query,
            parameters={
                "to": to,
                "subject": subject,
                "body": body,
            },
        ),
    )

    if tool_result.error:
        return OperationResult(output_text=f"⚠️ {tool_result.error}")

    summary = f"Drafted to {to} — subject: {subject}"
    return OperationResult(
        output_text=summary,
        output_json=tool_result.output_json or {},
    )