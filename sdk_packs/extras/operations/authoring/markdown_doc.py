"""Generate markdown documents on demand."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from kernel.models import OperationExecutionContext, OperationResult


class Parameters(BaseModel):
    """Optional structured parameters for markdown generation."""

    request: str = Field(
        default="",
        validation_alias=AliasChoices("request", "query", "prompt", "topic"),
    )


class MarkdownDocument(BaseModel):
    """Structured output contract used to render one markdown document."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra=lambda schema: _compact_json_schema(schema),
    )

    title: str
    body_markdown: str = Field(validation_alias=AliasChoices("body_markdown", "body", "markdown"))


def _require_authored_text_reader(ctx: OperationExecutionContext) -> Any:
    """Return the authored-text loader injected by the host runtime."""

    if callable(ctx.read_authored_text_fn):
        return ctx.read_authored_text_fn
    raise RuntimeError("Authored content reader not injected into operation context.")


def _load_system_prompt(ctx: OperationExecutionContext) -> str:
    """Load the authored system prompt for markdown document generation."""

    return str(_require_authored_text_reader(ctx)("prompts/markdown_doc/system.md")).strip()


def _load_user_prompt_template(ctx: OperationExecutionContext) -> str:
    """Load the authored user-prompt template for markdown document generation."""

    return str(_require_authored_text_reader(ctx)("prompts/markdown_doc/user.md")).strip()


def _build_user_prompt(ctx: OperationExecutionContext, request_text: str) -> str:
    """Render the authored user prompt for one markdown generation request."""

    return _load_user_prompt_template(ctx).format(request_text=request_text)


def _compact_json_schema(schema: dict[str, Any]) -> None:
    """Drop non-validation metadata to keep structured-output schemas compact."""
    schema.pop("description", None)
    schema.pop("title", None)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for property_schema in properties.values():
        if isinstance(property_schema, dict):
            property_schema.pop("description", None)
            property_schema.pop("title", None)


def _normalize_markdown(text: str, title: str) -> str:
    """Return clean markdown and strip accidental fenced-wrapper output."""
    normalized = text.strip()
    if normalized.startswith("```markdown"):
        normalized = normalized[len("```markdown") :].lstrip()
    elif normalized.startswith("```"):
        normalized = normalized[len("```") :].lstrip()

    if normalized.endswith("```"):
        normalized = normalized[: -len("```")].rstrip()

    first_line = normalized.splitlines()[0].strip() if normalized else ""
    if not first_line.startswith("# "):
        heading = title.strip() or "Generated Document"
        normalized = f"# {heading}\n\n{normalized}" if normalized else f"# {heading}"

    return normalized


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Generate one markdown document for a user request."""
    if not ctx.complete_fn:
        return OperationResult(
            output_text="Missing model completion function in operation context."
        )

    parameters = Parameters.model_validate(ctx.params or {})
    request_text = (parameters.request or "").strip() or (ctx.query or "").strip()
    if not request_text:
        return OperationResult(
            output_text=(
                "Please provide what you want the markdown document to cover. "
                "Example: /markdown Draft an incident postmortem for today's Cloudflare 5xx spike."
            )
        )

    if not callable(ctx.resolve_model_fn):
        return OperationResult(
            output_text="Markdown generation unavailable: model resolver not injected."
        )

    model_def = ctx.resolve_model_fn(ctx.preferred_model_id or ctx.model_id or "gpt-4o")
    if model_def is None:
        model_def = ctx.resolve_model_fn("gpt-4o")
    if model_def is None:
        return OperationResult(
            output_text="Markdown generation failed: no compatible model is configured."
        )

    try:
        system_prompt = _load_system_prompt(ctx)
        user_prompt = _build_user_prompt(ctx, request_text)
    except Exception as exc:  # pylint: disable=broad-except
        return OperationResult(
            output_text=f"Markdown generation unavailable: {type(exc).__name__}: {exc}"
        )

    try:
        completion = await ctx.complete_fn(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model_def=model_def,
            max_tokens=4096 if model_def.reasoning_model else 1800,
            temperature=0.2,
            response_format=MarkdownDocument,
        )
        doc = MarkdownDocument.model_validate_json(completion.content)
    except Exception as exc:  # pylint: disable=broad-except
        return OperationResult(
            output_text=f"Markdown generation failed: {type(exc).__name__}: {exc}"
        )

    markdown_text = _normalize_markdown(doc.body_markdown, doc.title)
    slug = re.sub(r"[^\w-]", "-", doc.title.lower().strip())[:60].strip("-") or "document"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    blob_name = f"{timestamp}-{slug}.md"

    upload_fn = getattr(ctx.blob_io, "upload_markdown_doc", None)
    if not callable(upload_fn):
        return OperationResult(
            output_text="Markdown generation unavailable: blob writer not injected."
        )

    download_url = upload_fn(markdown_text, blob_name=blob_name)
    return OperationResult(
        output_text=markdown_text,
        output_json={
            "title": doc.title,
            "generated_at": datetime.now(UTC).isoformat(),
            "request": request_text,
            "download_url": download_url,
            "blob_name": blob_name,
        },
    )
