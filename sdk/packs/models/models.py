"""Pydantic output contracts for the model advisor pack."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["AdviseModelsResult", "ManifestChange"]


class ManifestChange(BaseModel):
    """One atomic change the advisor applied to the manifest."""

    model_id: str
    field: str
    action: str  # "added", "modified", "removed"
    old_value: str = ""
    new_value: str = ""


class AdviseModelsResult(BaseModel):
    """Structured result of one ``/models update`` invocation."""

    target_path: str
    changes: list[ManifestChange] = Field(default_factory=list)
    summary: str = ""
    validated: bool = True
