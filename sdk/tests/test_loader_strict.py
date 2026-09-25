"""Tests for the strict authored-text loader helper.

Verifies the three failure modes raise :class:`MissingAuthoredContentError`
and that valid content passes through stripped.
"""

from __future__ import annotations

import pytest
from sdk.content import (
    MissingAuthoredContentError,
    require_authored_text,
    validate_required_content,
)


class TestRequireAuthoredText:
    def test_valid_content_passes_through_stripped(self) -> None:
        def loader(path: str) -> str:
            return "  authored body\n\n"

        result = require_authored_text(loader, "prompts/system-prompt.md")
        assert result == "authored body"

    def test_none_loader_raises(self) -> None:
        with pytest.raises(MissingAuthoredContentError, match="not wired") as exc_info:
            require_authored_text(None, "prompts/system-prompt.md")
        assert exc_info.value.path == "prompts/system-prompt.md"

    def test_non_callable_loader_raises(self) -> None:
        with pytest.raises(MissingAuthoredContentError, match="not wired"):
            require_authored_text("not-a-callable", "prompts/system-prompt.md")  # type: ignore[arg-type]

    def test_empty_string_result_raises(self) -> None:
        def empty_loader(path: str) -> str:
            return ""

        with pytest.raises(MissingAuthoredContentError, match="empty or missing") as exc_info:
            require_authored_text(empty_loader, "prompts/system-prompt.md")
        assert exc_info.value.path == "prompts/system-prompt.md"

    def test_whitespace_only_result_raises(self) -> None:
        def whitespace_loader(path: str) -> str:
            return "   \n\t  "

        with pytest.raises(MissingAuthoredContentError, match="empty or missing"):
            require_authored_text(whitespace_loader, "prompts/system-prompt.md")

    def test_none_result_raises(self) -> None:
        def none_loader(path: str) -> str:
            return None  # type: ignore[return-value]

        with pytest.raises(MissingAuthoredContentError, match="empty or missing"):
            require_authored_text(none_loader, "prompts/system-prompt.md")

    def test_loader_exception_raises(self) -> None:
        def raising_loader(path: str) -> str:
            raise OSError("disk error")

        with pytest.raises(MissingAuthoredContentError, match="failed for") as exc_info:
            require_authored_text(raising_loader, "prompts/system-prompt.md")
        assert "disk error" in str(exc_info.value)
        assert exc_info.value.path == "prompts/system-prompt.md"

    def test_context_suffix_appended_to_message(self) -> None:
        with pytest.raises(MissingAuthoredContentError, match="synthesize operation"):
            require_authored_text(None, "prompts/x.md", context="synthesize operation")

    def test_missing_authored_content_error_is_runtime_error(self) -> None:
        """Consumers catching RuntimeError still work (back-compat)."""
        assert issubclass(MissingAuthoredContentError, RuntimeError)


class TestValidateRequiredContent:
    def test_all_paths_present_passes(self) -> None:
        def loader(path: str) -> str:
            return f"content for {path}"

        validate_required_content(loader, ["a.md", "b.md"])

    def test_first_missing_path_raises(self) -> None:
        def loader(path: str) -> str:
            if path == "b.md":
                return ""
            return "content"

        with pytest.raises(MissingAuthoredContentError, match=r"b\.md"):
            validate_required_content(loader, ["a.md", "b.md", "c.md"])

    def test_empty_paths_iterable_passes(self) -> None:
        def loader(path: str) -> str:
            return "content"

        validate_required_content(loader, [])
