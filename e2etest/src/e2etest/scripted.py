"""Scripted model: drive a real runtime with a deterministic, token-free "model".

A scenario that declares ``model: {script: <file>.py}`` replaces every provider
gateway with :class:`ScriptedGatewayProvider`. The script module defines
``respond(turn: ScriptTurn) -> Reply``; it sees the real request the kernel
built (messages, tool list, prior tool results) and answers with text or tool
calls, so values such as artifact ids and file digests flow from real tool
output exactly as they would for a live model. Everything below the model —
routing, policy, approvals, packs, sandbox, persistence — is production code.

Requests that carry no tools (classifiers, distillers, titles) are auxiliary;
the script may answer them with ``auxiliary(turn)``, otherwise they receive an
empty JSON object or empty text.
"""

from __future__ import annotations

import importlib.util
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from kernel.provider_contracts import CompletionRequest, CompletionResult, CompletionUsage


class ScriptError(RuntimeError):
    """Raised when a model script is malformed or runs out of replies."""


@dataclass(frozen=True)
class ToolOutcome:
    """One tool call the model made earlier in the transcript, with its result."""

    name: str
    arguments: dict[str, Any]
    text: str

    @property
    def data(self) -> dict[str, Any]:
        """The JSON object embedded in a tool result (``"<summary> {json}"``), or ``{}``."""
        start = self.text.find("{")
        if start < 0:
            return {}
        try:
            parsed = json.loads(self.text[start:])
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @property
    def failed(self) -> bool:
        return self.text.startswith(("Error:", "Tool execution failed"))


@dataclass(frozen=True)
class Reply:
    """What the scripted model answers: final text and/or tool calls."""

    content: str = ""
    tool_calls: tuple[tuple[str, dict[str, Any]], ...] = ()

    @classmethod
    def text(cls, content: str) -> Reply:
        return cls(content=content)

    @classmethod
    def call(cls, tool: str, /, **arguments: Any) -> Reply:
        return cls(tool_calls=((tool, arguments),))

    @classmethod
    def calls(cls, *calls: tuple[str, dict[str, Any]]) -> Reply:
        return cls(tool_calls=tuple(calls))


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else str(content)


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass
class ScriptTurn:
    """The request the kernel sent, with transcript helpers for scripts."""

    request: CompletionRequest
    index: int

    @property
    def tool_names(self) -> set[str]:
        names: set[str] = set()
        for tool in self.request.tools or ():
            function = tool.get("function", tool)
            if isinstance(function, dict) and function.get("name"):
                names.add(str(function["name"]))
        return names

    @property
    def auxiliary(self) -> bool:
        return not self.request.tools

    @property
    def user_text(self) -> str:
        """The most recent user message."""
        for message in reversed(self.request.messages):
            if message.get("role") == "user":
                return _text(message.get("content"))
        return ""

    def outcomes(self) -> list[ToolOutcome]:
        """Every tool call in the transcript, in order, paired with its result text."""
        results = {
            str(m.get("tool_call_id")): _text(m.get("content"))
            for m in self.request.messages
            if m.get("role") == "tool"
        }
        outcomes: list[ToolOutcome] = []
        for message in self.request.messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or ():
                function = call.get("function", call)
                outcomes.append(
                    ToolOutcome(
                        name=str(function.get("name", "")),
                        arguments=_arguments(function.get("arguments")),
                        text=results.get(str(call.get("id")), ""),
                    )
                )
        return outcomes

    def last(self, name: str) -> ToolOutcome | None:
        """The most recent call of ``name``, if any."""
        for outcome in reversed(self.outcomes()):
            if outcome.name == name:
                return outcome
        return None

    def count(self, name: str) -> int:
        return sum(1 for outcome in self.outcomes() if outcome.name == name)


@dataclass
class ScriptedGateway:
    """A ``ProviderGateway`` that answers from the scenario's model script."""

    owner: ScriptedGatewayProvider

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        return self.owner.answer(request)


@dataclass
class ScriptedGatewayProvider:
    """``ProviderGatewayProvider`` whose every provider resolves to the script."""

    respond: Callable[[ScriptTurn], Reply]
    auxiliary: Callable[[ScriptTurn], Reply] | None = None
    log: list[dict[str, Any]] = field(default_factory=list)
    _calls: Iterator[int] = field(default_factory=lambda: iter(range(1, 1_000_000)))

    def gateway_for(self, provider: str) -> ScriptedGateway:
        _ = provider
        return ScriptedGateway(self)

    def answer(self, request: CompletionRequest) -> CompletionResult:
        turn = ScriptTurn(request=request, index=len(self.log))
        if turn.auxiliary:
            reply = (
                self.auxiliary(turn)
                if self.auxiliary is not None
                else Reply.text("{}" if request.response_format is not None else "")
            )
        else:
            reply = self.respond(turn)
        if not isinstance(reply, Reply):
            raise ScriptError(f"model script returned {type(reply).__name__}, expected Reply")
        tool_calls = [
            {
                "id": f"call_{next(self._calls)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for name, arguments in reply.tool_calls
        ]
        self.log.append(
            {
                "t": round(time.monotonic(), 3),
                "auxiliary": turn.auxiliary,
                "content": reply.content,
                "tool_calls": [[name, arguments] for name, arguments in reply.tool_calls],
            }
        )
        model = request.model_def
        return CompletionResult(
            content=reply.content,
            provider=str(getattr(model, "provider", "") or ""),
            model_id=str(getattr(model, "model_id", "") or ""),
            display_name=str(getattr(model, "display_name", "") or ""),
            tool_calls=tool_calls,
            usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason="tool_calls" if tool_calls else "stop",
        )


def load_script(path: Path) -> ScriptedGatewayProvider:
    """Import a model script and wrap its ``respond``/``auxiliary`` hooks."""
    spec = importlib.util.spec_from_file_location(f"e2e_model_script_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ScriptError(f"cannot import model script: {path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    respond = getattr(module, "respond", None)
    if not callable(respond):
        raise ScriptError(f"model script defines no respond(turn): {path}")
    auxiliary = getattr(module, "auxiliary", None)
    return ScriptedGatewayProvider(
        respond=respond, auxiliary=auxiliary if callable(auxiliary) else None
    )
