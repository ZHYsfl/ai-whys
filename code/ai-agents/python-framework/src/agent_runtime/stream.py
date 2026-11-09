from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompletionUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ResponseFunction:
    name: str = ""
    arguments: str = ""


@dataclass
class ResponseToolCall:
    id: str = ""
    type: str = "function"
    function: ResponseFunction = field(default_factory=ResponseFunction)


@dataclass
class ResponseMessage:
    content: str = ""
    tool_calls: list[ResponseToolCall] = field(default_factory=list)


@dataclass
class CompletionChoice:
    message: ResponseMessage = field(default_factory=ResponseMessage)
    finish_reason: str = ""


@dataclass
class Completion:
    choices: list[CompletionChoice]
    usage: CompletionUsage


@dataclass
class _ChoiceState:
    content: list[str] = field(default_factory=list)
    tool_calls: dict[int, ResponseToolCall] = field(default_factory=dict)
    finish_reason: str = ""


class CompletionAccumulator:
    def __init__(self) -> None:
        self._choices: dict[int, _ChoiceState] = {}
        self._usage = CompletionUsage()

    def add_chunk(self, chunk: Any) -> None:
        usage = _value(chunk, "usage")
        if usage is not None:
            self._usage = CompletionUsage(
                prompt_tokens=_value(usage, "prompt_tokens", 0) or 0,
                completion_tokens=_value(usage, "completion_tokens", 0) or 0,
            )

        for choice in _value(chunk, "choices", []) or []:
            index = _value(choice, "index", 0)
            state = self._choices.setdefault(index, _ChoiceState())
            delta = _value(choice, "delta")
            content = _value(delta, "content")
            if content:
                state.content.append(content)

            for call in _value(delta, "tool_calls", []) or []:
                call_index = _value(call, "index", 0)
                accumulated = state.tool_calls.setdefault(
                    call_index,
                    ResponseToolCall(),
                )
                call_id = _value(call, "id")
                if call_id:
                    accumulated.id = call_id
                call_type = _value(call, "type")
                if call_type:
                    accumulated.type = call_type

                function = _value(call, "function")
                name = _value(function, "name")
                if name:
                    accumulated.function.name += name
                arguments = _value(function, "arguments")
                if arguments:
                    accumulated.function.arguments += arguments

            finish_reason = _value(choice, "finish_reason")
            if finish_reason:
                state.finish_reason = finish_reason

    def completion(self) -> Completion:
        choices = []
        for index in sorted(self._choices):
            state = self._choices[index]
            choices.append(
                CompletionChoice(
                    message=ResponseMessage(
                        content="".join(state.content),
                        tool_calls=[
                            state.tool_calls[call_index]
                            for call_index in sorted(state.tool_calls)
                        ],
                    ),
                    finish_reason=state.finish_reason,
                )
            )
        return Completion(choices=choices, usage=self._usage)


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def assistant_msg_to_param(message: Any) -> dict[str, Any]:
    tool_calls = []
    for call in _value(message, "tool_calls", []) or []:
        function = _value(call, "function")
        name = _value(function, "name", "") or ""
        if not name:
            continue
        tool_calls.append(
            {
                "id": _value(call, "id", "") or "",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _value(function, "arguments", "") or "",
                },
            }
        )

    output: dict[str, Any] = {"role": "assistant"}
    content = _value(message, "content", "") or ""
    if content:
        output["content"] = content
    if tool_calls:
        output["tool_calls"] = tool_calls
    return output
