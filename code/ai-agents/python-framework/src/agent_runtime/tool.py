from dataclasses import dataclass
from typing import Any, Callable


ToolFunc = Callable[[dict[str, Any]], str]


@dataclass
class Tool:
    name: str
    description: str
    func: ToolFunc
    parameters: dict[str, Any]


@dataclass
class ToolCallInput:
    id: str
    name: str
    arguments_json: str


@dataclass
class ToolResponse:
    message: dict[str, Any]
    content: str
    status: str
    error_type: str | None = None
