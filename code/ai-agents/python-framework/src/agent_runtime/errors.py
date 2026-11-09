from typing import Any


class ToolExecutionError(Exception):
    def __init__(self, message: str, result: str = "") -> None:
        super().__init__(message)
        self.result = result


class StreamTimeoutError(TimeoutError):
    def __init__(self) -> None:
        super().__init__("llm stream: watchdog timeout")


class AgentCancelledError(Exception):
    pass


class AgentLoopError(Exception):
    def __init__(self, message: str, messages: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.messages = messages
