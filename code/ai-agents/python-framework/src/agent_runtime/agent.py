import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator

from openai import APITimeoutError

from .client import new_openai_client
from .errors import (
    AgentCancelledError,
    AgentLoopError,
    StreamTimeoutError,
    ToolExecutionError,
)
from .llm_config import LLMConfig
from .memory import MemoryConfig, MemoryManager, new_memory
from .stream import (
    Completion,
    CompletionAccumulator,
    ResponseToolCall,
    _value,
    assistant_msg_to_param,
)
from .tool import Tool, ToolCallInput, ToolResponse


logger = logging.getLogger(__name__)
DEFAULT_STREAM_TIMEOUT = 15.0
BeforeRoundHook = Callable[[], list[dict[str, Any]]]
ProducedCallback = Callable[[dict[str, Any]], None]


class Agent:
    def __init__(
        self,
        config: LLMConfig,
        tools: list[Tool] | None = None,
        *,
        memory: MemoryManager | None = None,
        memory_config: MemoryConfig | None = None,
        debug: bool = False,
        max_tool_retries: int = 3,
        before_round_hook: BeforeRoundHook | None = None,
        produced_callback: ProducedCallback | None = None,
        client: Any | None = None,
    ) -> None:
        if memory is not None and memory_config is not None:
            raise ValueError("pass either memory or memory_config, not both")
        self.config = config
        self.tools = list(tools or [])
        self.debug = debug
        self.max_tool_retries = max_tool_retries
        self.before_round_hook = before_round_hook
        self.produced_callback = produced_callback
        self.memory = (
            memory
            if memory is not None
            else new_memory(memory_config or MemoryConfig())
        )
        self.client = client if client is not None else new_openai_client(config)

    def add_tool(self, tool: Tool) -> None:
        self.tools.append(tool)

    def remove_tool(self, name: str) -> None:
        self.tools = [tool for tool in self.tools if tool.name != name]

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.memory.prepare(messages)

    def _record_usage(self, prompt_tokens: int) -> None:
        recorder = getattr(self.memory, "record_usage", None)
        if recorder is not None:
            recorder(prompt_tokens)

    def build_chat_completion_params(
        self,
        messages: list[dict[str, Any]],
        with_tool: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": self.prepare(messages),
        }
        if with_tool:
            params["tools"] = self.get_tools()
            params["tool_choice"] = "auto"
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.extra_body is not None:
            params["extra_body"] = dict(self.config.extra_body)
        return params

    def chat_without_tool_call(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        params = self.build_chat_completion_params(messages, with_tool=False)
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as exc:
            raise RuntimeError(f"chat text: {exc}") from exc
        if not response.choices:
            raise RuntimeError("chat text: no choices returned")

        message = response.choices[0].message
        next_messages = [*messages, assistant_msg_to_param(message)]
        return next_messages, _value(message, "content", "") or ""

    def chat_without_tool_call_stream(
        self,
        messages: list[dict[str, Any]],
    ) -> Iterator[str]:
        params = self.build_chat_completion_params(messages, with_tool=False)
        stream = None
        try:
            stream = self.client.chat.completions.create(**params, stream=True)
            for chunk in stream:
                for choice in _value(chunk, "choices", []) or []:
                    delta = _value(choice, "delta")
                    content = _value(delta, "content", "") or ""
                    if content:
                        yield content
        except Exception as exc:
            logger.error("stream chat error: %s", exc)
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()

    def execute_tool_calls(
        self,
        calls: list[ToolCallInput],
    ) -> list[dict[str, Any]]:
        available_tools = {tool.name: tool for tool in self.tools}
        if not calls:
            return []
        with ThreadPoolExecutor(max_workers=len(calls)) as executor:
            futures = [
                executor.submit(
                    self._execute_tool_call_func,
                    call.id,
                    call.name,
                    call.arguments_json,
                    available_tools,
                )
                for call in calls
            ]
            results = [future.result() for future in futures]
        return [result.message for result in results]

    def _execute_tool_call(
        self,
        tool_call: ResponseToolCall,
        available_tools: dict[str, Tool],
    ) -> ToolResponse:
        try:
            call_type = _value(tool_call, "type", "")
            call_id = _value(tool_call, "id", "") or ""
            function = _value(tool_call, "function")
            name = _value(function, "name", "") or ""
            if call_type != "function" or not call_id:
                content = (
                    "[NOT_FOUND] Unsupported tool call type "
                    "(only function tool calls are supported)"
                )
                return self._error_response(
                    "unsupported-tool-call",
                    "not_found",
                    content,
                )
            return self._execute_tool_call_func(
                call_id,
                name,
                _value(function, "arguments", "") or "",
                available_tools,
            )
        except Exception as exc:
            return self._panic_response(
                _value(tool_call, "id", "") or "",
                _value(_value(tool_call, "function"), "name", "") or "",
                exc,
            )

    def _execute_tool_call_func(
        self,
        tool_call_id: str,
        function_name: str,
        raw_args: str,
        available_tools: dict[str, Tool],
    ) -> ToolResponse:
        try:
            if raw_args and raw_args.strip() and raw_args != "{}":
                try:
                    args = json.loads(raw_args)
                    if args is None:
                        args = {}
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    content = (
                        f"[PARSE_ERROR] JSON parse failed: {exc}. "
                        f"Raw: '{raw_args}'"
                    )
                    return self._error_response(
                        tool_call_id,
                        "parse_error",
                        content,
                    )
            else:
                args = {}

            tool = available_tools.get(function_name)
            if tool is None:
                return self._error_response(
                    tool_call_id,
                    "not_found",
                    f"[NOT_FOUND] Function '{function_name}' not found",
                )

            missing = validate_required_args(tool.parameters, args)
            if missing:
                missing_text = "[" + " ".join(missing) + "]"
                return self._error_response(
                    tool_call_id,
                    "arg_error",
                    f"[ARG_ERROR] Missing required arguments for "
                    f"'{function_name}': {missing_text}",
                )

            if self.debug:
                print(
                    f"[DEBUG] Executing {function_name} with args: "
                    f"{list(args)}"
                )
            try:
                result = tool.func(args)
                if not isinstance(result, str):
                    raise TypeError("tool function must return str")
            except ToolExecutionError as exc:
                return self._error_response(
                    tool_call_id,
                    "exec_error",
                    f"[EXEC_ERROR] Execution failed: {exc}",
                )
            except Exception as exc:
                return self._panic_response(tool_call_id, function_name, exc)

            if self.debug:
                print(
                    f"[DEBUG] Tool {function_name} returned "
                    f"(id={tool_call_id}): {result}"
                )
            return ToolResponse(
                message={
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                },
                content=result,
                status="success",
            )
        except Exception as exc:
            return self._panic_response(tool_call_id, function_name, exc)

    def _error_response(
        self,
        tool_call_id: str,
        error_type: str,
        content: str,
    ) -> ToolResponse:
        if self.debug:
            print(f"[ERROR] {content}")
        return ToolResponse(
            message={
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            },
            content=content,
            status="error",
            error_type=error_type,
        )

    def _panic_response(
        self,
        tool_call_id: str,
        function_name: str,
        error: Exception,
    ) -> ToolResponse:
        content = f"[PANIC] Tool '{function_name}' panicked: {error}"
        logger.error(
            "tool %s panicked (id=%s): %s",
            function_name,
            tool_call_id,
            error,
            exc_info=True,
        )
        return self._error_response(tool_call_id, "panic", content)

    def _get_tool_responses(
        self,
        tool_calls: list[ResponseToolCall],
    ) -> list[ToolResponse]:
        available_tools = {tool.name: tool for tool in self.tools}
        if not tool_calls:
            return []
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            futures = [
                executor.submit(
                    self._execute_tool_call,
                    call,
                    available_tools,
                )
                for call in tool_calls
            ]
            return [future.result() for future in futures]

    def _append_produced(
        self,
        messages: list[dict[str, Any]],
        message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.produced_callback is not None:
            self.produced_callback(message)
        messages.append(message)
        return messages

    def loop(
        self,
        messages: list[dict[str, Any]],
        cancel_event: threading.Event | None = None,
    ) -> list[dict[str, Any]]:
        if self.before_round_hook is not None:
            messages = [*messages, *self.before_round_hook()]
        params = self.build_chat_completion_params(messages, with_tool=True)

        round_start = time.monotonic()
        try:
            response = self._round_with_retry(params, 1, cancel_event)
        except Exception as exc:
            raise AgentLoopError(
                f"failed to get chat completion: {exc}",
                messages,
            ) from exc
        self._record_usage(response.usage.prompt_tokens)
        self._log_round(round_start, 1, response)

        next_messages = list(params["messages"])
        retry_count = 0
        round_number = 1

        while response.choices[0].finish_reason == "tool_calls":
            if cancel_event is not None and cancel_event.is_set():
                raise AgentLoopError("agent loop cancelled", next_messages)

            self._append_produced(
                next_messages,
                assistant_msg_to_param(response.choices[0].message),
            )
            tool_results = self._get_tool_responses(
                response.choices[0].message.tool_calls
            )
            for result in tool_results:
                self._append_produced(next_messages, result.message)

            if has_tool_errors(tool_results) and retry_count < self.max_tool_retries:
                retry_count += 1
                summary = get_error_summary(tool_results)
                if self.debug:
                    print(
                        f"[RETRY {retry_count}/{self.max_tool_retries}] "
                        f"Tool errors:\n{summary}"
                    )
                self._append_produced(
                    next_messages,
                    {
                        "role": "user",
                        "content": (
                            "Tool execution errors detected:\n\n"
                            f"{summary}\n\nPlease fix and retry. Retries left: "
                            f"{self.max_tool_retries - retry_count}"
                        ),
                    },
                )

            if self.debug:
                print(
                    f"[DEBUG] Sending {len(next_messages)} messages to LLM:\n"
                    + json.dumps(next_messages, indent=2, ensure_ascii=False)
                )
            if self.before_round_hook is not None:
                next_messages.extend(self.before_round_hook())
            params["messages"] = self.prepare(next_messages)
            next_messages = params["messages"]
            round_number += 1
            round_start = time.monotonic()
            try:
                response = self._round_with_retry(
                    params,
                    round_number,
                    cancel_event,
                )
            except Exception as exc:
                raise AgentLoopError(
                    f"chat completion (tool loop): {exc}",
                    next_messages,
                ) from exc
            self._record_usage(response.usage.prompt_tokens)
            self._log_round(round_start, round_number, response)

        self._append_produced(
            next_messages,
            assistant_msg_to_param(response.choices[0].message),
        )
        return next_messages

    def _stream_timeout(self) -> float:
        if self.config.timeout is not None and self.config.timeout > 0:
            return self.config.timeout
        return DEFAULT_STREAM_TIMEOUT

    def _round_with_retry(
        self,
        params: dict[str, Any],
        round_number: int,
        cancel_event: threading.Event | None,
    ) -> Completion:
        try:
            return self._stream_round(params, cancel_event)
        except StreamTimeoutError:
            if cancel_event is not None and cancel_event.is_set():
                raise
            logger.warning(
                "llm round %d: stream watchdog timeout, retrying once",
                round_number,
            )
            return self._stream_round(params, cancel_event)

    def _stream_round(
        self,
        params: dict[str, Any],
        cancel_event: threading.Event | None,
    ) -> Completion:
        if cancel_event is not None and cancel_event.is_set():
            raise AgentCancelledError("agent call cancelled")

        timeout = self._stream_timeout()
        stop = threading.Event()
        timed_out = threading.Event()
        lock = threading.Lock()
        last_beat = [time.monotonic()]
        stream_holder: dict[str, Any] = {}

        def close_stream() -> None:
            with lock:
                stream = stream_holder.get("stream")
            if stream is not None:
                close = getattr(stream, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass

        def watchdog() -> None:
            while not stop.is_set():
                if cancel_event is not None and cancel_event.is_set():
                    close_stream()
                    return
                with lock:
                    elapsed = time.monotonic() - last_beat[0]
                remaining = timeout - elapsed
                if remaining > 0:
                    stop.wait(min(remaining, 0.05))
                    continue
                with lock:
                    elapsed = time.monotonic() - last_beat[0]
                if elapsed < timeout:
                    continue
                timed_out.set()
                close_stream()
                return

        watcher = threading.Thread(target=watchdog, daemon=True)
        watcher.start()
        accumulator = CompletionAccumulator()
        request = dict(params)
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}

        try:
            stream = self.client.chat.completions.create(
                **request,
                timeout=timeout,
            )
            with lock:
                stream_holder["stream"] = stream
            if timed_out.is_set():
                raise StreamTimeoutError()
            if cancel_event is not None and cancel_event.is_set():
                raise AgentCancelledError("agent call cancelled")

            for chunk in stream:
                with lock:
                    last_beat[0] = time.monotonic()
                accumulator.add_chunk(chunk)

            if timed_out.is_set():
                raise StreamTimeoutError()
            if cancel_event is not None and cancel_event.is_set():
                raise AgentCancelledError("agent call cancelled")
            response = accumulator.completion()
            if not response.choices:
                raise RuntimeError("llm stream: no choices returned")
            return response
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise AgentCancelledError("agent call cancelled") from exc
            if timed_out.is_set() or isinstance(exc, APITimeoutError):
                raise StreamTimeoutError() from exc
            raise
        finally:
            stop.set()
            close_stream()
            watcher.join()

    def _log_round(
        self,
        start: float,
        round_number: int,
        response: Completion,
    ) -> None:
        message = response.choices[0].message
        names = [
            call.function.name + digest_args(call.function.arguments)
            for call in message.tool_calls
        ]
        elapsed_ms = round((time.monotonic() - start) * 1000)
        logger.info(
            "llm round %d: %dms prompt=%d completion=%d finish=%s tools=%s",
            round_number,
            elapsed_ms,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.choices[0].finish_reason,
            names,
        )


def validate_required_args(
    parameters: dict[str, Any] | None,
    args: dict[str, Any],
) -> list[str]:
    if parameters is None:
        return []
    required = parameters.get("required")
    if not isinstance(required, list):
        return []
    return [name for name in required if isinstance(name, str) and name not in args]


def has_tool_errors(results: list[ToolResponse]) -> bool:
    return any(result.status == "error" for result in results)


def get_error_summary(results: list[ToolResponse]) -> str:
    parts = [
        f"- {result.error_type or 'unknown'}: {result.content[:200]}"
        for result in results
        if result.status == "error"
    ]
    return "\n".join(parts) if parts else "unknown error"


def digest_args(raw: str) -> str:
    if not raw:
        return ""
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(arguments, dict):
        return ""
    for key in ("command", "path", "message", "content", "old_string", "input_token"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            value = " ".join(value.split())
            if len(value) > 24:
                value = value[:24] + "…"
            return f"({value})"
    return ""
