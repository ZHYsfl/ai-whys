import threading
import time
from types import SimpleNamespace

import pytest

from agent_runtime import (
    Agent,
    AgentLoopError,
    LLMConfig,
    MemoryConfig,
    MemoryMode,
    Tool,
    ToolCallInput,
    ToolExecutionError,
)
from agent_runtime.stream import CompletionAccumulator, assistant_msg_to_param


class FakeStream:
    def __init__(self, chunks, delays=None):
        self.chunks = list(chunks)
        self.delays = delays or {}
        self.index = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise RuntimeError("stream closed")
        if self.index >= len(self.chunks):
            raise StopIteration
        index = self.index
        self.index += 1
        if index in self.delays:
            time.sleep(self.delays[index])
        if self.closed:
            raise RuntimeError("stream closed")
        return self.chunks[index]

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, results=()):
        self.results = list(results)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.results:
            raise AssertionError("no fake response configured")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        if kwargs.get("stream") and isinstance(result, list):
            return FakeStream(result)
        return result


def content_chunks(text="done", prompt_tokens=10):
    return [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2},
        },
    ]


def tool_chunks(name="echo", arguments='{"text":"hello"}', call_id="call-1"):
    midpoint = len(arguments) // 2
    return [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments[:midpoint],
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": arguments[midpoint:]},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        },
    ]


def make_agent(client, **kwargs):
    return Agent(LLMConfig(model="test-model"), client=client, **kwargs)


def test_build_params_and_tool_registration() -> None:
    tool = Tool(
        name="echo",
        description="Echo text",
        func=lambda args: args["text"],
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    client = FakeClient()
    agent = make_agent(
        client,
        tools=[tool],
        debug=True,
    )
    agent.config.temperature = 0.2
    agent.config.extra_body = {"custom": True}

    params = agent.build_chat_completion_params(
        [{"role": "user", "content": "hi"}],
        with_tool=True,
    )

    assert params == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo text",
                    "parameters": tool.parameters,
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0.2,
        "extra_body": {"custom": True},
    }
    agent.remove_tool("echo")
    assert agent.get_tools() == []
    agent.add_tool(tool)
    assert agent.get_tools()[0]["function"]["name"] == "echo"


def test_chat_without_tools_appends_to_original_history() -> None:
    message = SimpleNamespace(content="answer", tool_calls=[])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = FakeClient([response])
    original = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "old"},
        {"role": "user", "content": "new"},
    ]
    agent = make_agent(
        client,
        memory_config=MemoryConfig(mode=MemoryMode.SLIDE_WINDOW, max=1),
    )

    next_messages, text = agent.chat_without_tool_call(original)

    assert text == "answer"
    assert len(client.calls[0]["messages"]) == 2
    assert next_messages == [*original, {"role": "assistant", "content": "answer"}]


def test_chat_stream_yields_content() -> None:
    client = FakeClient([content_chunks("streamed")])
    agent = make_agent(client)
    assert "".join(agent.chat_without_tool_call_stream([])) == "streamed"
    assert "stream_options" not in client.calls[0]


def test_execute_tool_calls_preserves_order_and_runs_concurrently() -> None:
    barrier = threading.Barrier(2)

    def wait_tool(args):
        barrier.wait(timeout=1)
        if args["value"] == "slow":
            time.sleep(0.02)
        return args["value"]

    agent = make_agent(
        FakeClient(),
        tools=[Tool("wait", "", wait_tool, {"type": "object"})],
    )
    calls = [
        ToolCallInput("1", "wait", '{"value":"slow"}'),
        ToolCallInput("2", "wait", '{"value":"fast"}'),
    ]

    messages = agent.execute_tool_calls(calls)

    assert [message["content"] for message in messages] == ["slow", "fast"]
    assert [message["tool_call_id"] for message in messages] == ["1", "2"]


def test_tool_failures_become_ordered_error_messages() -> None:
    def execute_error(args):
        raise ToolExecutionError("bad execution")

    def panic(args):
        raise RuntimeError("bad bug")

    agent = make_agent(
        FakeClient(),
        tools=[
            Tool(
                "needs_arg",
                "",
                lambda args: "ok",
                {"required": ["value"]},
            ),
            Tool("execute_error", "", execute_error, {}),
            Tool("panic", "", panic, {}),
        ],
    )
    calls = [
        ToolCallInput("1", "needs_arg", "{bad"),
        ToolCallInput("2", "missing", "{}"),
        ToolCallInput("3", "needs_arg", "{}"),
        ToolCallInput("4", "execute_error", "{}"),
        ToolCallInput("5", "panic", "{}"),
        ToolCallInput("6", "needs_arg", '{"value":null}'),
    ]

    messages = agent.execute_tool_calls(calls)

    prefixes = ["[PARSE_ERROR]", "[NOT_FOUND]", "[ARG_ERROR]", "[EXEC_ERROR]", "[PANIC]"]
    for message, prefix in zip(messages[:5], prefixes, strict=True):
        assert message["content"].startswith(prefix)
    assert messages[5]["content"] == "ok"


def test_loop_executes_tools_and_tracks_produced_messages() -> None:
    echo = Tool(
        "echo",
        "",
        lambda args: args["text"],
        {"required": ["text"]},
    )
    client = FakeClient([tool_chunks(), content_chunks("finished", 12)])
    produced = []
    hook_calls = []

    def before_round():
        hook_calls.append(len(hook_calls))
        return [{"role": "user", "content": f"now-{len(hook_calls)}"}]

    agent = make_agent(
        client,
        tools=[echo],
        before_round_hook=before_round,
        produced_callback=produced.append,
        memory_config=MemoryConfig(mode=MemoryMode.COMPRESS, max=10000),
    )
    initial = [{"role": "user", "content": "run echo"}]

    result = agent.loop(initial)

    assert [message["role"] for message in produced] == ["assistant", "tool", "assistant"]
    assert produced[0]["tool_calls"][0]["function"]["arguments"] == '{"text":"hello"}'
    assert produced[1]["content"] == "hello"
    assert result[-1] == {"role": "assistant", "content": "finished"}
    assert len(hook_calls) == 2
    assert [call["stream_options"] for call in client.calls] == [
        {"include_usage": True},
        {"include_usage": True},
    ]
    assert agent.memory.last_prompt_tokens == 12


def test_tool_error_adds_model_visible_retry_prompt() -> None:
    client = FakeClient([tool_chunks(name="missing"), content_chunks()])
    agent = make_agent(client, max_tool_retries=1)

    result = agent.loop([])

    retry_messages = [
        message
        for message in result
        if message["role"] == "user"
        and message.get("content", "").startswith("Tool execution errors detected")
    ]
    assert len(retry_messages) == 1
    assert "Retries left: 0" in retry_messages[0]["content"]


def test_loop_error_keeps_partial_tool_round() -> None:
    client = FakeClient([tool_chunks(), RuntimeError("api failed")])
    agent = make_agent(
        client,
        tools=[Tool("echo", "", lambda args: "ok", {})],
    )

    with pytest.raises(AgentLoopError, match=r"chat completion \(tool loop\)") as caught:
        agent.loop([{"role": "user", "content": "start"}])

    assert [message["role"] for message in caught.value.messages] == [
        "user",
        "assistant",
        "tool",
    ]


def test_stream_timeout_retries_once() -> None:
    slow = FakeStream(content_chunks("late"), delays={1: 0.05})
    client = FakeClient([slow, content_chunks("recovered")])
    agent = make_agent(client)
    agent.config.timeout = 0.02

    result = agent.loop([])

    assert result[-1]["content"] == "recovered"
    assert len(client.calls) == 2
    assert all(call["timeout"] == 0.02 for call in client.calls)


def test_cancelled_loop_does_not_call_api() -> None:
    client = FakeClient()
    agent = make_agent(client)
    cancel_event = threading.Event()
    cancel_event.set()
    initial = [{"role": "user", "content": "start"}]

    with pytest.raises(AgentLoopError) as caught:
        agent.loop(initial, cancel_event=cancel_event)

    assert caught.value.messages == initial
    assert client.calls == []


def test_accumulator_handles_multiple_choices_and_assistant_conversion() -> None:
    accumulator = CompletionAccumulator()
    accumulator.add_chunk(
        {
            "choices": [
                {"index": 1, "delta": {"content": "B"}, "finish_reason": None},
                {"index": 0, "delta": {"content": "A"}, "finish_reason": None},
            ]
        }
    )
    accumulator.add_chunk(
        {
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"},
                {"index": 1, "delta": {}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
    )
    completion = accumulator.completion()
    assert [choice.message.content for choice in completion.choices] == ["A", "B"]
    assert completion.usage.prompt_tokens == 3

    message = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(
                id="id",
                type="function",
                function=SimpleNamespace(name="f", arguments="{}"),
            ),
            SimpleNamespace(
                id="skip",
                type="function",
                function=SimpleNamespace(name="", arguments="{}"),
            ),
        ],
    )
    converted = assistant_msg_to_param(message)
    assert len(converted["tool_calls"]) == 1
    assert "content" not in converted
