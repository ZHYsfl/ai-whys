from agent_runtime.estimate import estimate_messages, estimate_tokens, message_text
from agent_runtime.memory import (
    SUMMARY_OPEN_TAG,
    CompressMemory,
    MemoryConfig,
    MemoryMode,
    SlideWindowMemory,
    new_memory,
    summary_message,
)


def user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def test_token_estimate_matches_cjk_and_ascii_rules() -> None:
    assert estimate_tokens("你好abc") == 3
    assert estimate_tokens("全角，！") == 4
    assert estimate_tokens("") == 0

    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "tool_calls": [
            {"function": {"name": "run", "arguments": '{"x":1}'}}
        ],
    }
    assert message_text(message) == 'hellorun{"x":1}'
    assert estimate_messages([message]) == 8 + estimate_tokens('hellorun{"x":1}')


def test_slide_window_keeps_last_system_and_drops_orphan_tool() -> None:
    old_system = {"role": "system", "content": "old"}
    new_system = {"role": "system", "content": "new"}
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "f"}}
        ],
    }
    tool = {"role": "tool", "tool_call_id": "1", "content": "result"}
    latest = user("latest")
    memory = SlideWindowMemory(max_messages=2)

    prepared = memory.prepare(
        [old_system, user("old"), assistant, tool, latest, new_system]
    )

    assert prepared == [new_system, latest]


def test_compress_uses_tool_group_safe_cut() -> None:
    system = {"role": "system", "content": "system"}
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
    }
    tool = {"role": "tool", "tool_call_id": "call", "content": "result"}
    messages = [
        system,
        user("summarize me"),
        assistant,
        tool,
        user("also recent"),
        user("recent one"),
        user("recent two"),
    ]
    regions = []

    def summarize(region):
        regions.append(region)
        return "short summary"

    memory = CompressMemory(max_input_tokens=20, keep_recent=3, summarizer=summarize)
    prepared = memory.prepare(messages)

    assert regions == [[system, user("summarize me")]]
    assert prepared[0] == system
    assert prepared[1]["content"].startswith(SUMMARY_OPEN_TAG)
    assert prepared[2:] == [assistant, tool, user("also recent"), user("recent one"), user("recent two")]
    assert memory.last_prompt_tokens == 0


def test_compress_folds_five_old_summaries_into_new_summary() -> None:
    system = {"role": "system", "content": "system"}
    old_summaries = [summary_message(str(index)) for index in range(5)]
    recent = user("recent")
    memory = CompressMemory(
        max_input_tokens=20,
        keep_recent=1,
        summarizer=lambda messages: "folded",
    )

    prepared = memory.prepare([system, *old_summaries, user("old"), recent])

    assert prepared == [system, summary_message("folded"), recent]


def test_real_usage_can_trigger_compression_and_failure_falls_back() -> None:
    system = {"role": "system", "content": "system"}
    memory = CompressMemory(max_input_tokens=100, keep_recent=1)
    memory.record_usage(80)

    prepared = memory.prepare([system, user("old one"), user("old two"), user("recent")])

    assert prepared == [system, user("recent")]
    assert memory.last_prompt_tokens == 0


def test_below_threshold_returns_same_message_list() -> None:
    memory = CompressMemory(max_input_tokens=1000, keep_recent=1)
    messages = [user("small")]
    assert memory.prepare(messages) is messages


def test_memory_factory_and_summary_retry(monkeypatch) -> None:
    assert isinstance(
        new_memory(MemoryConfig(mode=MemoryMode.SLIDE_WINDOW, max=2)),
        SlideWindowMemory,
    )
    assert isinstance(new_memory(MemoryConfig()), type(new_memory(MemoryConfig())))

    attempts = []

    def summarize(messages):
        attempts.append(messages)
        if len(attempts) < 3:
            raise RuntimeError("temporary")
        return "ok"

    monkeypatch.setattr("agent_runtime.memory.SUMMARIZE_RETRY_GAP", 0)
    memory = CompressMemory(max_input_tokens=20, keep_recent=1, summarizer=summarize)
    prepared = memory.prepare(
        [{"role": "system", "content": "s"}, user("old"), user("recent")]
    )
    assert prepared[-2]["content"] == f"{SUMMARY_OPEN_TAG}\nok\n</conversation_summary>"
    assert len(attempts) == 3
