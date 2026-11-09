import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent_runtime import CompressMemory, LLMConfig, is_summary_message, new_openai_client
from agent_runtime.estimate import estimate_messages
from agent_runtime.memory import (
    COMPRESS_TRIGGER_RATIO,
    SUMMARIZE_ATTEMPTS,
    SUMMARIZE_ATTEMPT_TIMEOUT,
    SUMMARIZE_RETRY_GAP,
    SUMMARY_FOLD_AT,
    summary_message,
    tool_group_safe_cut,
)


def load_config() -> LLMConfig:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    return LLMConfig(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ["DEEPSEEK_BASEURL"],
        model=os.environ["DEEPSEEK_MODEL"],
        temperature=float(os.environ["DEEPSEEK_TEMPERATURE"]),
        timeout=float(os.environ["DEEPSEEK_TIMEOUT"]),
    )


def show(title, messages):
    print(f"\n{title}")
    print(json.dumps(messages, ensure_ascii=False, indent=2))


def main() -> None:
    config = load_config()
    client = new_openai_client(config)

    def summarize(messages):
        response = client.chat.completions.create(
            model=config.model,
            temperature=0,
            timeout=config.timeout,
            messages=[
                {
                    "role": "system",
                    "content": "你是中文对话摘要器，保留目标、事实、工具结果和待办事项。",
                },
                {
                    "role": "user",
                    "content": "请简洁总结以下消息：\n"
                    + json.dumps(messages, ensure_ascii=False),
                },
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("模型返回了空摘要")
        return content

    print(
        f"触发比例={COMPRESS_TRIGGER_RATIO}，摘要折叠点={SUMMARY_FOLD_AT}，"
        f"摘要最多尝试={SUMMARIZE_ATTEMPTS} 次，单次超时={SUMMARIZE_ATTEMPT_TIMEOUT}s，"
        f"重试间隔={SUMMARIZE_RETRY_GAP}s"
    )
    system = {"role": "system", "content": "你是中文技术学习助手。"}
    small = [system, {"role": "user", "content": "短消息"}]
    disabled = CompressMemory(max_input_tokens=0, summarizer=summarize)
    below_threshold = CompressMemory(max_input_tokens=10_000, summarizer=summarize)
    print(f"max<=0 时禁用且不复制：{disabled.prepare(small) is small}")
    print(f"低于 80% 阈值时不压缩：{below_threshold.prepare(small) is small}")

    assistant_call = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"北京"}',
                },
            }
        ],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"weather":"晴","temperature":"22°C"}',
    }
    recent_one = {"role": "user", "content": "最近消息一：保留工具调用和结果的对应关系。"}
    recent_two = {"role": "assistant", "content": "最近消息二：明白。"}
    messages = [
        system,
        {"role": "user", "content": "我的学习目标是掌握 Agent 的工具调用与上下文压缩。"},
        {"role": "assistant", "content": "我会保留这个目标，并观察旧消息如何被摘要。"},
        {"role": "user", "content": "请查询北京天气。"},
        assistant_call,
        tool_result,
        {"role": "user", "content": "这条消息位于工具组之后，也需要按安全切点处理。"},
        recent_one,
        recent_two,
    ]
    keep_recent = 3
    raw_cut = len(messages[1:]) - keep_recent
    safe_cut = tool_group_safe_cut(messages[1:], 0, raw_cut)
    print(
        f"\n基础压缩：估算={estimate_messages(messages)} tokens，"
        f"阈值={int(100 * COMPRESS_TRIGGER_RATIO)}，"
        f"原始切点={raw_cut}，工具组安全切点={safe_cut}"
    )
    basic_memory = CompressMemory(
        max_input_tokens=100,
        keep_recent=keep_recent,
        summarizer=summarize,
    )
    basic_result = basic_memory.prepare(messages)
    show("system 保留，旧区间变成摘要，工具组完整留在尾部：", basic_result)
    print(f"新消息是摘要：{is_summary_message(basic_result[1])}")
    print(f"实际使用计数在压缩后清零：{basic_memory.last_prompt_tokens}")

    two_summaries = [summary_message("旧摘要一"), summary_message("旧摘要二")]
    protected_messages = [
        system,
        *two_summaries,
        {"role": "user", "content": "需要并入新摘要的旧消息。"},
        {"role": "assistant", "content": "旧回复。"},
        {"role": "user", "content": "最新消息。"},
    ]
    protected_memory = CompressMemory(
        max_input_tokens=100,
        keep_recent=1,
        summarizer=summarize,
    )
    protected_result = protected_memory.prepare(protected_messages)
    show("少于五个旧摘要时，旧摘要继续受保护：", protected_result)
    print(f"保护成功：{protected_result[1:3] == two_summaries}")

    five_summaries = [summary_message(f"第 {index} 个旧摘要") for index in range(1, 6)]
    folding_messages = [
        system,
        *five_summaries,
        {"role": "user", "content": "这一阶段学习摘要折叠。"},
        {"role": "assistant", "content": "五个旧摘要不再受保护。"},
        {"role": "user", "content": "保留这条最新消息。"},
    ]
    folding_memory = CompressMemory(
        max_input_tokens=100,
        keep_recent=1,
        summarizer=summarize,
    )
    folding_result = folding_memory.prepare(folding_messages)
    show("达到五个旧摘要时，它们全部参与折叠：", folding_result)
    print(
        f"折叠后摘要数：{sum(is_summary_message(message) for message in folding_result)}；"
        f"最新消息保留：{folding_result[-1] == folding_messages[-1]}"
    )

    usage_messages = [
        system,
        {"role": "user", "content": "短旧消息"},
        {"role": "assistant", "content": "短旧回复"},
        {"role": "user", "content": "最新消息"},
    ]
    usage_memory = CompressMemory(
        max_input_tokens=1000,
        keep_recent=1,
        summarizer=summarize,
    )
    print(f"\n真实 usage 触发前，估算值={estimate_messages(usage_messages)}，阈值=800")
    usage_memory.record_usage(800)
    usage_result = usage_memory.prepare(usage_messages)
    print(f"record_usage(800) 可以独立触发压缩：{is_summary_message(usage_result[1])}")
    print(f"触发后 usage 计数清零：{usage_memory.last_prompt_tokens}")

    no_region_memory = CompressMemory(max_input_tokens=1000, keep_recent=40, summarizer=summarize)
    no_region_memory.record_usage(800)
    print(
        "超过阈值但没有可压缩区间时原样返回："
        f"{no_region_memory.prepare(small) is small}"
    )

    attempts = 0

    def retry_then_succeed(messages):
        nonlocal attempts
        summary = summarize(messages)
        attempts += 1
        print(f"真实摘要调用第 {attempts} 次")
        if attempts <= 5:
            raise RuntimeError(f"演示主动丢弃第 {attempts} 次摘要结果")
        return summary

    retry_messages = [
        system,
        {"role": "user", "content": "旧消息一：需要摘要。"},
        {"role": "assistant", "content": "旧消息二：包含重要上下文。"},
        {"role": "user", "content": "最新消息：必须保留。"},
    ]
    retry_memory = CompressMemory(
        max_input_tokens=80,
        keep_recent=1,
        summarizer=retry_then_succeed,
    )
    fallback_result = retry_memory.prepare(retry_messages)
    show("前三次摘要失败，降级为滑动窗口（system + 最新消息）：", fallback_result)

    retry_memory.record_usage(80)
    retry_result = retry_memory.prepare(retry_messages)
    show("再次触发后，第六次真实摘要成功，生成摘要消息：", retry_result)
    print(f"总尝试次数={attempts}，摘要成功={is_summary_message(retry_result[1])}")

    no_summarizer = CompressMemory(max_input_tokens=100, keep_recent=1)
    no_summarizer.record_usage(80)
    no_summarizer_result = no_summarizer.prepare(retry_messages)
    show("未配置 summarizer 时也使用同样的降级策略：", no_summarizer_result)

    no_system_memory = CompressMemory(max_input_tokens=20, keep_recent=1, summarizer=summarize)
    no_system_messages = [
        {"role": "user", "content": "旧消息"},
        {"role": "assistant", "content": "旧回复"},
        {"role": "user", "content": "最新消息"},
    ]
    no_system_result = no_system_memory.prepare(no_system_messages)
    show("没有字符串 system 提示时不调用摘要器，直接降级：", no_system_result)


if __name__ == "__main__":
    main()
