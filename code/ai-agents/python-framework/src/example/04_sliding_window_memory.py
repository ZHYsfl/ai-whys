import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent_runtime import (
    Agent,
    LLMConfig,
    MemoryConfig,
    MemoryMode,
    SlideWindowMemory,
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


def print_window(agent, messages, title):
    prepared = agent.prepare(messages)
    print(f"\n{title}：system + 最近 {len(prepared) - 1} 条非 system 消息")
    print(json.dumps(prepared, ensure_ascii=False, indent=2))
    return prepared


def main() -> None:
    agent = Agent(
        load_config(),
        memory_config=MemoryConfig(mode=MemoryMode.SLIDE_WINDOW, max=4),
    )
    history = [
        {
            "role": "system",
            "content": "你是记忆演示助手。只能根据当前请求里实际存在的对话回答，不知道就明确说不知道。",
        }
    ]
    questions = [
        "请记住：我的幸运数字是 31415。只回复“已经记住”。",
        "我们换一个话题：用一句话解释滑动窗口。",
        "再换一个话题：用一句话解释工具调用。",
        "我的幸运数字是多少？不要猜测。",
    ]

    for index, question in enumerate(questions, start=1):
        outgoing = [*history, {"role": "user", "content": question}]
        prepared = print_window(agent, outgoing, f"第 {index} 轮实际发送窗口")
        if index == len(questions):
            assert "31415" not in json.dumps(prepared, ensure_ascii=False)
            print("最早包含幸运数字的消息已经滑出窗口。")
        history, answer = agent.chat_without_tool_call(outgoing)
        print(f"模型回答：{answer}")

    assistant_call = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ],
    }
    orphan_tool = {"role": "tool", "tool_call_id": "call-1", "content": "旧结果"}
    latest = {"role": "user", "content": "最新消息"}
    orphan_demo = [history[0], assistant_call, orphan_tool, latest]
    print("\n窗口若从孤立 tool 消息开始，会继续丢弃它：")
    print(
        json.dumps(
            SlideWindowMemory(max_messages=2).prepare(orphan_demo),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
