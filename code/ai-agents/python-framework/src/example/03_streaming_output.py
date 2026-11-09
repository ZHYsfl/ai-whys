import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent_runtime import Agent, LLMConfig, Tool, ToolCallInput, assistant_msg_to_param
from agent_runtime.stream import CompletionAccumulator


def load_config() -> LLMConfig:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    return LLMConfig(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ["DEEPSEEK_BASEURL"],
        model=os.environ["DEEPSEEK_MODEL"],
        temperature=float(os.environ["DEEPSEEK_TEMPERATURE"]),
        timeout=float(os.environ["DEEPSEEK_TIMEOUT"]),
    )


def get_current_timestamp(args):
    timestamp = int(time.time())
    print(f"\n第一阶段，获取时间戳：{timestamp}")
    return str(timestamp)


def get_weather(args):
    city = args["city"]
    timestamp = args["timestamp"]
    delays = {"杭州": 0.5, "上海": 0.3, "北京": 0.4}
    weather = {
        "杭州": ("小雨", "18°C"),
        "上海": ("多云", "20°C"),
        "北京": ("晴", "22°C"),
    }
    print(f"并行天气任务开始：{city}，线程={threading.current_thread().name}")
    time.sleep(delays[city])
    condition, temperature = weather[city]
    result = {
        "city": city,
        "date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d"),
        "weather": condition,
        "temperature": temperature,
    }
    print(f"并行天气任务结束：{city}")
    return json.dumps(result, ensure_ascii=False)


def stream_round(agent, messages):
    params = agent.build_chat_completion_params(messages, with_tool=True)
    stream = agent.client.chat.completions.create(
        **params,
        stream=True,
        stream_options={"include_usage": True},
        timeout=agent.config.timeout,
    )
    accumulator = CompletionAccumulator()
    try:
        for chunk in stream:
            accumulator.add_chunk(chunk)
            for choice in chunk.choices:
                if choice.delta.content:
                    print(choice.delta.content, end="", flush=True)
    finally:
        stream.close()
    response = accumulator.completion()
    if not response.choices:
        raise RuntimeError("模型没有返回 choice")
    return response


def main() -> None:
    tools = [
        Tool(
            name="get_current_timestamp",
            description="获取今天的 Unix 时间戳，必须先于天气查询调用",
            func=get_current_timestamp,
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_weather",
            description="根据城市和时间戳查询天气；不同城市可以并行调用",
            func=get_weather,
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": ["杭州", "上海", "北京"]},
                    "timestamp": {"type": "integer"},
                },
                "required": ["city", "timestamp"],
            },
        ),
    ]
    agent = Agent(load_config(), tools=tools)
    messages = [
        {
            "role": "system",
            "content": (
                "严格遵守工具编排：第一轮只调用一次 get_current_timestamp。"
                "收到时间戳后，第二轮必须在同一条助手消息中同时发起三次 "
                "get_weather，分别查询杭州、上海、北京，禁止逐个查询。"
                "三个结果都返回后再流式汇总答案。"
            ),
        },
        {"role": "user", "content": "今天杭州、上海、北京的天气各自如何？"},
    ]

    round_number = 1
    while True:
        print(f"\n第 {round_number} 轮模型输出：\n", flush=True)
        response = stream_round(agent, messages)
        message = response.choices[0].message
        messages.append(assistant_msg_to_param(message))
        if not message.tool_calls:
            print()
            break

        calls = [
            ToolCallInput(call.id, call.function.name, call.function.arguments)
            for call in message.tool_calls
        ]
        print(f"\n并行执行 {len(calls)} 个工具调用")
        messages.extend(agent.execute_tool_calls(calls))
        round_number += 1


if __name__ == "__main__":
    main()
