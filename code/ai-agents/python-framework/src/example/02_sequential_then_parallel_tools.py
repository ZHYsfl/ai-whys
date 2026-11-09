import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent_runtime import Agent, LLMConfig, Tool


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
    print(f"第一阶段，获取时间戳：{timestamp}")
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
    history = agent.loop(
        [
            {
                "role": "system",
                "content": (
                    "严格遵守工具编排：第一轮只调用一次 get_current_timestamp。"
                    "收到时间戳后，第二轮必须在同一条助手消息中同时发起三次 "
                    "get_weather，分别查询杭州、上海、北京，禁止逐个查询。"
                    "三个结果都返回后再汇总。"
                ),
            },
            {"role": "user", "content": "今天杭州、上海、北京的天气各自如何？"},
        ]
    )

    tool_rounds = [
        message["tool_calls"]
        for message in history
        if message["role"] == "assistant" and message.get("tool_calls")
    ]
    for index, calls in enumerate(tool_rounds, start=1):
        print(f"第 {index} 个工具轮次：{[call['function']['name'] for call in calls]}")

    if len(tool_rounds) != 2 or len(tool_rounds[0]) != 1 or len(tool_rounds[1]) != 3:
        raise RuntimeError("模型没有遵循先串联、再三路并联的两轮工具编排，请重新运行")
    print(f"模型最终回答：\n{history[-1].get('content', '')}")


if __name__ == "__main__":
    main()
