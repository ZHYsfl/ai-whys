import json
import os
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


def get_weather(args):
    city = args["city"]
    result = {"city": city, "weather": "晴", "temperature": "22°C"}
    print(f"工具确实执行了：{result}")
    return json.dumps(result, ensure_ascii=False)


def main() -> None:
    agent = Agent(
        load_config(),
        tools=[
            Tool(
                name="get_weather",
                description="查询指定城市的当前天气",
                func=get_weather,
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ],
    )
    history = agent.loop(
        [{"role": "user", "content": "请调用 get_weather 查询北京天气，再告诉我结果。"}]
    )

    tool_messages = [message for message in history if message["role"] == "tool"]
    print(f"工具返回消息：{tool_messages}")
    print(f"模型最终回答：{history[-1].get('content', '')}")


if __name__ == "__main__":
    main()
