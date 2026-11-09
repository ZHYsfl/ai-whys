from openai import OpenAI

from .llm_config import LLMConfig


def new_openai_client(config: LLMConfig) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url)
