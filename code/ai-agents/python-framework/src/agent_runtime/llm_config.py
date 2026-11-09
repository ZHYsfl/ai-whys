from dataclasses import dataclass
from typing import Any


@dataclass
class LLMConfig:
    model: str
    api_key: str = ""
    base_url: str | None = None
    temperature: float | None = None
    timeout: float | None = None
    extra_body: dict[str, Any] | None = None
