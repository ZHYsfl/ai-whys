from typing import Any


def estimate_messages(messages: list[dict[str, Any]]) -> int:
    return sum(8 + estimate_tokens(message_text(message)) for message in messages)


def message_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])

    if message.get("role") == "assistant":
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(function.get("name") or "")
            parts.append(function.get("arguments") or "")
    return "".join(parts)


def estimate_tokens(text: str) -> int:
    cjk = sum(1 for char in text if is_cjk_heavy(char))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def is_cjk_heavy(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3000 <= codepoint <= 0x303F
        or 0xFF00 <= codepoint <= 0xFFEF
    )
