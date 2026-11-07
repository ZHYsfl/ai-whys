import re
import string
from collections.abc import Iterable
from typing import Any


DEFAULT_TOKENS = (" ", "'") + tuple(string.ascii_lowercase)


class CharTokenizer:
    def __init__(self, tokens: tuple[str, ...] = DEFAULT_TOKENS) -> None:
        if not tokens or any(len(token) != 1 for token in tokens) or len(set(tokens)) != len(tokens):
            raise ValueError("tokens must be unique, single-character strings")
        self.tokens = tokens
        self.blank_id = 0
        self._token_to_id = {token: index + 1 for index, token in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens) + 1

    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def encode(self, text: str) -> list[int]:
        normalized = self.normalize(text)
        ids = []
        for char in normalized:
            if char not in self._token_to_id:
                raise ValueError(f"unsupported character {char!r} in transcript {normalized!r}")
            ids.append(self._token_to_id[char])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for index in ids:
            index = int(index)
            if index == self.blank_id:
                continue
            if not 1 <= index < self.vocab_size:
                raise ValueError(f"token id {index} is outside [1, {self.vocab_size})")
            chars.append(self.tokens[index - 1])
        return "".join(chars)
    
    def to_dict(self) -> dict[str, Any]:
        return {"tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CharTokenizer":
        return cls(tuple(values["tokens"]))
