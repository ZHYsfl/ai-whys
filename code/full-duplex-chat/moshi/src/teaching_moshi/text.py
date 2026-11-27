from pathlib import Path
from typing import Iterable, Protocol


class TextTokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    def encode(self, text: str, bos: bool = False) -> list[int]: ...

    def decode(self, ids: Iterable[int]) -> str: ...


class ByteTokenizer:
    """UTF-8 byte tokenizer for small experiments; IDs 0..3 are Moshi special tokens."""

    def __init__(self) -> None:
        self.end_of_text_padding_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.text_padding_id = 3

    @property
    def vocab_size(self) -> int:
        return 260

    def encode(self, text: str, bos: bool = False) -> list[int]:
        ids = [byte + 4 for byte in text.encode("utf-8")]
        return ([self.bos_id] if bos else []) + ids

    def decode(self, ids: Iterable[int]) -> str:
        data = bytes(int(index) - 4 for index in ids if int(index) >= 4) # ID 小于 4 的直接跳过。所以 BOS、EOS、padding 混在序列里时，decode 会静默丢掉它们——特殊 token 只对模型有意义，对人不可见。
        return data.decode("utf-8", errors="replace")


class SentencePieceTokenizer:
    def __init__(self, path: str | Path) -> None:
        import sentencepiece

        self.processor = sentencepiece.SentencePieceProcessor(str(path))

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    def encode(self, text: str, bos: bool = False) -> list[int]:
        ids = self.processor.encode(text)
        return ([self.bos_id] if bos else []) + ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.processor.decode(list(ids))
