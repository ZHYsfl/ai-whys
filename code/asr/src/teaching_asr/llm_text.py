import re
import string

import torch

from .data import SpeechBatch


class LanguageTokenizer:
    def __init__(self) -> None:
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.language_en_id = 3
        self.asr_text_id = 4
        self.characters = (" ", "'") + tuple(string.ascii_lowercase)
        self._char_to_id = {char: index + 5 for index, char in enumerate(self.characters)}
        self.vocab_size = len(self.characters) + 5

    @property
    def prompt_ids(self) -> list[int]:
        return [self.bos_id, self.language_en_id, self.asr_text_id] # 1，3，4

    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def encode(self, text: str) -> list[int]:
        ids = []
        for char in self.normalize(text):
            if char not in self._char_to_id:
                raise ValueError(f"unsupported character {char!r}")
            ids.append(self._char_to_id[char])
        return ids

    def decode(self, ids) -> str:
        reverse = {index: char for char, index in self._char_to_id.items()}
        chars = []
        for index in ids:
            index = int(index)
            if index == self.eos_id:
                break
            if index in reverse:
                chars.append(reverse[index])
        return "".join(chars)


def collate_language_targets(
    batch: SpeechBatch, tokenizer: LanguageTokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [
        [tokenizer.bos_id] + tokenizer.encode(transcript) + [tokenizer.eos_id]
        for transcript in batch.transcripts
    ]
    max_length = max(len(sequence) for sequence in sequences)
    targets = torch.full(
        (len(sequences), max_length), tokenizer.pad_id, dtype=torch.long
    ) # (batch_size, max_length)
    for index, sequence in enumerate(sequences):
        targets[index, : len(sequence)] = torch.tensor(sequence)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long) # (batch_size,)
    return targets.to(batch.waveforms.device), lengths.to(batch.waveforms.device)
