# Derived from kyutai-labs/moshi-finetune (Apache-2.0); see NOTICE and LICENSES/.

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import MoshiConfig
from .mimi import MimiModel
from .text import TextTokenizer


@dataclass
class DialogueSample:
    codes: torch.Tensor
    sample_id: str


@dataclass
class DialogueBatch:
    codes: torch.Tensor  # [B,1+2Q,T]
    sample_ids: tuple[str, ...]

    def to(self, device: str | torch.device) -> "DialogueBatch":
        return DialogueBatch(self.codes.to(device), self.sample_ids)


def collate_dialogue(samples: list[DialogueSample]) -> DialogueBatch:
    if not samples:
        raise ValueError("cannot collate an empty dialogue batch")
    streams = samples[0].codes.shape[0]
    if any(sample.codes.ndim != 2 or sample.codes.shape[0] != streams for sample in samples):
        raise ValueError("all dialogue samples must have the same stream count")
    max_frames = max(sample.codes.shape[-1] for sample in samples)
    codes = torch.full(
        (len(samples), streams, max_frames), -1, dtype=torch.long
    )
    for index, sample in enumerate(samples):
        codes[index, :, : sample.codes.shape[-1]] = sample.codes
    return DialogueBatch(codes, tuple(sample.sample_id for sample in samples))


class SyntheticDialogueDataset(Dataset):
    """Small discrete conversations for testing model mechanics without a codec."""

    def __init__(
        self,
        config: MoshiConfig | None = None,
        size: int = 16,
        frames: int = 8,
        seed: int = 0,
        text_dropout: float = 0.0,
        text_only_probability: float = 0.0,
    ) -> None:
        self.config = config or MoshiConfig()
        if size < 1 or frames < self.config.max_delay + 2:
            raise ValueError("not enough synthetic frames")
        if not 0 <= text_dropout <= 1 or not 0 <= text_only_probability <= 1:
            raise ValueError("probabilities must be in [0,1]")
        self.size = size
        self.frames = frames
        self.seed = seed
        self.text_dropout = text_dropout
        self.text_only_probability = text_only_probability

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> DialogueSample:
        rng = np.random.default_rng(self.seed + index)
        config = self.config
        frames = self.frames # 8
        text = np.full(frames, config.existing_text_padding_id, dtype=np.int64) # [3, 3, 3, 3, 3, 3, 3, 3]
        words = ("a", "b", "ab", "ba")
        for start in range(0, frames, 3):
            # start is 0, 3, 6
            word = words[int(rng.integers(len(words)))]
            if start > 0 and text[start - 1] == config.existing_text_padding_id:
                text[start - 1] = config.existing_text_end_padding_id # 0
            for offset, byte in enumerate(word.encode("ascii")):
                if start + offset < frames:
                    text[start + offset] = byte + 4

        semantic = (text + index) % config.card
        semantic[text == config.existing_text_padding_id] = index % config.card
        moshi_audio = [semantic]
        for codebook in range(1, config.dep_q):
            moshi_audio.append((semantic + codebook + index) % config.card)
        user_semantic = (np.arange(frames) + 2 * index) % config.card
        user_audio = [user_semantic]
        for codebook in range(1, config.dep_q):
            user_audio.append((user_semantic + 2 * codebook) % config.card)

        if rng.random() < self.text_dropout:
            text[:] = -1
        codes = np.concatenate(
            (
                text[None], # (1,T)
                np.stack(moshi_audio), # (8,T)
                np.stack(user_audio), # (8,T)
            ),
            axis=0,
        ) # (17,T)
        if rng.random() < self.text_only_probability:
            codes[1:] = -1
        return DialogueSample(torch.from_numpy(codes), f"synthetic-{index}")


class Interleaver:
    """Place word tokens on the audio-frame grid and mark word boundaries with EPAD."""

    def __init__(
        self,
        tokenizer: TextTokenizer,
        audio_frame_rate: float, # 12.5
        text_padding: int, # 3
        end_of_text_padding: int, # 0
        audio_delay: float = 0.0, # 文本相对音频的整体平移
    ) -> None:
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.audio_delay = audio_delay

    def build_token_stream(
        self,
        alignments: list[tuple[str, tuple[float, float], str]], # 一个列表，每项(词，(开始时间,结束时间),说话人)
        segment_duration: float,
        main_speaker: str = "SPEAKER_MAIN",
    ) -> torch.Tensor:
        frame_count = math.ceil(segment_duration * self.audio_frame_rate)
        text_tokens = [self.text_padding] * frame_count
        pending: deque[int] = deque()
        prepared = []
        for word, (start, end), speaker in alignments:
            start -= self.audio_delay
            end -= self.audio_delay
            if speaker == main_speaker and end > 0 and end > start:
                prepared.append((self.tokenizer.encode(word.strip()), start, end))
        prepared.sort(key=lambda item: item[1])
        alignment_index = 0
        for frame in range(frame_count):
            while (
                alignment_index < len(prepared)
                and prepared[alignment_index][1] * self.audio_frame_rate < frame + 1
            ):
                pending = deque(prepared[alignment_index][0])
                alignment_index += 1
            if pending:
                if frame > 0 and text_tokens[frame - 1] == self.text_padding:
                    text_tokens[frame - 1] = self.end_of_text_padding
                text_tokens[frame] = pending.popleft()
        if self.audio_delay < 0:
            prefix = min(frame_count, int(-self.audio_delay * self.audio_frame_rate))
            text_tokens[:prefix] = [-1] * prefix
        return torch.tensor(text_tokens, dtype=torch.long)


class StereoDialogueDataset(Dataset):
    """Official fine-tune layout: left=Moshi, right=user, JSON word alignments."""

    def __init__(
        self,
        manifest: str | Path,
        mimi: MimiModel,
        tokenizer: TextTokenizer,
        model_config: MoshiConfig,
        duration_sec: float = 10.0,
    ) -> None:
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        self.manifest = Path(manifest)
        self.mimi = mimi
        self.mimi.freeze()
        self.model_config = model_config
        self.interleaver = Interleaver(
            tokenizer,
            mimi.frame_rate,
            model_config.existing_text_padding_id,
            model_config.existing_text_end_padding_id,
        )
        self.records = []
        with self.manifest.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                path = Path(record["path"])
                if not path.is_absolute():
                    path = self.manifest.parent / path
                duration = float(record["duration"])
                start = 0.0
                while start < duration:
                    self.records.append((path, start, min(duration_sec, duration - start)))
                    start += duration_sec
        if not self.records:
            raise ValueError("manifest contains no audio")

    def __len__(self) -> int:
        return len(self.records)

    @torch.no_grad()
    def __getitem__(self, index: int) -> DialogueSample:
        import soundfile as sf

        path, start_sec, duration_sec = self.records[index]
        waveform, sample_rate = sf.read(
            path,
            start=int(start_sec * self.mimi.sample_rate),
            stop=int((start_sec + duration_sec) * self.mimi.sample_rate),
            dtype="float32",
            always_2d=True,
        )
        if sample_rate != self.mimi.sample_rate:
            raise ValueError(f"{path} must be {self.mimi.sample_rate} Hz")
        if waveform.shape[1] != 2:
            raise ValueError(f"{path} must be stereo [Moshi,user]")
        waveform = torch.from_numpy(waveform.T.copy())
        moshi_codes = self.mimi.encode(waveform[0][None, None])[0]
        user_codes = self.mimi.encode(waveform[1][None, None])[0]
        frame_count = moshi_codes.shape[-1]

        alignment_path = path.with_suffix(".json")
        alignments = json.loads(alignment_path.read_text(encoding="utf-8"))["alignments"]
        selected = []
        for word, (word_start, word_end), speaker in alignments:
            if word_end > start_sec and word_start < start_sec + duration_sec:
                selected.append(
                    (word, (word_start - start_sec, word_end - start_sec), speaker)
                )
        text = self.interleaver.build_token_stream(selected, frame_count / self.mimi.frame_rate)
        if text.shape[0] < frame_count:
            text = F.pad(text, (0, frame_count - text.shape[0]), value=-1)
        text = text[:frame_count]
        codes = torch.cat((text[None], moshi_codes, user_codes), dim=0)
        return DialogueSample(codes, f"{path.stem}:{start_sec:.3f}")
