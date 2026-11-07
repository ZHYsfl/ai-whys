from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_io import load_audio
from .text import CharTokenizer


@dataclass
class SpeechExample:
    waveform: torch.Tensor # 单声道波形，形状[T]
    target: torch.Tensor # 文本编码后的字符 ID， 形状[L]
    transcript: str # 文本转录
    sample_id: str # 样本 ID


@dataclass
class SpeechBatch:
    waveforms: torch.Tensor # (batch_size, max_time_steps)
    waveform_lengths: torch.Tensor # (batch_size,) 每个样本的波形长度
    targets: torch.Tensor #  (batch_size, max_target_length) 文本编码后的字符 ID
    target_lengths: torch.Tensor # (batch_size,) 每条文本原来的真实长度
    transcripts: tuple[str, ...] # batch_size 个字符串
    sample_ids: tuple[str, ...] # batch_size 个样本 ID

    def to(self, device: str | torch.device) -> "SpeechBatch":
        return SpeechBatch(
            self.waveforms.to(device),
            self.waveform_lengths.to(device),
            self.targets.to(device),
            self.target_lengths.to(device),
            self.transcripts,
            self.sample_ids,
        )


def collate_speech(examples: list[SpeechExample]) -> SpeechBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(examples) # batch_size
    max_waveform = max(example.waveform.shape[0] for example in examples) # max_time_steps
    max_target = max(example.target.shape[0] for example in examples) # max_target_length
    waveforms = torch.zeros(batch_size, max_waveform, dtype=examples[0].waveform.dtype) # (batch_size, max_time_steps)
    targets = torch.zeros(batch_size, max_target, dtype=torch.long) # (batch_size, max_target_length)
    for index, example in enumerate(examples):
        waveform_length = example.waveform.shape[0]
        target_length = example.target.shape[0]
        if waveform_length == 0 or target_length == 0:
            raise ValueError(f"empty waveform or target for sample {example.sample_id}")
        waveforms[index, :waveform_length] = example.waveform
        targets[index, :target_length] = example.target
    return SpeechBatch(
        waveforms=waveforms,
        waveform_lengths=torch.tensor(
            [example.waveform.shape[0] for example in examples], dtype=torch.long
        ),
        targets=targets,
        target_lengths=torch.tensor(
            [example.target.shape[0] for example in examples], dtype=torch.long
        ),
        transcripts=tuple(example.transcript for example in examples),
        sample_ids=tuple(example.sample_id for example in examples),
    )


class SyntheticSpeechDataset(Dataset):
    '''
    合成教学数据
    '''
    def __init__(
        self,
        size: int = 16,
        sample_rate: int = 16_000,
        tokenizer: CharTokenizer | None = None,
        seed: int = 0,
    ) -> None:
        if size < 1 or sample_rate < 8_000:
            raise ValueError("size must be positive and sample_rate must be at least 8000")
        self.size = size
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer or CharTokenizer()
        self.seed = seed
        self.words = ("ab", "ba", "aa", "abc", "cab", "bb", "ca", "bcb")

    def __len__(self) -> int:
        return self.size

    def _render_symbol(self, symbol: str, rng: np.random.Generator) -> np.ndarray:
        '''
        把一个字符渲染成声音
        '''
        duration = rng.uniform(0.10, 0.18)
        count = int(duration * self.sample_rate)
        time = np.arange(count, dtype=np.float64) / self.sample_rate
        base_frequency = {"a": 220.0, "b": 440.0, "c": 880.0}[symbol]
        frequency = base_frequency * rng.uniform(0.96, 1.04) # 频率
        phase = rng.uniform(0, 2 * np.pi) # 初始相位
        tone = np.sin(2 * np.pi * frequency * time + phase) # 正弦波
        tone += 0.3 * np.sin(2 * np.pi * 2 * frequency * time + phase / 2) # 添加二次谐波，让声音不只是最单调单纯的正弦波
        envelope = np.sin(np.pi * np.linspace(0, 1, count)) ** 2 # 音量包络
        return tone * envelope

    def _render_text(self, text: str, rng: np.random.Generator) -> np.ndarray:
        '''
        把一段文字渲染成声音
        '''
        segments = [np.zeros(int(rng.uniform(0.02, 0.06) * self.sample_rate))]
        for char in text:
            if char == " ":
                segments.append(np.zeros(int(rng.uniform(0.08, 0.14) * self.sample_rate)))
            else:
                segments.append(self._render_symbol(char, rng))
                segments.append(np.zeros(int(rng.uniform(0.02, 0.05) * self.sample_rate)))
        waveform = np.concatenate(segments)
        waveform += 0.003 * rng.normal(size=waveform.shape[0]) # 加入少量高斯噪声
        peak = max(float(np.abs(waveform).max()), 1e-8) # 最大绝对振幅
        return (0.8 * waveform / peak).astype(np.float32) # 归一化到[-0.8, 0.8]之间，防止过载

    def __getitem__(self, index: int) -> SpeechExample:
        rng = np.random.default_rng(self.seed + index)
        word_count = int(rng.integers(1, 4))
        words = [self.words[int(rng.integers(len(self.words)))] for _ in range(word_count)]
        transcript = " ".join(words)
        waveform = torch.from_numpy(self._render_text(transcript, rng)) # (time_steps,)
        target = torch.tensor(self.tokenizer.encode(transcript), dtype=torch.long) # (target_length,)
        return SpeechExample(waveform, target, transcript, f"synthetic-{index}")


class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 16_000,
        tokenizer: CharTokenizer | None = None,
    ) -> None:
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer or CharTokenizer()
        records = []
        for transcript_path in sorted(self.root.rglob("*.trans.txt")):
            with transcript_path.open(encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    sample_id, transcript = line.split(maxsplit=1)
                    audio_path = transcript_path.with_name(f"{sample_id}.flac")
                    if not audio_path.exists():
                        audio_path = transcript_path.with_name(f"{sample_id}.wav")
                    if not audio_path.exists():
                        raise FileNotFoundError(f"audio for {sample_id} not found near {transcript_path}")
                    records.append((audio_path, self.tokenizer.normalize(transcript), sample_id))
        if not records:
            raise ValueError(f"no LibriSpeech .trans.txt records found under {self.root}")
        self.records = sorted(records, key=lambda record: record[2])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SpeechExample:
        audio_path, transcript, sample_id = self.records[index]
        waveform = load_audio(audio_path, self.sample_rate)
        target = torch.tensor(self.tokenizer.encode(transcript), dtype=torch.long)
        if target.numel() == 0:
            raise ValueError(f"empty transcript for {sample_id}")
        return SpeechExample(waveform, target, transcript, sample_id)
