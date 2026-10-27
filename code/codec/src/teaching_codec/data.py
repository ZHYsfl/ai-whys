from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class SyntheticAudioDataset(Dataset):
    def __init__(
        self,
        size: int = 16,
        sample_rate: int = 16_000,
        num_samples: int = 16_000,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed + index)
        time = np.arange(self.num_samples, dtype=np.float64) / self.sample_rate
        fundamental = rng.uniform(80, 240)
        waveform = np.zeros(self.num_samples, dtype=np.float64)
        for harmonic in range(1, 7):
            frequency = fundamental * harmonic
            if frequency >= self.sample_rate / 2:
                break
            phase = rng.uniform(0, 2 * np.pi)
            waveform += np.sin(2 * np.pi * frequency * time + phase) / harmonic
        envelope_phase = rng.uniform(0, 2 * np.pi)
        envelope = 0.7 + 0.2 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * time + envelope_phase)
        waveform *= envelope
        waveform += 0.005 * rng.normal(size=self.num_samples)
        waveform /= np.maximum(np.abs(waveform).max(), 1e-8)
        return torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)


class WavFolderDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 16_000,
        num_samples: int = 16_000,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.files = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
        )
        if not self.files:
            raise ValueError(f"no WAV or FLAC files found under {self.root}")
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{path} has sample rate {sample_rate}; expected {self.sample_rate}. "
                "The teaching dataset does not resample implicitly."
            )
        waveform = waveform.mean(axis=1)
        if waveform.shape[0] >= self.num_samples:
            max_start = waveform.shape[0] - self.num_samples
            start = (self.seed + index) % (max_start + 1)
            waveform = waveform[start : start + self.num_samples]
        else:
            waveform = np.pad(waveform, (0, self.num_samples - waveform.shape[0]))
        return torch.from_numpy(waveform.copy()).unsqueeze(0)
