from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str | Path, expected_sample_rate: int | None = None) -> torch.Tensor:
    path = Path(path)
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    # waveform.shape is (num_samples, num_channels) 即 [T, C]
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise ValueError(
            f"{path} has sample rate {sample_rate}; expected {expected_sample_rate}. "
            "The teaching project does not resample implicitly."
        )
    if waveform.shape[0] == 0:
        raise ValueError(f"{path} contains no audio samples")
    mono = waveform.mean(axis=1) # 将多通道音频转换为单通道音频,即对声道求平均
    # mono.shape is (num_samples,) 即 [T]
    if not np.isfinite(mono).all():
        raise ValueError(f"{path} contains NaN or infinite samples")
    return torch.from_numpy(mono.copy()) # 将numpy数组转换为torch张量
