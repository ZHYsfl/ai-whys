from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_wav(path: str | Path, expected_sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise ValueError(
            f"{path} has sample rate {sample_rate}; expected {expected_sample_rate}. "
            "Resample it explicitly before encoding."
        )
    waveform = waveform.mean(axis=1)
    return torch.from_numpy(waveform.copy()).unsqueeze(0), sample_rate


def save_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.ndim == 3 and waveform.shape[:2] == (1, 1):
        waveform = waveform[0]
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"expected mono waveform [1, T] or [1, 1, T], got {tuple(waveform.shape)}")
    samples = waveform.detach().cpu().squeeze(0).clamp(-1, 1).numpy()
    sf.write(path, samples, sample_rate, subtype="PCM_16")


def save_codes(path: str | Path, codes: torch.Tensor, original_length: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if codes.ndim != 3:
        raise ValueError(f"expected codes [B, Q, F], got {tuple(codes.shape)}")
    np.savez(
        path,
        codes=codes.detach().cpu().numpy().astype(np.int32),
        original_length=np.asarray(original_length, dtype=np.int64),
    )


def load_codes(path: str | Path) -> tuple[torch.Tensor, int]:
    with np.load(path, allow_pickle=False) as payload:
        codes = torch.from_numpy(payload["codes"].astype(np.int64))
        original_length = int(payload["original_length"])
    return codes, original_length
