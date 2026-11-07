from pathlib import Path
from typing import Any

import torch

from .aut import AuT
from .modern_config import ModernASRConfig
from .qwen3_asr import Qwen3ASR


def save_llm_checkpoint(path: str | Path, model: Qwen3ASR, step: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "qwen3-asr-teaching-v1",
            "config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "step": step,
        },
        path,
    )


def load_llm_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[Qwen3ASR, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "qwen3-asr-teaching-v1":
        raise ValueError("not a teaching Qwen3ASR checkpoint")
    model = Qwen3ASR(ModernASRConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def save_aut_checkpoint(path: str | Path, model: AuT, step: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = ModernASRConfig(model.audio_config, model.llm_config)
    torch.save(
        {
            "format": "aut-teaching-v1",
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "step": step,
        },
        path,
    )


def load_aut_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[AuT, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "aut-teaching-v1":
        raise ValueError("not a teaching AuT checkpoint")
    config = ModernASRConfig.from_dict(payload["config"])
    model = AuT(config.audio, config.llm)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload
