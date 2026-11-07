from pathlib import Path
from typing import Any

import torch

from .config import ASRConfig
from .model import ConformerCTC
from .text import CharTokenizer


def save_checkpoint(path: str | Path, model: ConformerCTC, step: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": model.config.to_dict(),
        "tokenizer": model.tokenizer.to_dict(),
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ConformerCTC, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ConformerCTC(
        ASRConfig.from_dict(payload["config"]),
        CharTokenizer.from_dict(payload["tokenizer"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload
