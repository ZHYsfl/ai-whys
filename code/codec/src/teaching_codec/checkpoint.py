from pathlib import Path
from typing import Any

import torch

from .codec import TeachingCodec
from .config import CodecConfig
from .discriminator import MultiScaleDiscriminator


def save_checkpoint(
    path: str | Path,
    model: TeachingCodec,
    step: int,
    discriminator: MultiScaleDiscriminator | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    if discriminator is not None:
        payload["discriminator_config"] = {
            "scales": list(discriminator.scales),
            "base_channels": discriminator.discriminators[0].input_conv.out_channels,
        }
        payload["discriminator_state_dict"] = discriminator.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[TeachingCodec, MultiScaleDiscriminator | None, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = TeachingCodec(CodecConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    discriminator = None
    if "discriminator_state_dict" in payload:
        discriminator_config = payload.get(
            "discriminator_config", {"scales": [1, 2, 4], "base_channels": 16}
        )
        discriminator = MultiScaleDiscriminator(
            scales=tuple(discriminator_config["scales"]),
            base_channels=discriminator_config["base_channels"],
        )
        discriminator.load_state_dict(payload["discriminator_state_dict"])
        discriminator.to(device).eval()
    return model, discriminator, payload
