# Derived from kyutai-labs/moshi-finetune (Apache-2.0); see NOTICE and LICENSES/.

from pathlib import Path
from typing import Any

import torch

from .config import MimiConfig, MoshiConfig
from .lora import apply_lora, trainable_state_dict
from .mimi import MimiModel
from .model import MoshiLM


def _save(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def save_mimi_checkpoint(path: str | Path, model: MimiModel, step: int = 0) -> None:
    _save(
        path,
        {
            "format": "mimi-teaching-v1",
            "config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "step": step,
        },
    )


def load_mimi_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[MimiModel, dict[str, Any]]:
    payload = _load(path)
    if payload.get("format") != "mimi-teaching-v1":
        raise ValueError("not a teaching Mimi checkpoint")
    model = MimiModel(MimiConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def save_moshi_checkpoint(path: str | Path, model: MoshiLM, step: int = 0) -> None:
    _save(
        path,
        {
            "format": "moshi-teaching-v1",
            "config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "step": step,
        },
    )


def load_moshi_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[MoshiLM, dict[str, Any]]:
    payload = _load(path)
    if payload.get("format") != "moshi-teaching-v1":
        raise ValueError("not a teaching Moshi checkpoint")
    model = MoshiLM(MoshiConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def save_lora_checkpoint(path: str | Path, model: MoshiLM, step: int = 0) -> None:
    adapters = trainable_state_dict(model)
    if not adapters:
        raise ValueError("no trainable LoRA/embedding parameters found")
    from .lora import LoRALinear

    layers = [module for module in model.modules() if isinstance(module, LoRALinear)]
    if not layers:
        raise ValueError("no LoRA layers found")
    _save(
        path,
        {
            "format": "moshi-lora-teaching-v1",
            "config": model.config.to_dict(),
            "lora_config": {
                "rank": layers[0].rank,
                "scaling": layers[0].scaling,
                "finetune_embeddings": any(
                    parameter.requires_grad and "emb" in name
                    for name, parameter in model.named_parameters()
                ),
            },
            "adapter_state_dict": adapters,
            "step": step,
        },
    )


def load_lora_checkpoint(
    path: str | Path,
    base_model: MoshiLM,
    rank: int | None = None,
    scaling: float | None = None,
    finetune_embeddings: bool | None = None,
    device: str | torch.device = "cpu",
) -> tuple[MoshiLM, dict[str, Any]]:
    payload = _load(path)
    if payload.get("format") != "moshi-lora-teaching-v1":
        raise ValueError("not a teaching LoRA checkpoint")
    if MoshiConfig.from_dict(payload["config"]) != base_model.config:
        raise ValueError("adapter and base model configurations differ")
    lora_config = payload.get("lora_config", {})
    rank = lora_config.get("rank", 8) if rank is None else rank
    scaling = lora_config.get("scaling", 2.0) if scaling is None else scaling
    finetune_embeddings = (
        lora_config.get("finetune_embeddings", False)
        if finetune_embeddings is None
        else finetune_embeddings
    )
    apply_lora(base_model, rank, scaling, finetune_embeddings)
    result = base_model.load_state_dict(payload["adapter_state_dict"], strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected adapter keys: {result.unexpected_keys}")
    base_model.to(device).eval()
    return base_model, payload
