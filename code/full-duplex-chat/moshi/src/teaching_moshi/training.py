# Derived from kyutai-labs/moshi-finetune (Apache-2.0); see NOTICE and LICENSES/.

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .checkpoint import save_lora_checkpoint, save_moshi_checkpoint
from .data import DialogueBatch, DialogueSample, collate_dialogue
from .losses import MoshiLossOutput, moshi_loss
from .lora import LoRALinear, merge_lora
from .model import MoshiLM


@dataclass
class TrainingConfig:
    max_steps: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    first_codebook_weight_multiplier: float = 100.0
    text_padding_weight: float = 0.5
    seed: int = 0
    log_every: int = 10
    save_adapters: bool = False

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("invalid training steps, batch size or learning rate")


def calculate_loss(
    model: MoshiLM, batch: DialogueBatch, config: TrainingConfig
) -> MoshiLossOutput:
    output = model(batch.codes)
    return moshi_loss(
        model,
        batch.codes,
        output,
        config.first_codebook_weight_multiplier,
        config.text_padding_weight,
    )


def training_step(
    model: MoshiLM,
    batch: DialogueBatch,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = calculate_loss(model, batch, config)
    if not torch.isfinite(losses.total):
        raise RuntimeError(f"non-finite Moshi loss: {losses.total.item()}")
    losses.total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        config.grad_clip,
    )
    if not torch.isfinite(grad_norm):
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()
    return {
        "loss": float(losses.total.detach()),
        "text": float(losses.text.detach()),
        "audio": float(losses.audio.detach()),
        "grad_norm": float(grad_norm.detach()),
    }


@torch.no_grad()
def evaluate(
    model: MoshiLM,
    dataset: Dataset[DialogueSample],
    config: TrainingConfig,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_dialogue)
    model.eval()
    totals = np.zeros(3)
    for batch in loader:
        batch = batch.to(device)
        losses = calculate_loss(model, batch, config)
        totals += [float(losses.total), float(losses.text), float(losses.audio)]
    count = max(len(loader), 1)
    return {"loss": totals[0] / count, "text": totals[1] / count, "audio": totals[2] / count}


def _save_training_checkpoint(
    path: Path, model: MoshiLM, step: int, save_adapters: bool
) -> None:
    has_lora = any(isinstance(module, LoRALinear) for module in model.modules())
    if save_adapters:
        if not has_lora:
            raise ValueError("save_adapters=True requires a LoRA-wrapped model")
        save_lora_checkpoint(path, model, step)
    elif has_lora:
        merged = copy.deepcopy(model)
        merge_lora(merged)
        save_moshi_checkpoint(path, merged, step)
    else:
        save_moshi_checkpoint(path, model, step)


def run_training(
    model: MoshiLM,
    dataset: Dataset[DialogueSample],
    config: TrainingConfig,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    eval_dataset: Dataset[DialogueSample] | None = None,
) -> list[dict[str, float]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_dialogue,
        generator=torch.Generator().manual_seed(config.seed),
    )
    history = []
    step = 0
    while step < config.max_steps:
        for batch in loader:
            step += 1
            metrics = training_step(model, batch.to(device), optimizer, config)
            metrics["step"] = step
            history.append(metrics)
            if step == 1 or step % config.log_every == 0:
                print(json.dumps(metrics, ensure_ascii=False))
            if eval_dataset is not None and step == config.max_steps:
                metrics.update({f"eval_{key}": value for key, value in evaluate(model, eval_dataset, config, device).items()})
            if step >= config.max_steps:
                break
    _save_training_checkpoint(
        output_dir / "checkpoint-last.pt", model, config.max_steps, config.save_adapters
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "training-config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    return history
