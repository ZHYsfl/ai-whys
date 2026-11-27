from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from teaching_codec.discriminator import (
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from teaching_codec.losses import ReconstructionLoss

from .checkpoint import save_mimi_checkpoint
from .config import MimiConfig
from .mimi import MimiModel


@dataclass
class MimiBatch:
    waveform: torch.Tensor # (batch_size,input_channels=1, time_steps = sample_rate*seconds)
    semantic_targets: torch.Tensor | None = None # (batch_size, output_dimension, num_frames)


class SyntheticMimiDataset(Dataset):
    def __init__(
        self,
        config: MimiConfig,
        size: int = 16,
        frames: int = 2,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.size = size
        self.frames = frames
        self.sample_count = frames * config.frame_size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        time = np.arange(self.sample_count) / self.config.sample_rate
        frequency = 110 * (1 + index % 8)
        waveform = 0.5 * np.sin(2 * np.pi * frequency * time)
        waveform += 0.1 * np.sin(2 * np.pi * 2 * frequency * time)
        waveform += 0.005 * rng.normal(size=self.sample_count)
        targets = np.zeros((self.config.semantic_dimension, self.frames), dtype=np.float32)
        targets[index % self.config.semantic_dimension] = 1.0
        return (
            torch.from_numpy(waveform.astype(np.float32))[None], # (1, time_steps)
            torch.from_numpy(targets), # (semantic_dimension, num_frames)
        )


def _collate(items) -> MimiBatch:
    return MimiBatch(
        torch.stack([item[0] for item in items]),
        torch.stack([item[1] for item in items]),
    )


@dataclass
class MimiTrainingConfig:
    max_steps: int = 200
    batch_size: int = 4
    learning_rate: float = 8e-4
    discriminator_learning_rate: float = 8e-4
    use_gan: bool = True
    adversarial_only: bool = True
    adversarial_weight: float = 1.0
    feature_matching_weight: float = 2.0
    commitment_weight: float = 0.25
    semantic_weight: float = 1.0
    grad_clip: float = 1.0
    log_every: int = 10


def _requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def mimi_training_step(
    model: MimiModel,
    batch: MimiBatch,
    discriminator: MultiScaleDiscriminator | None,
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer | None,
    config: MimiTrainingConfig,
) -> dict[str, float]:
    if config.use_gan and (discriminator is None or discriminator_optimizer is None):
        raise ValueError("GAN training needs a discriminator and its optimizer")
    model.train()
    discriminator_metrics = 0.0
    if config.use_gan:
        _requires_grad(discriminator, True)
        discriminator.train()
        discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = model(batch.waveform, batch.semantic_targets).reconstruction
        real_scores, _ = discriminator(batch.waveform)
        fake_scores, _ = discriminator(fake)
        d_loss = discriminator_loss(real_scores, fake_scores)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_clip)
        discriminator_optimizer.step()
        discriminator_metrics = float(d_loss.detach())
        _requires_grad(discriminator, False)

    optimizer.zero_grad(set_to_none=True)
    output = model(batch.waveform, batch.semantic_targets)
    reconstruction = ReconstructionLoss()(batch.waveform, output.reconstruction)
    total = (
        config.commitment_weight * output.commitment_loss
        + config.semantic_weight * output.semantic_loss
    )
    adversarial = torch.zeros((), device=batch.waveform.device)
    feature_matching = torch.zeros((), device=batch.waveform.device)
    if config.use_gan:
        _, real_features = discriminator(batch.waveform)
        fake_scores, fake_features = discriminator(output.reconstruction)
        adversarial = generator_adversarial_loss(fake_scores)
        feature_matching = feature_matching_loss(real_features, fake_features)
        total = total + config.adversarial_weight * adversarial + config.feature_matching_weight * feature_matching
    if not config.adversarial_only:
        total = total + reconstruction.total
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    return {
        "loss": float(total.detach()),
        "reconstruction": float(reconstruction.total.detach()),
        "commitment": float(output.commitment_loss.detach()),
        "semantic": float(output.semantic_loss.detach()),
        "adversarial": float(adversarial.detach()),
        "feature_matching": float(feature_matching.detach()),
        "discriminator": discriminator_metrics,
        "grad_norm": float(grad_norm.detach()),
    }


def run_mimi_training(
    model: MimiModel,
    dataset: Dataset,
    config: MimiTrainingConfig,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    discriminator = MultiScaleDiscriminator().to(device) if config.use_gan else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    discriminator_optimizer = (
        torch.optim.AdamW(discriminator.parameters(), lr=config.discriminator_learning_rate)
        if discriminator is not None
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(0),
    )
    history = []
    step = 0
    while step < config.max_steps:
        for batch in loader:
            step += 1
            batch = MimiBatch(batch.waveform.to(device), batch.semantic_targets.to(device))
            metrics = mimi_training_step(
                model,
                batch,
                discriminator,
                optimizer,
                discriminator_optimizer,
                config,
            )
            metrics["step"] = step
            history.append(metrics)
            if step == 1 or step % config.log_every == 0:
                print(metrics)
            if step >= config.max_steps:
                break
    save_mimi_checkpoint(output_dir / "checkpoint-last.pt", model, config.max_steps)
    return history
