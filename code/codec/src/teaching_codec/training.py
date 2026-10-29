import torch

from .codec import TeachingCodec
from .discriminator import (
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from .losses import ReconstructionLoss


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def discriminator_step(
    model: TeachingCodec,
    discriminator: MultiScaleDiscriminator,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    quantizer_dropout: bool,
    dropout_rate: float,
    grad_clip: float,
) -> float:
    _set_requires_grad(discriminator, True)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake = model(
            batch,
            quantizer_dropout=quantizer_dropout,
            dropout_rate=dropout_rate,
        ).reconstruction
    real_scores, _ = discriminator(batch)
    fake_scores, _ = discriminator(fake)
    loss = discriminator_loss(real_scores, fake_scores)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach())


def generator_step(
    model: TeachingCodec,
    discriminator: MultiScaleDiscriminator | None,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    reconstruction_loss: ReconstructionLoss,
    quantizer_dropout: bool,
    dropout_rate: float,
    adversarial_weight: float,
    feature_matching_weight: float,
    grad_clip: float,
) -> dict[str, float]:
    use_gan = discriminator is not None
    if use_gan:
        _set_requires_grad(discriminator, False)
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch,
        quantizer_dropout=quantizer_dropout,
        dropout_rate=dropout_rate,
    )
    reconstruction = reconstruction_loss(batch, output.reconstruction)
    total = (
        reconstruction.total
        + model.config.codebook_weight * output.codebook_loss
        + model.config.commitment_weight * output.commitment_loss
    )
    adversarial = torch.zeros((), device=batch.device)
    feature_matching = torch.zeros((), device=batch.device)
    if discriminator is not None:
        _, real_features = discriminator(batch)
        fake_scores, fake_features = discriminator(output.reconstruction)
        adversarial = generator_adversarial_loss(fake_scores)
        feature_matching = feature_matching_loss(real_features, fake_features)
        total = (
            total
            + adversarial_weight * adversarial
            + feature_matching_weight * feature_matching
        )
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return {
        "total": float(total.detach()),
        "waveform": float(reconstruction.waveform.detach()),
        "stft": float(reconstruction.stft.detach()),
        "codebook": float(output.codebook_loss.detach()),
        "commitment": float(output.commitment_loss.detach()),
        "adversarial": float(adversarial.detach()),
        "feature_matching": float(feature_matching.detach()),
    }


def training_step(
    model: TeachingCodec,
    discriminator: MultiScaleDiscriminator | None,
    batch: torch.Tensor,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer | None,
    reconstruction_loss: ReconstructionLoss,
    quantizer_dropout: bool = False,
    dropout_rate: float = 1.0,
    adversarial_weight: float = 1.0,
    feature_matching_weight: float = 2.0,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    metrics = {"discriminator": 0.0}
    if discriminator is not None:
        if discriminator_optimizer is None:
            raise ValueError("discriminator_optimizer is required when GAN is enabled")
        discriminator.train()
        metrics["discriminator"] = discriminator_step(
            model,
            discriminator,
            batch,
            discriminator_optimizer,
            quantizer_dropout,
            dropout_rate,
            grad_clip,
        )
    metrics.update(
        generator_step(
            model,
            discriminator,
            batch,
            generator_optimizer,
            reconstruction_loss,
            quantizer_dropout,
            dropout_rate,
            adversarial_weight,
            feature_matching_weight,
            grad_clip,
        )
    )
    return metrics
