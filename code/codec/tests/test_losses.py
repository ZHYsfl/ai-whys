import torch

from teaching_codec.discriminator import (
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from teaching_codec.losses import MultiScaleSTFTLoss, ReconstructionLoss


def test_reconstruction_losses() -> None:
    x = torch.randn(2, 1, 800)
    stft = MultiScaleSTFTLoss()
    assert torch.allclose(stft(x, x), torch.tensor(0.0), atol=1e-7)
    assert stft(x, torch.zeros_like(x)) > 0
    result = ReconstructionLoss()(x, x)
    assert torch.allclose(result.total, torch.tensor(0.0), atol=1e-7)


def test_discriminator_data_flow() -> None:
    discriminator = MultiScaleDiscriminator(scales=(1, 2), base_channels=4)
    real = torch.randn(2, 1, 800)
    fake = torch.randn(2, 1, 800)
    real_scores, real_features = discriminator(real)
    fake_scores, fake_features = discriminator(fake)
    assert len(real_scores) == 2
    assert all(score.ndim == 3 and score.shape[1] == 1 for score in real_scores)
    assert torch.isfinite(discriminator_loss(real_scores, fake_scores))
    assert torch.isfinite(generator_adversarial_loss(fake_scores))
    assert torch.isfinite(feature_matching_loss(real_features, fake_features))


def test_hinge_losses_match_hand_computation() -> None:
    real_scores = [torch.ones(1, 1, 2)]
    fake_scores = [-torch.ones(1, 1, 2)]
    assert discriminator_loss(real_scores, fake_scores).item() == 0.0
    assert generator_adversarial_loss(fake_scores).item() == 2.0
