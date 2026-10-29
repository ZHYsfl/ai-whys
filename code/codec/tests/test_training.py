import math

import torch

from teaching_codec import CodecConfig, TeachingCodec
from teaching_codec.checkpoint import load_checkpoint, save_checkpoint
from teaching_codec.data import SyntheticAudioDataset
from teaching_codec.discriminator import MultiScaleDiscriminator
from teaching_codec.losses import ReconstructionLoss
from teaching_codec.training import training_step


def _batch(config: CodecConfig):
    dataset = SyntheticAudioDataset(size=2, sample_rate=config.sample_rate, num_samples=800)
    return torch.stack([dataset[0], dataset[1]])


def test_reconstruction_training_step_and_checkpoint(small_config: CodecConfig, tmp_path) -> None:
    model = TeachingCodec(small_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _batch(small_config)
    metrics = training_step(
        model,
        None,
        batch,
        optimizer,
        None,
        ReconstructionLoss(),
    )
    assert all(math.isfinite(value) for value in metrics.values())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, step=1)
    restored, discriminator, payload = load_checkpoint(path)
    assert discriminator is None
    assert payload["step"] == 1
    model.eval()
    with torch.no_grad():
        expected = model(batch)
        actual = restored(batch)
    assert torch.allclose(expected.reconstruction, actual.reconstruction)


def test_gan_training_step(small_config: CodecConfig, tmp_path) -> None:
    model = TeachingCodec(small_config)
    discriminator = MultiScaleDiscriminator(scales=(1, 2), base_channels=4)
    generator_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    generator_before = [parameter.detach().clone() for parameter in model.parameters()]
    discriminator_before = [parameter.detach().clone() for parameter in discriminator.parameters()]
    metrics = training_step(
        model,
        discriminator,
        _batch(small_config),
        generator_optimizer,
        discriminator_optimizer,
        ReconstructionLoss(),
        quantizer_dropout=True,
        dropout_rate=0.5,
    )
    assert metrics["discriminator"] > 0
    assert metrics["adversarial"] > 0
    assert metrics["feature_matching"] > 0
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(generator_before, model.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(discriminator_before, discriminator.parameters())
    )

    path = tmp_path / "gan-checkpoint.pt"
    save_checkpoint(path, model, step=1, discriminator=discriminator)
    _, restored_discriminator, _ = load_checkpoint(path)
    assert restored_discriminator is not None
    assert restored_discriminator.scales == (1, 2)
    for expected, actual in zip(discriminator.parameters(), restored_discriminator.parameters()):
        assert torch.allclose(expected, actual)
