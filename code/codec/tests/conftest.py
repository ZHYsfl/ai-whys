import pytest

from teaching_codec import CodecConfig


@pytest.fixture
def small_config() -> CodecConfig:
    return CodecConfig(
        sample_rate=8_000,
        latent_dim=8,
        base_channels=4,
        strides=(2, 4),
        codebook_size=8,
        num_codebooks=2,
    )
