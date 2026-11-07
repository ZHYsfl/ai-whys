import pytest

from teaching_asr import ASRConfig


@pytest.fixture
def small_config() -> ASRConfig:
    return ASRConfig(
        sample_rate=8_000,
        n_fft=200,
        win_length=200,
        hop_length=80,
        n_mels=16,
        d_model=16,
        num_heads=2,
        num_layers=1,
        ffn_dim=32,
        conv_kernel_size=5,
        dropout=0.0,
    )
