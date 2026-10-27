import math

import pytest
import torch

from teaching_codec import CodecConfig, TeachingCodec


def test_encode_decode_matches_forward(small_config: CodecConfig) -> None:
    model = TeachingCodec(small_config).eval()
    x = torch.randn(2, 1, 800)
    with torch.no_grad():
        forward = model(x)
        codes = model.encode(x)
        decoded = model.decode(codes)
    assert torch.equal(codes, forward.codes)
    torch.testing.assert_close(decoded, forward.reconstruction, atol=1e-6, rtol=1e-5)

    with torch.no_grad():
        forward_prefix = model(x, n_quantizers=1)
        prefix_codes = model.encode(x, n_quantizers=1)
        prefix_decoded = model.decode(prefix_codes)
    assert prefix_codes.shape == (2, 1, 100)
    assert torch.equal(prefix_codes, forward_prefix.codes)
    torch.testing.assert_close(prefix_decoded, forward_prefix.reconstruction, atol=1e-6, rtol=1e-5)


def test_bitrate_uses_log2_codebook_size() -> None:
    config = CodecConfig()
    assert config.hop_length == 128
    assert config.frame_rate == 125
    assert config.bitrate() == 3_000
    assert config.bitrate(1) == 750
    custom = CodecConfig(codebook_size=100)
    assert custom.bitrate(1) == 125 * math.log2(100)


def test_config_round_trip() -> None:
    config = CodecConfig(strides=(2, 2, 4), codebook_size=100)
    restored = CodecConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.strides == (2, 2, 4)


def test_unquantized_autoencoder(small_config: CodecConfig) -> None:
    small_config.use_quantization = False
    model = TeachingCodec(small_config)
    output = model(torch.randn(1, 1, 800))
    assert output.codes.shape == (1, 0, 100)
    assert output.codebook_loss.item() == 0
    assert output.commitment_loss.item() == 0
    with pytest.raises(RuntimeError, match="use_quantization"):
        model.encode(torch.randn(1, 1, 800))
