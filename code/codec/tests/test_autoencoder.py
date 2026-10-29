import pytest
import torch

from teaching_codec import CodecConfig, TeachingCodec
from teaching_codec.autoencoder import Decoder, Encoder


def test_encoder_decoder_shapes(small_config: CodecConfig) -> None:
    encoder = Encoder(small_config)
    decoder = Decoder(small_config)
    x = torch.randn(2, 1, 800)
    z = encoder(x)
    assert z.shape == (2, small_config.latent_dim, 100)
    assert decoder(z).shape == x.shape


def test_forward_requires_hop_aligned_input(small_config: CodecConfig) -> None:
    model = TeachingCodec(small_config)
    with pytest.raises(ValueError, match="multiple"):
        model(torch.randn(1, 1, 803))


def test_reconstruct_pads_and_trims(small_config: CodecConfig) -> None:
    model = TeachingCodec(small_config).eval()
    x = torch.randn(2, 1, 803)
    codes = model.encode(x)
    assert codes.shape == (2, 2, 101)
    assert model.decode(codes).shape == (2, 1, 808)
    assert model.reconstruct(x).shape == x.shape


def test_teaching_strides_must_be_even() -> None:
    with pytest.raises(ValueError, match="even"):
        CodecConfig(strides=(2, 5))
