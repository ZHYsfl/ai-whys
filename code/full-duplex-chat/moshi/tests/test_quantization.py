import pytest
import torch

from teaching_moshi.quantization import SplitResidualVectorQuantizer


def make_quantizer():
    return SplitResidualVectorQuantizer(
        dimension=8,
        input_dimension=16,
        output_dimension=16,
        n_q=4,
        bins=8,
        q_dropout=False,
        no_quantization_rate=0.0,
    )


def test_split_rvq_shapes_codes_and_gradients():
    torch.manual_seed(0)
    quantizer = make_quantizer().train()
    x = torch.randn(2, 16, 5, requires_grad=True)
    output = quantizer(x, 12.5)
    assert output.x.shape == x.shape
    assert output.codes.shape == (2, 4, 5)
    assert output.semantic.shape == x.shape
    assert output.bandwidth.item() == pytest.approx(0.15)
    (output.x.square().mean() + output.penalty).backward()
    assert x.grad is not None
    assert quantizer.rvq_first.vq.layers[0]._codebook.initialized


def test_encode_decode_uses_semantic_plus_acoustic_codebooks():
    quantizer = make_quantizer().eval()
    x = torch.randn(2, 16, 5)
    with torch.no_grad():
        codes = quantizer.encode(x)
        decoded = quantizer.decode(codes)
    assert codes.shape == (2, 4, 5)
    assert decoded.shape == x.shape
