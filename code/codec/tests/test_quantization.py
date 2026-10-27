import pytest
import torch

from teaching_codec.quantization import ResidualVectorQuantizer, VectorQuantizer


def test_nearest_codebook_lookup() -> None:
    quantizer = VectorQuantizer(codebook_size=3, dimension=2)
    with torch.no_grad():
        quantizer.codebook.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    x = torch.tensor([[[0.8, 0.1], [0.4, 0.9]]])
    result = quantizer(x)
    assert result.codes.tolist() == [[0, 1]]
    expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert torch.allclose(result.quantized, expected)
    assert result.codebook_loss.shape == (1,)
    assert result.commitment_loss.shape == (1,)
    assert torch.allclose(quantizer.decode(result.codes), result.quantized)


def test_vq_losses_have_separate_gradient_targets() -> None:
    quantizer = VectorQuantizer(codebook_size=4, dimension=2)
    x = torch.randn(2, 2, 3, requires_grad=True)
    result = quantizer(x)
    (result.codebook_loss.sum() + result.commitment_loss.sum()).backward()
    assert x.grad is not None
    assert quantizer.codebook.grad is not None
    assert torch.count_nonzero(quantizer.codebook.grad) > 0


def test_rvq_decode_and_straight_through_gradient() -> None:
    quantizer = ResidualVectorQuantizer(codebook_size=4, dimension=2, num_codebooks=3)
    x = torch.randn(2, 2, 5, requires_grad=True)
    result = quantizer(x)
    assert result.codes.shape == (2, 3, 5)
    assert torch.allclose(quantizer.decode(result.codes), result.quantized)
    result.quantized.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_quantizer_dropout_is_per_sample(monkeypatch) -> None:
    quantizer = ResidualVectorQuantizer(codebook_size=4, dimension=2, num_codebooks=3)
    quantizer.train()
    x = torch.randn(2, 2, 5)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: torch.tensor([1, 2], device=kwargs.get("device")),
    )
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: torch.zeros(args[0], device=kwargs.get("device")),
    )
    dropped = quantizer(x, quantizer_dropout=True)
    two_layers = quantizer(x, n_quantizers=2)
    one_layer_first_sample = quantizer(x[:1], n_quantizers=1)
    expected = two_layers.quantized.clone()
    expected[0] = one_layer_first_sample.quantized[0]
    assert torch.allclose(dropped.quantized, expected)
    assert torch.equal(dropped.codes[0, 1:], torch.full((2, 5), -1))
    assert torch.equal(dropped.codes[1, 2], torch.full((5,), -1))


def test_dropout_rate_zero_keeps_all_layers() -> None:
    quantizer = ResidualVectorQuantizer(codebook_size=4, dimension=2, num_codebooks=3)
    quantizer.train()
    x = torch.randn(2, 2, 5)
    result = quantizer(x, quantizer_dropout=True, dropout_rate=0.0)
    expected = quantizer(x)
    assert torch.allclose(result.quantized, expected.quantized)


def test_dropout_codes_cannot_be_decoded_as_valid_prefix(monkeypatch) -> None:
    quantizer = ResidualVectorQuantizer(codebook_size=4, dimension=2, num_codebooks=2)
    quantizer.train()
    x = torch.randn(1, 2, 5)
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.ones(1, dtype=torch.long))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(1))
    result = quantizer(x, quantizer_dropout=True)
    assert torch.equal(result.codes[:, 1], torch.full((1, 5), -1))
    with pytest.raises(ValueError, match="codes must be in"):
        quantizer.decode(result.codes)


def test_decode_accepts_int32_and_checks_range() -> None:
    quantizer = VectorQuantizer(codebook_size=3, dimension=2)
    codes = torch.tensor([[0, 2]], dtype=torch.int32)
    assert quantizer.decode(codes).shape == (1, 2, 2)
    with pytest.raises(ValueError, match="codes must be in"):
        quantizer.decode(torch.tensor([[3]], dtype=torch.int32))
