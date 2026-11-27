import torch

from teaching_moshi.config import MimiConfig
from teaching_moshi.mimi import MimiModel


def test_forward_backward_with_semantic_distillation():
    torch.manual_seed(0)
    config = MimiConfig(quantizer_dropout=False, no_quantization_rate=0.0)
    model = MimiModel(config).train()
    waveform = torch.randn(2, 1, 2 * config.frame_size, requires_grad=True)
    targets = torch.randn(2, config.semantic_dimension, 2)
    output = model(waveform, targets)
    assert output.reconstruction.shape == waveform.shape
    assert output.codes.shape == (2, 8, 2)
    assert output.semantic_prediction.shape == targets.shape
    loss = output.reconstruction.square().mean() + output.commitment_loss + output.semantic_loss
    loss.backward()
    assert waveform.grad is not None
    assert model.encoder.model[0].conv.conv.weight.grad is not None
    assert model.semantic_projection.weight.grad is not None


def test_streaming_matches_offline_reconstruction():
    torch.manual_seed(2)
    config = MimiConfig(quantizer_dropout=False, no_quantization_rate=0.0)
    model = MimiModel(config)
    waveform = torch.randn(1, 1, 2 * config.frame_size).clamp(-1, 1)
    model.train()
    model(waveform)
    model.eval()
    with torch.no_grad():
        latent = model.encode_to_latent(waveform, quantize=False)
        expected = model.decode(model.encode(waveform))
        audio_chunks = []
        with model.streaming(1):
            for frame in waveform.split(config.frame_size, dim=-1):
                audio_chunks.append(model.decode(model.encode(frame)))
        actual = torch.cat(audio_chunks, dim=-1)
        streamed_latent = []
        with model.streaming(1):
            for frame in waveform.split(config.frame_size, dim=-1):
                streamed_latent.append(model.encode_to_latent(frame, quantize=False))
        streamed_latent = torch.cat(streamed_latent, dim=-1)
    torch.testing.assert_close(streamed_latent, latent, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_encode_pads_and_decode_can_restore_original_length():
    model = MimiModel().eval()
    waveform = torch.randn(1, 1, 2000)
    with torch.no_grad():
        codes = model.encode(waveform)
        reconstruction = model.decode(codes, original_length=2000)
    assert codes.shape[-1] == 2
    assert reconstruction.shape == waveform.shape
