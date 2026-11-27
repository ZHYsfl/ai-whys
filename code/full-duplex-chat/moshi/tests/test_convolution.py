import pytest
import torch

from teaching_moshi.convolution import (
    StreamingConv1d,
    StreamingConvTranspose1d,
    pad_for_conv1d,
)


@pytest.mark.parametrize("layer_class", [StreamingConv1d, StreamingConvTranspose1d])
def test_streaming_convolution_matches_offline(layer_class):
    torch.manual_seed(0)
    layer = layer_class(4, 6, kernel_size=8, stride=4).eval()
    x = torch.randn(2, 4, 20)
    with torch.no_grad():
        expected = layer(x)
        outputs = []
        with layer.streaming(2):
            for chunk in x.split(4, dim=-1):
                outputs.append(layer(chunk))
        actual = torch.cat(outputs, dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_causal_convolution_does_not_read_future():
    layer = StreamingConv1d(1, 2, kernel_size=5).eval()
    x = torch.randn(1, 1, 8)
    changed = x.clone()
    changed[..., 6:] = 100
    with torch.no_grad():
        original = layer(x)
        modified = layer(changed)
    torch.testing.assert_close(original[..., :6], modified[..., :6])


def test_padding_completes_the_last_frame():
    x = torch.randn(2, 1, 2001)
    padded = pad_for_conv1d(x, 1920, 1920)
    assert padded.shape[-1] == 3840
