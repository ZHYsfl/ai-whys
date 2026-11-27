import pytest
import torch

from teaching_moshi.transformer import StreamingTransformer


@pytest.mark.parametrize("position", ["none", "rope"])
@pytest.mark.parametrize("chunk_size", [1, 2])
def test_streaming_matches_offline(position, chunk_size):
    torch.manual_seed(0)
    model = StreamingTransformer(
        d_model=16,
        num_heads=4,
        num_layers=2,
        dim_feedforward=64,
        context=4,
        positional_embedding=position,
    ).eval()
    x = torch.randn(2, 9, 16)
    with torch.no_grad():
        expected = model(x)
        chunks = []
        with model.streaming(2):
            for chunk in x.split(chunk_size, dim=1):
                chunks.append(model(chunk))
        actual = torch.cat(chunks, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_future_tokens_do_not_change_past():
    torch.manual_seed(1)
    model = StreamingTransformer(
        d_model=16, num_heads=4, num_layers=1, dim_feedforward=32, context=16
    ).eval()
    x = torch.randn(1, 5, 16)
    changed = x.clone()
    changed[:, 3:] += 100
    with torch.no_grad():
        original = model(x)
        modified = model(changed)
    torch.testing.assert_close(original[:, :3], modified[:, :3])
