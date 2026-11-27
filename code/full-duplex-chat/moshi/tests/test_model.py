import torch

from teaching_moshi.data import SyntheticDialogueDataset, collate_dialogue
from teaching_moshi.losses import moshi_loss
from teaching_moshi.model import MoshiGenerator, MoshiLM


def test_full_multistream_forward_and_gradients():
    torch.manual_seed(0)
    model = MoshiLM().train()
    batch = collate_dialogue([SyntheticDialogueDataset()[0], SyntheticDialogueDataset()[1]])
    output = model(batch.codes)
    assert output.text_logits.shape == (2, 1, 8, 260)
    assert output.logits.shape == (2, 8, 8, 64)
    assert output.mask.shape == (2, 8, 8)
    assert not output.mask[:, 1:, -1].any()
    losses = moshi_loss(model, batch.codes, output)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert model.text_linear.weight.grad is not None
    assert model.transformer.layers[0].self_attn.in_projs[0].weight.grad is not None
    assert model.depformer.layers[0].self_attn.in_projs[1].weight.grad is not None
    assert model.linears[-1].weight.grad is not None


def test_depformer_streaming_matches_training():
    torch.manual_seed(1)
    model = MoshiLM().eval()
    sequence = torch.randint(0, model.card, (2, 17, 1))
    sequence[:, 0] %= model.text_card
    temporal = torch.randn(2, 1, model.dim)
    with torch.no_grad():
        expected = model.forward_depformer_training(sequence, temporal)
        actual = []
        with model.depformer.streaming(2):
            for index in range(model.dep_q):
                logits = model.forward_depformer(index, sequence[:, index : index + 1], temporal)
                actual.append(logits[:, 0])
        actual = torch.stack(actual, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_generator_outputs_valid_delayed_tokens():
    torch.manual_seed(2)
    model = MoshiLM().eval()
    generator = MoshiGenerator(model, use_sampling=False)
    user = torch.randint(0, model.card, (1, 8, 6))
    outputs = []
    with generator.streaming(1):
        for index in range(6):
            output = generator.step(user[:, :, index : index + 1])
            if output is not None:
                outputs.append(output)
    generated = torch.cat(outputs, dim=-1)
    assert generated.shape == (1, 9, 5)
    assert (generated[:, 0] >= 0).all()
    assert (generated[:, 0] < model.text_card).all()
    assert (generated[:, 1:] >= 0).all()
    assert (generated[:, 1:] < model.card).all()
