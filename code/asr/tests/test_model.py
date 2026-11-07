import torch

from teaching_asr import ASRConfig, ConformerCTC
from teaching_asr.ctc import ctc_loss
from teaching_asr.features import spec_augment


def _targets(model: ConformerCTC) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [model.tokenizer.encode("aa"), model.tokenizer.encode("ab")]
    targets = torch.zeros(2, 2, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        targets[index, : len(sequence)] = torch.tensor(sequence)
    return targets, torch.tensor([2, 2])


def test_conformer_shapes_and_batch_independence(small_config: ASRConfig) -> None:
    model = ConformerCTC(small_config).eval()
    waveforms = torch.randn(2, 1600)
    lengths = torch.tensor([1600, 1200])
    with torch.no_grad():
        batched = model(waveforms, lengths)
        assert batched.logits.shape == (2, 3, model.tokenizer.vocab_size)
        assert batched.output_lengths.tolist() == [3, 2]
        for index, length in enumerate(lengths.tolist()):
            single = model(waveforms[index : index + 1, :length], lengths[index : index + 1])
            torch.testing.assert_close(
                batched.logits[index, : int(batched.output_lengths[index])],
                single.logits[0],
                atol=1e-5,
                rtol=1e-5,
            )


def test_conformer_ctc_gradients(small_config: ASRConfig) -> None:
    model = ConformerCTC(small_config)
    targets, target_lengths = _targets(model)
    output = model(torch.randn(2, 1600), torch.tensor([1600, 1200]))
    loss = ctc_loss(output.logits, targets, output.output_lengths, target_lengths)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_spec_augment_respects_lengths(small_config: ASRConfig) -> None:
    small_config.spec_augment = True
    small_config.time_masks = 2
    small_config.freq_masks = 2
    features = torch.ones(2, 18, 16)
    lengths = torch.tensor([18, 13])
    augmented = spec_augment(features, lengths, small_config)
    assert augmented.shape == features.shape
    assert torch.count_nonzero(augmented[1, 13:]) == 0
    assert torch.any(augmented != features)


def test_model_rejects_too_short_input(small_config: ASRConfig) -> None:
    model = ConformerCTC(small_config)
    try:
        model(torch.randn(1, 100), torch.tensor([100]))
    except ValueError as error:
        assert "4x" in str(error)
    else:
        raise AssertionError("expected a subsampling length error")
