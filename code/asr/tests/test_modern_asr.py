import torch

from teaching_asr.aut import AuT
from teaching_asr.data import SyntheticSpeechDataset, collate_speech
from teaching_asr.llm_checkpoint import load_llm_checkpoint, save_llm_checkpoint
from teaching_asr.llm_text import LanguageTokenizer, collate_language_targets
from teaching_asr.llm_training import evaluate_llm
from teaching_asr.modern_config import ModernASRConfig
from teaching_asr.qwen3_asr import Qwen3ASR
from teaching_asr.qwen3_lm import Qwen3LM


def _batch():
    dataset = SyntheticSpeechDataset(size=2)
    return collate_speech([dataset[0], dataset[1]])


def test_modern_config_and_language_targets() -> None:
    config = ModernASRConfig()
    restored = ModernASRConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.audio.output_frame_rate == 12.5
    assert restored.audio.downsampled_mels == 16
    tokenizer = LanguageTokenizer()
    targets, lengths = collate_language_targets(_batch(), tokenizer)
    assert targets.shape == (2, 11)
    assert lengths.tolist() == [11, 8]
    assert targets[0, 0] == tokenizer.bos_id
    assert targets[0, 10] == tokenizer.eos_id
    assert tokenizer.decode([7, 8, tokenizer.eos_id, 9]) == "ab"


def test_qwen3_causal_attention_and_gqa_cache() -> None:
    torch.manual_seed(0)
    config = ModernASRConfig().llm
    model = Qwen3LM(config).eval()
    tokens = torch.tensor([[1, 5, 6, 7, 8]])
    with torch.no_grad():
        full = model(tokens).logits
        changed_tokens = tokens.clone()
        changed_tokens[0, 3] = 9
        changed = model(changed_tokens).logits
        torch.testing.assert_close(full[:, :3], changed[:, :3], atol=1e-6, rtol=1e-6)

        prefix = model(tokens[:, :3], use_cache=True)
        suffix = model(
            tokens[:, 3:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        torch.testing.assert_close(full[:, 3:], suffix.logits, atol=1e-5, rtol=1e-5)
        key, value = suffix.past_key_values[0]
        assert key.shape == (1, config.num_key_value_heads, 5, config.head_dim)
        assert value.shape == key.shape
        assert config.num_attention_heads > config.num_key_value_heads


def test_qwen3_asr_forward_gradients_and_generation() -> None:
    torch.manual_seed(1)
    model = Qwen3ASR(ModernASRConfig())
    batch = _batch()
    speech, speech_lengths = model.speech_embeddings(batch.waveforms, batch.waveform_lengths)
    assert speech.shape == (2, 18, 64)
    assert speech_lengths.tolist() == [18, 13]
    _, attention_mask, _ = model._build_inputs(speech, speech_lengths, batch.transcripts)
    expected_lengths = [
        len(model.tokenizer.prompt_ids)
        + int(length)
        + len(model.tokenizer.encode(transcript))
        + 1
        for length, transcript in zip(speech_lengths, batch.transcripts, strict=True)
    ]
    assert attention_mask.sum(dim=1).tolist() == expected_lengths
    output = model(batch.waveforms, batch.waveform_lengths, batch.transcripts)
    assert output.logits.shape == (2, 31, 33)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    model.eval()
    with torch.no_grad():
        full, full_lengths, _ = model.audio_encoder(
            batch.waveforms, batch.waveform_lengths, attention_window=0
        )
        local, local_lengths, _ = model.audio_encoder(
            batch.waveforms, batch.waveform_lengths, attention_window=1
        )
        assert full.shape == local.shape == speech.shape
        assert full_lengths.tolist() == local_lengths.tolist() == speech_lengths.tolist()
        assert torch.isfinite(full).all() and torch.isfinite(local).all()
    texts = model.transcribe(batch.waveforms, batch.waveform_lengths, max_new_tokens=4)
    assert len(texts) == 2 and all(isinstance(text, str) for text in texts)


def test_qwen3_batch_generation_matches_single() -> None:
    torch.manual_seed(7)
    model = Qwen3ASR(ModernASRConfig()).eval()
    batch = _batch()
    actual = model.transcribe(batch.waveforms, batch.waveform_lengths, max_new_tokens=16)
    expected = [
        model.transcribe(
            batch.waveforms[index : index + 1, : batch.waveform_lengths[index]],
            batch.waveform_lengths[index : index + 1],
            max_new_tokens=16,
        )[0]
        for index in range(2)
    ]
    assert actual == expected


def test_aut_encoder_batch_matches_single() -> None:
    torch.manual_seed(7)
    model = Qwen3ASR(ModernASRConfig()).eval()
    dataset = SyntheticSpeechDataset(size=8, seed=10_000)
    batch = collate_speech([dataset[index] for index in range(len(dataset))])
    with torch.no_grad():
        encoded, lengths, _ = model.audio_encoder(batch.waveforms, batch.waveform_lengths)
        for index in range(len(dataset)):
            single_batch = collate_speech([dataset[index]])
            single, single_lengths, _ = model.audio_encoder(
                single_batch.waveforms, single_batch.waveform_lengths
            )
            assert int(lengths[index]) == int(single_lengths[0])
            torch.testing.assert_close(
                encoded[index, : lengths[index]], single[0], atol=1e-5, rtol=1e-5
            )


def test_llm_evaluation_loss_is_batch_size_invariant() -> None:
    torch.manual_seed(7)
    model = Qwen3ASR(ModernASRConfig()).eval()
    dataset = SyntheticSpeechDataset(size=8, seed=10_000)
    batched = evaluate_llm(model, dataset, batch_size=4, generate=False)
    single = evaluate_llm(model, dataset, batch_size=1, generate=False)
    torch.testing.assert_close(
        torch.tensor(batched["loss"]), torch.tensor(single["loss"]), atol=1e-6, rtol=1e-6
    )


def test_qwen3_checkpoint_restores_generation(tmp_path) -> None:
    model = Qwen3ASR(ModernASRConfig()).eval()
    batch = _batch()
    expected = model.transcribe(batch.waveforms, batch.waveform_lengths, max_new_tokens=4)
    path = tmp_path / "qwen3-asr.pt"
    save_llm_checkpoint(path, model, step=7)
    restored, payload = load_llm_checkpoint(path)
    assert payload["step"] == 7
    actual = restored.transcribe(batch.waveforms, batch.waveform_lengths, max_new_tokens=4)
    assert actual == expected


def test_aut_pretraining_and_encoder_transfer() -> None:
    torch.manual_seed(2)
    config = ModernASRConfig()
    aut = AuT(config.audio, config.llm)
    batch = _batch()
    tokenizer = LanguageTokenizer()
    targets, lengths = collate_language_targets(batch, tokenizer)
    output = aut(batch.waveforms, batch.waveform_lengths, targets, lengths)
    assert output.logits.shape == (2, 10, 33)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in aut.parameters()
    )

    modern = Qwen3ASR(config)
    modern.audio_encoder.load_state_dict(aut.audio_encoder.state_dict())
    for expected, actual in zip(
        aut.audio_encoder.parameters(), modern.audio_encoder.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
