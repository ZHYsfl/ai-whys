import soundfile as sf
import torch

from teaching_asr.data import LibriSpeechDataset, SyntheticSpeechDataset, collate_speech
from teaching_asr.metrics import edit_distance, error_counts, error_rate
from teaching_asr.text import CharTokenizer


def test_tokenizer_round_trip_and_validation() -> None:
    tokenizer = CharTokenizer()
    text = "  HELLO   WORLD'S "
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == "hello world's"
    assert tokenizer.blank_id not in ids
    assert tokenizer.vocab_size == 29
    try:
        tokenizer.encode("你好")
    except ValueError as error:
        assert "unsupported character" in str(error)
    else:
        raise AssertionError("expected unsupported character error")


def test_synthetic_dataset_and_variable_length_collate() -> None:
    dataset = SyntheticSpeechDataset(size=2, sample_rate=8_000)
    first = dataset[0]
    assert torch.equal(first.waveform, dataset[0].waveform)
    assert first.target.dtype == torch.long
    batch = collate_speech([dataset[0], dataset[1]])
    assert batch.waveforms.shape[0] == batch.targets.shape[0] == 2
    assert batch.waveform_lengths.tolist() == [first.waveform.shape[0], dataset[1].waveform.shape[0]]
    for index, transcript in enumerate(batch.transcripts):
        length = int(batch.target_lengths[index])
        assert CharTokenizer().decode(batch.targets[index, :length]) == transcript


def test_librispeech_dataset_reads_transcripts_and_downmixes(tmp_path) -> None:
    sample_id = "84-121-0000"
    transcript_path = tmp_path / "84-121.trans.txt"
    transcript_path.write_text(f"{sample_id} HELLO WORLD\n")
    time = torch.arange(800, dtype=torch.float32) / 8_000
    stereo = torch.stack((torch.sin(2 * torch.pi * 220 * time),) * 2, dim=1)
    sf.write(tmp_path / f"{sample_id}.flac", stereo.numpy(), 8_000)
    dataset = LibriSpeechDataset(tmp_path, sample_rate=8_000)
    example = dataset[0]
    assert example.transcript == "hello world"
    assert example.sample_id == sample_id
    assert example.waveform.shape == (800,)
    torch.testing.assert_close(
        example.waveform, torch.sin(2 * torch.pi * 220 * time), atol=1e-4, rtol=1e-4
    )


def test_edit_distance_and_corpus_error_rates() -> None:
    assert edit_distance(list("abc"), list("axc")) == 1
    assert edit_distance("a b".split(), "a b c".split()) == 1
    references = ["a b", "c"]
    hypotheses = ["a x", "c"]
    assert error_counts(references, hypotheses, "word") == (1, 3)
    assert error_rate(references, hypotheses, "char") == 1 / 3
    assert error_rate(["a"], ["a b"], "word") == 1.0
