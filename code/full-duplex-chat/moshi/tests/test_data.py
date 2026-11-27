import json

import numpy as np
import soundfile as sf
import torch

from teaching_moshi.config import MoshiConfig
from teaching_moshi.data import (
    Interleaver,
    StereoDialogueDataset,
    SyntheticDialogueDataset,
    collate_dialogue,
)
from teaching_moshi.mimi import MimiModel
from teaching_moshi.text import ByteTokenizer


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "Moshi 你好"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_interleaver_places_text_and_epad():
    tokenizer = ByteTokenizer()
    interleaver = Interleaver(tokenizer, 2.0, 3, 0)
    stream = interleaver.build_token_stream(
        [("ab", (1.0, 1.5), "SPEAKER_MAIN"), ("x", (0.0, 0.5), "SPEAKER_OTHER")],
        3.0,
    )
    assert stream.tolist() == [3, 0, 101, 102, 3, 3]


def test_synthetic_batch_has_all_17_streams_and_padding():
    dataset = SyntheticDialogueDataset(size=2, frames=8)
    first = dataset[0]
    assert first.codes.shape == (17, 8)
    assert torch.equal(first.codes, dataset[0].codes)
    batch = collate_dialogue([first, dataset[1]])
    assert batch.codes.shape == (2, 17, 8)


def test_stereo_dataset_uses_left_moshi_right_user(tmp_path):
    sample_rate = 24_000
    samples = np.zeros((1920, 2), dtype=np.float32)
    samples[:, 0] = 0.5
    samples[:, 1] = -0.25
    audio_path = tmp_path / "dialogue.wav"
    sf.write(audio_path, samples, sample_rate)
    audio_path.with_suffix(".json").write_text(
        json.dumps({"alignments": [["a", [0.0, 0.08], "SPEAKER_MAIN"]]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(
        json.dumps({"path": "dialogue.wav", "duration": 0.08}) + "\n",
        encoding="utf-8",
    )
    mimi = MimiModel().eval()
    dataset = StereoDialogueDataset(manifest, mimi, ByteTokenizer(), MoshiConfig())
    sample = dataset[0]
    assert sample.codes.shape == (17, 1)
    assert sample.codes[0, 0] == 101
    assert (sample.codes[1:] == 0).all()
