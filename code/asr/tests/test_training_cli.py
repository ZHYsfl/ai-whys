import json
import math
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

from teaching_asr import ASRConfig, ConformerCTC
from teaching_asr.checkpoint import load_checkpoint, save_checkpoint
from teaching_asr.ctc import beam_search_decode
from teaching_asr.data import SyntheticSpeechDataset, collate_speech
from teaching_asr.training import training_step


ROOT = Path(__file__).resolve().parents[1]


def test_training_step_and_checkpoint_round_trip(small_config: ASRConfig, tmp_path) -> None:
    model = ConformerCTC(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = SyntheticSpeechDataset(size=2, sample_rate=8_000)
    batch = collate_speech([dataset[0], dataset[1]])
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = training_step(model, batch, optimizer)
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(previous, parameter)
        for previous, parameter in zip(before, model.parameters(), strict=True)
    )

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, step=1)
    restored, payload = load_checkpoint(path)
    assert payload["step"] == 1
    assert restored.config == model.config
    assert restored.tokenizer.tokens == model.tokenizer.tokens
    model.eval()
    with torch.no_grad():
        expected = model(batch.waveforms, batch.waveform_lengths)
        actual = restored(batch.waveforms, batch.waveform_lengths)
    torch.testing.assert_close(actual.logits, expected.logits)


def test_independent_transcribe_cli(small_config: ASRConfig, tmp_path) -> None:
    model = ConformerCTC(small_config)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, step=0)
    waveform = SyntheticSpeechDataset(size=1, sample_rate=8_000)[0].waveform
    input_path = tmp_path / "input.flac"
    output_path = tmp_path / "transcript.json"
    sf.write(input_path, waveform.numpy(), 8_000)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "transcribe.py"),
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output_path.read_text())
    assert result["checkpoint_step"] == 0
    assert result["output_frames"] > 0
    assert isinstance(result["text"], str)

    with torch.no_grad():
        output = model(waveform[None], torch.tensor([waveform.shape[0]]))
        expected = beam_search_decode(output.logits, output.output_lengths, model.tokenizer)[0]
    assert result["text"] == expected
