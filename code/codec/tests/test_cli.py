import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from teaching_codec import TeachingCodec
from teaching_codec.audio_io import load_codes, save_wav
from teaching_codec.checkpoint import save_checkpoint
from teaching_codec.data import SyntheticAudioDataset


ROOT = Path(__file__).resolve().parents[1]


def test_wav_codes_wav_cli(small_config, tmp_path) -> None:
    model = TeachingCodec(small_config)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, step=0)
    waveform = SyntheticAudioDataset(size=1, sample_rate=8_000, num_samples=800)[0]
    input_wav = tmp_path / "input.wav"
    codes_path = tmp_path / "codes.npz"
    output_wav = tmp_path / "output.wav"
    save_wav(input_wav, waveform, 8_000)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "encode.py"),
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_wav),
            "--output",
            str(codes_path),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    codes, original_length = load_codes(codes_path)
    assert codes.shape == (1, 2, 100)
    assert original_length == 800

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "decode.py"),
            "--checkpoint",
            str(checkpoint),
            "--codes",
            str(codes_path),
            "--output",
            str(output_wav),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    reconstruction, sample_rate = sf.read(output_wav)
    assert sample_rate == 8_000
    assert reconstruction.shape == (800,)
    assert np.isfinite(reconstruction).all()
