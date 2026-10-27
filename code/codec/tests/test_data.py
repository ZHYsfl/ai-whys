import io
import tarfile

import numpy as np
import pytest
import soundfile as sf

from teaching_codec.data import WavFolderDataset
from teaching_codec.librispeech import extract_librispeech_subset


def test_flac_folder_dataset(tmp_path) -> None:
    time = np.arange(800, dtype=np.float32) / 8_000
    waveform = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    sf.write(tmp_path / "example.flac", waveform, 8_000, format="FLAC", subtype="PCM_24")
    dataset = WavFolderDataset(tmp_path, sample_rate=8_000, num_samples=800)
    assert len(dataset) == 1
    assert dataset[0].shape == (1, 800)
    np.testing.assert_allclose(dataset[0].numpy()[0], waveform, atol=1e-5)


def test_librispeech_subset_extraction_and_split(tmp_path) -> None:
    archive_path = tmp_path / "dev-clean.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for index in range(8):
            content = f"fake-audio-{index}".encode()
            info = tarfile.TarInfo(f"LibriSpeech/dev-clean/84/121/{index:04d}.flac")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    output_dir = tmp_path / "subset"
    metadata = extract_librispeech_subset(
        archive_path,
        output_dir,
        max_files=8,
        eval_fraction=0.25,
    )
    assert metadata["train_files"] == 6
    assert metadata["val_files"] == 2
    assert len(list((output_dir / "train").rglob("*.flac"))) == 6
    assert len(list((output_dir / "val").rglob("*.flac"))) == 2
    reused = extract_librispeech_subset(
        archive_path,
        output_dir,
        max_files=8,
        eval_fraction=0.25,
    )
    assert reused == metadata
    with pytest.raises(FileExistsError):
        extract_librispeech_subset(
            archive_path,
            output_dir,
            max_files=4,
            eval_fraction=0.25,
        )
