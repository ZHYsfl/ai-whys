import io
import tarfile

import pytest

from teaching_asr.librispeech import extract_librispeech_subset


def test_librispeech_subset_keeps_audio_and_transcripts(tmp_path) -> None:
    archive_path = tmp_path / "dev-clean.tar.gz"
    transcript = "\n".join(f"84-121-{index:04d} WORD {index}" for index in range(8)) + "\n"
    with tarfile.open(archive_path, "w:gz") as archive:
        for index in range(8):
            content = f"fake-audio-{index}".encode()
            info = tarfile.TarInfo(f"LibriSpeech/dev-clean/84/121/84-121-{index:04d}.flac")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        content = transcript.encode()
        info = tarfile.TarInfo("LibriSpeech/dev-clean/84/121/84-121.trans.txt")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    output_dir = tmp_path / "subset"
    metadata = extract_librispeech_subset(archive_path, output_dir, max_files=8)
    assert metadata["train_files"] == 6
    assert metadata["val_files"] == 2
    assert len(list(output_dir.rglob("*.flac"))) == 8
    lines = []
    for path in output_dir.rglob("*.trans.txt"):
        lines.extend(path.read_text().splitlines())
    assert len(lines) == 8
    assert extract_librispeech_subset(archive_path, output_dir, max_files=8) == metadata
    with pytest.raises(FileExistsError):
        extract_librispeech_subset(archive_path, output_dir, max_files=4)
