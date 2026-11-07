import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


DOWNLOAD_URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
}
MD5SUMS = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
}


def verify_archive(path: Path, split: str) -> None:
    digest = hashlib.md5()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != MD5SUMS[split]:
        raise RuntimeError(f"MD5 mismatch for {path}")


def download_archive(split: str, archive_dir: str | Path) -> Path:
    if split not in DOWNLOAD_URLS:
        raise ValueError(f"unsupported split {split}")
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / f"{split}.tar.gz"
    if destination.exists() and destination.stat().st_size > 0:
        verify_archive(destination, split)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        DOWNLOAD_URLS[split], headers={"User-Agent": "teaching-version-asr/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_report = 0.1
        with partial.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
                downloaded += len(chunk)
                if total and downloaded / total >= next_report:
                    print(f"download {split}: {downloaded / total:.0%}", flush=True)
                    next_report += 0.1
    if total and downloaded != total:
        raise RuntimeError(f"incomplete download: got {downloaded} of {total} bytes")
    verify_archive(partial, split)
    partial.replace(destination)
    return destination


def extract_librispeech_subset(
    archive_path: str | Path,
    output_dir: str | Path,
    source_split: str = "dev-clean",
    max_files: int = 64,
    eval_fraction: float = 0.25,
) -> dict:
    if max_files < 2:
        raise ValueError("max_files must be at least 2")
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    output_dir = Path(output_dir)
    metadata_path = output_dir / "prepare-metadata.json"
    requested = {
        "source_split": source_split,
        "max_files": max_files,
        "eval_fraction": eval_fraction,
    }
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if all(metadata.get(key) == value for key, value in requested.items()):
            return metadata
        raise FileExistsError(f"{output_dir} contains a different prepared subset")
    if any(output_dir.rglob("*.flac")) or any(output_dir.rglob("*.trans.txt")):
        raise FileExistsError(f"{output_dir} is not clean; choose another output directory")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        audio_members = []
        for member in members:
            parts = PurePosixPath(member.name).parts
            if (
                member.isfile()
                and len(parts) >= 4
                and parts[0] == "LibriSpeech"
                and parts[1] == source_split
                and parts[-1].endswith(".flac")
            ):
                audio_members.append(member)
                if len(audio_members) == max_files:
                    break
        if len(audio_members) < 2:
            raise RuntimeError(f"fewer than two FLAC files found for {source_split}")

        actual_count = len(audio_members)
        eval_count = min(max(1, int(actual_count * eval_fraction)), actual_count - 1)
        eval_indices = {int(index * actual_count / eval_count) for index in range(eval_count)}
        counts = {"train": 0, "val": 0}
        speakers = {"train": set(), "val": set()}
        output_root = output_dir.resolve()
        transcript_cache: dict[str, dict[str, str]] = {}
        for index, member in enumerate(audio_members):
            subset = "val" if index in eval_indices else "train"
            parts = PurePosixPath(member.name).parts
            speaker, chapter = parts[-3], parts[-2]
            transcript_member = f"LibriSpeech/{source_split}/{speaker}/{chapter}/{speaker}-{chapter}.trans.txt"
            if transcript_member not in transcript_cache:
                transcript_file = archive.extractfile(transcript_member)
                if transcript_file is None:
                    raise RuntimeError(f"missing transcript {transcript_member}")
                with transcript_file:
                    transcript_cache[transcript_member] = {
                        line.split(maxsplit=1)[0]: line
                        for raw_line in transcript_file
                        if (line := raw_line.decode("utf-8").strip())
                    }
            sample_id = PurePosixPath(member.name).stem
            transcript = transcript_cache[transcript_member].get(sample_id)
            if transcript is None:
                raise RuntimeError(f"transcript for {sample_id} not found")

            relative = Path(*parts[2:])
            target = output_dir / subset / relative
            if not target.resolve().is_relative_to(output_root):
                raise RuntimeError(f"unsafe archive path: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read {member.name}")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            transcript_path = target.with_name(f"{speaker}-{chapter}.trans.txt")
            with transcript_path.open("a", encoding="utf-8") as file:
                file.write(transcript + "\n")
            counts[subset] += 1
            speakers[subset].add(speaker)

    metadata = {
        **requested,
        "archive": str(archive_path),
        "train_dir": str(output_dir / "train"),
        "val_dir": str(output_dir / "val"),
        "train_files": counts["train"],
        "val_files": counts["val"],
        "train_speakers": len(speakers["train"]),
        "val_speakers": len(speakers["val"]),
        "split_policy": "deterministic utterance-level smoke split; not speaker-independent",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def prepare_librispeech(
    output_dir: str | Path,
    archive_dir: str | Path,
    source_split: str = "dev-clean",
    max_files: int = 64,
    eval_fraction: float = 0.25,
) -> dict:
    archive = download_archive(source_split, archive_dir)
    return extract_librispeech_subset(
        archive, output_dir, source_split, max_files, eval_fraction
    )
