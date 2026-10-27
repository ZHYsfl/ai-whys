import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


DOWNLOAD_URLS = {
    "dev-clean": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "http://www.openslr.org/resources/12/test-clean.tar.gz",
    "train-clean-100": "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
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
    expected = MD5SUMS[split]
    if digest.hexdigest() != expected:
        raise RuntimeError(f"MD5 mismatch for {path}: expected {expected}, got {digest.hexdigest()}")


def download_archive(split: str, archive_dir: str | Path) -> Path:
    if split not in DOWNLOAD_URLS:
        raise ValueError(f"unsupported split {split}; choose from {sorted(DOWNLOAD_URLS)}")
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / f"{split}.tar.gz"
    if destination.exists() and destination.stat().st_size > 0:
        verify_archive(destination, split)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        DOWNLOAD_URLS[split], headers={"User-Agent": "teaching-version-codec/0.1"}
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
        raise FileExistsError(
            f"{output_dir} already contains a different prepared subset; choose another output directory"
        )
    if any(
        path.is_file() and path.suffix.lower() in {".wav", ".flac"}
        for path in output_dir.rglob("*")
    ):
        raise FileExistsError(f"audio files already exist under {output_dir}; choose a clean directory")

    members = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if (
                member.isfile()
                and len(parts) >= 4
                and parts[0] == "LibriSpeech"
                and parts[1] == source_split
                and parts[-1].endswith(".flac")
            ):
                members.append(member)
                if len(members) == max_files:
                    break
        if len(members) < 2:
            raise RuntimeError(f"fewer than two FLAC files found for {source_split} in {archive_path}")

        actual_count = len(members)
        eval_count = max(1, int(actual_count * eval_fraction))
        eval_count = min(eval_count, actual_count - 1)
        eval_indices = {int(index * actual_count / eval_count) for index in range(eval_count)}
        counts = {"train": 0, "val": 0}
        output_root = output_dir.resolve()
        for index, member in enumerate(members):
            subset = "val" if index in eval_indices else "train"
            parts = PurePosixPath(member.name).parts
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
            counts[subset] += 1

    metadata = {
        **requested,
        "archive": str(archive_path),
        "train_dir": str(output_dir / "train"),
        "val_dir": str(output_dir / "val"),
        "train_files": counts["train"],
        "val_files": counts["val"],
        "split_policy": "deterministic utterance-level split; not speaker-independent",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
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
        archive,
        output_dir,
        source_split=source_split,
        max_files=max_files,
        eval_fraction=eval_fraction,
    )
