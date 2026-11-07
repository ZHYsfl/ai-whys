import argparse
import json

from teaching_asr.librispeech import extract_librispeech_subset, prepare_librispeech


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a tiny LibriSpeech train/val subset")
    parser.add_argument("--output-dir", default="data/librispeech-smoke")
    parser.add_argument("--archive-dir", default="data/downloads")
    parser.add_argument("--archive", help="reuse an existing verified archive instead of downloading")
    parser.add_argument("--source-split", default="dev-clean")
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if args.archive:
        metadata = extract_librispeech_subset(
            args.archive,
            args.output_dir,
            args.source_split,
            args.max_files,
            args.eval_fraction,
        )
    else:
        metadata = prepare_librispeech(
            args.output_dir,
            args.archive_dir,
            args.source_split,
            args.max_files,
            args.eval_fraction,
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
