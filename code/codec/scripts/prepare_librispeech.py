import argparse
import json

from teaching_codec.librispeech import prepare_librispeech


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LibriSpeech and prepare a small train/val subset")
    parser.add_argument("--split", default="dev-clean")
    parser.add_argument("--output-dir", default="data/librispeech-smoke")
    parser.add_argument("--archive-dir", default="data/downloads")
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    args = parser.parse_args()
    metadata = prepare_librispeech(
        args.output_dir,
        args.archive_dir,
        source_split=args.split,
        max_files=args.max_files,
        eval_fraction=args.eval_fraction,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
