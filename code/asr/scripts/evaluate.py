import argparse
import json
from pathlib import Path

from teaching_asr.checkpoint import load_checkpoint
from teaching_asr.data import LibriSpeechDataset, SyntheticSpeechDataset
from teaching_asr.evaluation import evaluate_model
from teaching_asr.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Conformer CTC checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", choices=["synthetic", "librispeech"], default="synthetic")
    parser.add_argument("--audio-dir")
    parser.add_argument("--dataset-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.data == "librispeech" and not args.audio_dir:
        parser.error("--audio-dir is required when --data=librispeech")
    if args.dataset_size < 1 or args.batch_size < 1:
        parser.error("dataset-size and batch-size must be positive")

    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    if args.data == "synthetic":
        dataset = SyntheticSpeechDataset(
            size=args.dataset_size,
            sample_rate=model.config.sample_rate,
            tokenizer=model.tokenizer,
            seed=args.seed,
        )
    else:
        dataset = LibriSpeechDataset(
            args.audio_dir, model.config.sample_rate, model.tokenizer
        )
    metrics = evaluate_model(model, dataset, args.batch_size, device)
    metrics.update(
        {
            "checkpoint": args.checkpoint,
            "checkpoint_step": payload["step"],
            "device": str(device),
        }
    )
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
