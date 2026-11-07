import argparse
import json
from pathlib import Path

from teaching_asr.data import LibriSpeechDataset, SyntheticSpeechDataset
from teaching_asr.llm_checkpoint import load_llm_checkpoint
from teaching_asr.llm_training import evaluate_llm
from teaching_asr.runtime import resolve_device
from teaching_asr.text import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and generate with teaching Qwen3ASR")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", choices=["synthetic", "librispeech"], default="synthetic")
    parser.add_argument("--audio-dir")
    parser.add_argument("--dataset-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.data == "librispeech" and not args.audio_dir:
        parser.error("--audio-dir is required with --data=librispeech")
    device = resolve_device(args.device)
    model, payload = load_llm_checkpoint(args.checkpoint, device)
    dataset_args = {"sample_rate": model.config.audio.sample_rate, "tokenizer": CharTokenizer()}
    if args.data == "synthetic":
        dataset = SyntheticSpeechDataset(size=args.dataset_size, seed=args.seed, **dataset_args)
    else:
        dataset = LibriSpeechDataset(args.audio_dir, **dataset_args)
    metrics = evaluate_llm(
        model, dataset, args.batch_size, device, generate=True, max_new_tokens=args.max_new_tokens
    )
    metrics.update({"checkpoint": args.checkpoint, "checkpoint_step": payload["step"]})
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
