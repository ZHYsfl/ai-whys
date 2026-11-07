import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from teaching_asr.aut import AuT
from teaching_asr.aut_training import aut_training_step
from teaching_asr.data import LibriSpeechDataset, SyntheticSpeechDataset, collate_speech
from teaching_asr.llm_checkpoint import save_aut_checkpoint
from teaching_asr.llm_text import LanguageTokenizer
from teaching_asr.modern_config import ModernASRConfig
from teaching_asr.runtime import resolve_device, seed_everything
from teaching_asr.text import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the from-scratch AuT encoder-decoder")
    parser.add_argument("--data", choices=["synthetic", "librispeech"], default="synthetic")
    parser.add_argument("--audio-dir")
    parser.add_argument("--output-dir", default="runs/aut-synthetic")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.data == "librispeech" and not args.audio_dir:
        parser.error("--audio-dir is required with --data=librispeech")
    if min(args.steps, args.batch_size, args.dataset_size, args.log_every) < 1:
        parser.error("steps, batch-size, dataset-size and log-every must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ModernASRConfig()
    model = AuT(config.audio, config.llm).to(device)
    tokenizer = LanguageTokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    dataset_args = {"sample_rate": config.audio.sample_rate, "tokenizer": CharTokenizer()}
    if args.data == "synthetic":
        dataset = SyntheticSpeechDataset(size=args.dataset_size, seed=args.seed, **dataset_args)
    else:
        dataset = LibriSpeechDataset(args.audio_dir, **dataset_args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_speech,
        generator=torch.Generator().manual_seed(args.seed),
    )

    def batches():
        while True:
            yield from loader

    iterator = batches()
    save_aut_checkpoint(output_dir / "checkpoint-init.pt", model, 0)
    history = []
    for step in range(1, args.steps + 1):
        batch = next(iterator).to(device)
        metrics = aut_training_step(model, batch, tokenizer, optimizer, args.grad_clip)
        history.append(metrics)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"step={step}/{args.steps} loss={metrics['loss']:.5f} "
                f"grad_norm={metrics['grad_norm']:.4f}", flush=True
            )
    save_aut_checkpoint(output_dir / "checkpoint.pt", model, args.steps)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"saved AuT checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
