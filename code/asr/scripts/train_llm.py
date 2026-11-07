import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from teaching_asr.data import LibriSpeechDataset, SyntheticSpeechDataset, collate_speech
from teaching_asr.llm_checkpoint import load_aut_checkpoint, save_llm_checkpoint
from teaching_asr.llm_training import evaluate_llm, llm_training_step
from teaching_asr.modern_config import ModernASRConfig
from teaching_asr.qwen3_asr import Qwen3ASR
from teaching_asr.runtime import resolve_device, seed_everything
from teaching_asr.text import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train from-scratch AuT + projector + Qwen3-style ASR")
    parser.add_argument("--data", choices=["synthetic", "librispeech"], default="synthetic")
    parser.add_argument("--audio-dir")
    parser.add_argument("--val-audio-dir")
    parser.add_argument("--aut-checkpoint", help="optional AuT pretraining initialization")
    parser.add_argument("--output-dir", default="runs/qwen3-asr-synthetic")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.dataset_size, args.eval_every, args.log_every) < 1:
        parser.error("steps, batch-size, dataset-size, eval-every and log-every must be positive")
    if args.data == "librispeech" and (not args.audio_dir or not args.val_audio_dir):
        parser.error("--audio-dir and --val-audio-dir are required with --data=librispeech")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ModernASRConfig()
    model = Qwen3ASR(config).to(device)
    if args.aut_checkpoint:
        aut, payload = load_aut_checkpoint(args.aut_checkpoint, device)
        model.audio_encoder.load_state_dict(aut.audio_encoder.state_dict())
        print(f"initialized AuT encoder from step {payload['step']}: {args.aut_checkpoint}")
        del aut

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"from-scratch Qwen3ASR parameters={total_parameters:,}; device={device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    def lr_factor(step: int) -> float:
        if args.warmup_steps and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    tokenizer = CharTokenizer()
    dataset_args = {"sample_rate": config.audio.sample_rate, "tokenizer": tokenizer}
    if args.data == "synthetic":
        dataset = SyntheticSpeechDataset(size=args.dataset_size, seed=args.seed, **dataset_args)
        eval_dataset = SyntheticSpeechDataset(
            size=max(8, args.dataset_size // 2), seed=args.seed + 10_000, **dataset_args
        )
    else:
        dataset = LibriSpeechDataset(args.audio_dir, **dataset_args)
        eval_dataset = LibriSpeechDataset(args.val_audio_dir, **dataset_args)
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
    save_llm_checkpoint(output_dir / "checkpoint-init.pt", model, 0)
    best_loss = float("inf")
    best_step = 0
    history = []
    evaluation_history = []
    for step in range(1, args.steps + 1):
        batch = next(iterator).to(device)
        metrics = llm_training_step(model, batch, optimizer, args.grad_clip)
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append(metrics)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"step={step}/{args.steps} loss={metrics['loss']:.5f} "
                f"grad_norm={metrics['grad_norm']:.4f} lr={metrics['learning_rate']:.7f}",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            evaluation = evaluate_llm(model, eval_dataset, args.batch_size, device, generate=False)
            evaluation_history.append({"step": step, **evaluation})
            print(f"validation step={step}: loss={evaluation['loss']:.5f}", flush=True)
            if evaluation["loss"] < best_loss:
                best_loss = evaluation["loss"]
                best_step = step
                save_llm_checkpoint(output_dir / "checkpoint-best.pt", model, step)

    save_llm_checkpoint(output_dir / "checkpoint.pt", model, args.steps)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output_dir / "evaluation-history.json").write_text(
        json.dumps(evaluation_history, indent=2) + "\n"
    )
    summary = {
        "arguments": vars(args),
        "device": str(device),
        "total_parameters": total_parameters,
        "first_metrics": history[0],
        "last_metrics": history[-1],
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "evaluation_history": evaluation_history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved final and best checkpoints; best_step={best_step}, val_loss={best_loss:.5f}")


if __name__ == "__main__":
    main()
