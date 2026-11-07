import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from teaching_asr import ASRConfig, CharTokenizer, ConformerCTC
from teaching_asr.checkpoint import load_checkpoint, save_checkpoint
from teaching_asr.data import LibriSpeechDataset, SyntheticSpeechDataset, collate_speech
from teaching_asr.evaluation import evaluate_model
from teaching_asr.runtime import resolve_device, seed_everything
from teaching_asr.training import training_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small modern Conformer CTC ASR")
    parser.add_argument("--data", choices=["synthetic", "librispeech"], default="synthetic")
    parser.add_argument("--audio-dir")
    parser.add_argument("--val-audio-dir")
    parser.add_argument("--init-checkpoint", help="initialize model weights, not optimizer state")
    parser.add_argument("--output-dir", default="runs/synthetic")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-size", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--conv-kernel-size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spec-augment", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(
        args.steps, args.batch_size, args.dataset_size, args.log_every, args.eval_every, args.save_every
    ) < 1:
        parser.error("steps, batch-size, dataset-size, log/eval/save-every must be positive")
    if args.warmup_steps < 0 or args.grad_clip <= 0 or args.learning_rate <= 0:
        parser.error("warmup-steps must be >= 0; grad-clip and learning-rate must be positive")
    if args.data == "librispeech" and (not args.audio_dir or not args.val_audio_dir):
        parser.error("--audio-dir and --val-audio-dir are required when --data=librispeech")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        probe = torch.ones(8, device=device)
        probe_result = probe @ probe
        torch.cuda.synchronize()
        print(
            f"torch={torch.__version__}, CUDA={torch.version.cuda}, "
            f"GPU={torch.cuda.get_device_name(device)}, kernel_smoke={probe_result.item()}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ASRConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        conv_kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
        spec_augment=args.spec_augment,
    )
    tokenizer = CharTokenizer()
    model = ConformerCTC(config, tokenizer).to(device)
    if args.init_checkpoint:
        initialized, payload = load_checkpoint(args.init_checkpoint, "cpu")
        if initialized.config != config:
            raise ValueError("init-checkpoint config does not match the training arguments")
        model.load_state_dict(initialized.state_dict())
        print(f"initialized weights from step {payload['step']}: {args.init_checkpoint}")
        del initialized
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    def lr_factor(step: int) -> float:
        if args.warmup_steps and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    save_checkpoint(output_dir / "checkpoint-init.pt", model, 0)

    dataset_args = {"sample_rate": args.sample_rate, "tokenizer": tokenizer}
    if args.data == "synthetic":
        dataset = SyntheticSpeechDataset(size=args.dataset_size, seed=args.seed, **dataset_args)
        eval_dataset = SyntheticSpeechDataset(
            size=max(8, args.dataset_size // 2), seed=args.seed + 10_000, **dataset_args
        )
    else:
        dataset = LibriSpeechDataset(args.audio_dir, **dataset_args)
        eval_dataset = LibriSpeechDataset(args.val_audio_dir, **dataset_args)
    print(f"dataset={args.data}, train={len(dataset)}, eval={len(eval_dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_speech,
        generator=torch.Generator().manual_seed(args.seed),
    )

    def batches():
        while True:
            yield from loader

    iterator = batches()
    history = []
    evaluation_history = []
    best_loss = float("inf")
    best_step = 0
    for step in range(1, args.steps + 1):
        batch = next(iterator).to(device)
        metrics = training_step(model, batch, optimizer, args.grad_clip)
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append(metrics)
        if step % args.save_every == 0:
            save_checkpoint(output_dir / "checkpoint-last.pt", model, step)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"step={step}/{args.steps} loss={metrics['loss']:.5f} "
                f"grad_norm={metrics['grad_norm']:.4f} lr={metrics['learning_rate']:.6f}",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            evaluation = evaluate_model(model, eval_dataset, args.batch_size, device)
            evaluation_history.append(
                {
                    "step": step,
                    "loss": evaluation["loss"],
                    "wer": evaluation["wer"],
                    "cer": evaluation["cer"],
                }
            )
            print(
                f"validation step={step}: loss={evaluation['loss']:.5f}, "
                f"WER={evaluation['wer']:.2%}, CER={evaluation['cer']:.2%}",
                flush=True,
            )
            if evaluation["loss"] < best_loss:
                best_loss = evaluation["loss"]
                best_step = step
                save_checkpoint(output_dir / "checkpoint-best.pt", model, step)

    save_checkpoint(output_dir / "checkpoint.pt", model, args.steps)
    evaluation = evaluate_model(model, eval_dataset, args.batch_size, device)
    evaluation.update({"checkpoint_step": args.steps, "device": str(device)})
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output_dir / "evaluation-history.json").write_text(
        json.dumps(evaluation_history, indent=2) + "\n"
    )
    summary = {
        "arguments": vars(args),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "first_metrics": history[0],
        "last_metrics": history[-1],
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "eval_loss": evaluation["loss"],
        "eval_wer": evaluation["wer"],
        "eval_cer": evaluation["cer"],
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"eval: loss={evaluation['loss']:.5f}, WER={evaluation['wer']:.2%}, "
        f"CER={evaluation['cer']:.2%}, samples={evaluation['num_samples']}"
    )
    for prediction in evaluation["predictions"][:3]:
        print(f"reference: {prediction['reference']}\n  hypothesis: {prediction['hypothesis']}")
    print(
        f"saved checkpoints to {output_dir}; parameters={summary['parameters']:,}; "
        f"best_step={best_step}, best_val_loss={best_loss:.5f}"
    )


if __name__ == "__main__":
    main()
