import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from teaching_codec import CodecConfig, TeachingCodec
from teaching_codec.audio_io import save_wav
from teaching_codec.checkpoint import save_checkpoint
from teaching_codec.data import SyntheticAudioDataset, WavFolderDataset
from teaching_codec.discriminator import MultiScaleDiscriminator
from teaching_codec.losses import ReconstructionLoss
from teaching_codec.runtime import resolve_device, seed_everything
from teaching_codec.training import training_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the teaching codec on small audio clips")
    parser.add_argument("--data", choices=["synthetic", "wav"], default="synthetic")
    parser.add_argument("--audio-dir")
    parser.add_argument("--output-dir", default="runs/synthetic")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-size", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--num-codebooks", type=int, default=4)
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--quantizer-dropout", action="store_true")
    parser.add_argument("--dropout-rate", type=float, default=1.0)
    parser.add_argument("--use-gan", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.dataset_size < 1:
        parser.error("steps, batch-size and dataset-size must be positive")
    if args.data == "wav" and not args.audio_dir:
        parser.error("--audio-dir is required when --data=wav")
    if args.no_quantization and args.quantizer_dropout:
        parser.error("--quantizer-dropout requires quantization")
    if args.log_every < 1 or args.save_every < 1:
        parser.error("--log-every and --save-every must be positive")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        probe = torch.ones(8, device=device)
        probe_result = probe @ probe
        torch.cuda.synchronize()
        print(
            f"torch={torch.__version__}, CUDA={torch.version.cuda}, "
            f"GPU={torch.cuda.get_device_name(device)}, capability={torch.cuda.get_device_capability(device)}, "
            f"kernel_smoke={probe_result.item()}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = CodecConfig(
        sample_rate=args.sample_rate,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        codebook_size=args.codebook_size,
        num_codebooks=args.num_codebooks,
        use_quantization=not args.no_quantization,
    )
    model = TeachingCodec(config).to(device)
    discriminator = MultiScaleDiscriminator().to(device) if args.use_gan else None
    reconstruction_loss = ReconstructionLoss().to(device)
    betas = (0.5, 0.9) if args.use_gan else (0.9, 0.999)
    generator_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=betas)
    discriminator_optimizer = (
        torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=betas)
        if discriminator is not None
        else None
    )
    save_checkpoint(output_dir / "checkpoint-init.pt", model, 0, discriminator)

    dataset_args = {
        "sample_rate": args.sample_rate,
        "num_samples": args.sample_rate,
        "seed": args.seed,
    }
    if args.data == "synthetic":
        dataset = SyntheticAudioDataset(size=args.dataset_size, **dataset_args)
    else:
        dataset = WavFolderDataset(args.audio_dir, **dataset_args)
    print(f"dataset={args.data}, clips={len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )

    def batches():
        while True:
            yield from loader

    iterator = batches()
    history = []
    for step in range(1, args.steps + 1):
        batch = next(iterator).to(device)
        metrics = training_step(
            model,
            discriminator,
            batch,
            generator_optimizer,
            discriminator_optimizer,
            reconstruction_loss,
            quantizer_dropout=args.quantizer_dropout,
            dropout_rate=args.dropout_rate,
        )
        history.append(metrics)
        if step % args.save_every == 0:
            save_checkpoint(output_dir / "checkpoint-last.pt", model, step, discriminator)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            fields = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            print(f"step={step}/{args.steps} {fields}", flush=True)

    save_checkpoint(output_dir / "checkpoint.pt", model, args.steps, discriminator)
    model.eval()
    with torch.no_grad():
        sample = batch[:1]
        if config.use_quantization:
            reconstruction = model.reconstruct(sample)
            codes = model.encode(sample)
            usage = [
                f"q{index}={codes[:, index].unique().numel()}/{config.codebook_size}"
                for index in range(codes.shape[1])
            ]
            print("sample codebook usage: " + ", ".join(usage))
        else:
            reconstruction = model(sample).reconstruction
    save_wav(output_dir / "original.wav", sample.cpu(), config.sample_rate)
    save_wav(output_dir / "reconstructed.wav", reconstruction.cpu(), config.sample_rate)

    with (output_dir / "history.json").open("w") as file:
        json.dump(history, file, indent=2)
    summary = {
        "arguments": vars(args),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "first_metrics": history[0],
        "last_metrics": history[-1],
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"saved checkpoint, WAVs and metrics to {output_dir}")
    print(f"parameters: {summary['parameters']:,}; device: {device}")
    if device.type == "cuda":
        print(f"peak CUDA memory: {summary['peak_cuda_memory_mib']:.1f} MiB")


if __name__ == "__main__":
    main()
