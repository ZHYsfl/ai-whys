import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from teaching_codec.checkpoint import load_checkpoint
from teaching_codec.data import WavFolderDataset
from teaching_codec.losses import ReconstructionLoss
from teaching_codec.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a codec checkpoint on a WAV/FLAC folder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, _, payload = load_checkpoint(args.checkpoint, device)
    dataset = WavFolderDataset(
        args.audio_dir,
        sample_rate=model.config.sample_rate,
        num_samples=model.config.sample_rate,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    reconstruction_loss = ReconstructionLoss().to(device)
    totals = {
        "waveform": 0.0,
        "stft": 0.0,
        "reconstruction_total": 0.0,
        "codebook": 0.0,
        "commitment": 0.0,
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            reconstruction = reconstruction_loss(batch, output.reconstruction)
            batch_size = batch.shape[0]
            totals["waveform"] += float(reconstruction.waveform) * batch_size
            totals["stft"] += float(reconstruction.stft) * batch_size
            totals["reconstruction_total"] += float(reconstruction.total) * batch_size
            totals["codebook"] += float(output.codebook_loss) * batch_size
            totals["commitment"] += float(output.commitment_loss) * batch_size

    metrics = {
        key: value / len(dataset) for key, value in totals.items()
    }
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": payload["step"],
            "audio_dir": str(args.audio_dir),
            "num_clips": len(dataset),
            "device": str(device),
        }
    )
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
