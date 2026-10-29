import argparse

from teaching_codec.audio_io import load_codes, save_wav
from teaching_codec.checkpoint import load_checkpoint
from teaching_codec.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode discrete codec tokens into a WAV file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, _, _ = load_checkpoint(args.checkpoint, device)
    codes, original_length = load_codes(args.codes)
    reconstruction = model.decode(codes.to(device), original_length)
    save_wav(args.output, reconstruction, model.config.sample_rate)
    print(f"device: {device}")
    print(f"reconstruction: {tuple(reconstruction.shape)} -> {args.output}")
    print("decoder inputs: codes + learned codebooks/decoder weights; no original waveform or encoder latent")


if __name__ == "__main__":
    main()
