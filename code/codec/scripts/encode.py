import argparse

from teaching_codec.audio_io import load_wav, save_codes
from teaching_codec.checkpoint import load_checkpoint
from teaching_codec.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a WAV file into discrete codec tokens")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="output .npz codes file")
    parser.add_argument("--num-quantizers", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, _, _ = load_checkpoint(args.checkpoint, device)
    waveform, _ = load_wav(args.input, model.config.sample_rate)
    original_length = waveform.shape[-1]
    codes = model.encode(waveform.unsqueeze(0).to(device), args.num_quantizers)
    save_codes(args.output, codes, original_length)
    print(f"device: {device}")
    print(f"codes: {tuple(codes.shape)} -> {args.output}")
    print(f"frame rate: {model.config.frame_rate} frames/s")
    print(f"bitrate: {model.config.bitrate(codes.shape[1])} bit/s (before file/entropy overhead)")


if __name__ == "__main__":
    main()
