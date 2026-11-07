import argparse
import json
from pathlib import Path

import torch

from teaching_asr.audio_io import load_audio
from teaching_asr.checkpoint import load_checkpoint
from teaching_asr.ctc import beam_search_decode
from teaching_asr.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe one WAV/FLAC with Conformer CTC")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    waveform = load_audio(args.input, model.config.sample_rate)
    lengths = torch.tensor([waveform.shape[0]], device=device)
    with torch.no_grad():
        output = model(waveform[None, :].to(device), lengths)
        transcript = beam_search_decode(output.logits, output.output_lengths, model.tokenizer)[0]
    print(transcript)
    if args.output:
        result = {
            "input": args.input,
            "checkpoint": args.checkpoint,
            "checkpoint_step": payload["step"],
            "text": transcript,
            "output_frames": int(output.output_lengths[0]),
            "output_frame_rate": model.config.output_frame_rate,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
