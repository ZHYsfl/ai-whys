import argparse
import json
from pathlib import Path

import torch

from teaching_asr.audio_io import load_audio
from teaching_asr.llm_checkpoint import load_llm_checkpoint
from teaching_asr.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe with from-scratch AuT + Qwen3-style ASR")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, payload = load_llm_checkpoint(args.checkpoint, device)
    waveform = load_audio(args.input, model.config.audio.sample_rate)
    lengths = torch.tensor([waveform.shape[0]], device=device)
    transcript = model.transcribe(
        waveform[None, :].to(device), lengths, max_new_tokens=args.max_new_tokens
    )[0]
    print(transcript)
    if args.output:
        result = {
            "input": args.input,
            "checkpoint": args.checkpoint,
            "checkpoint_step": payload["step"],
            "text": transcript,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
