import argparse
import json

from teaching_codec.audio_io import load_wav, save_wav
from teaching_moshi.checkpoint import load_mimi_checkpoint, load_moshi_checkpoint
from teaching_moshi.inference import MoshiDialogue
from teaching_moshi.loaders import load_official_models
from teaching_moshi.text import ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming user audio -> Moshi text and audio")
    parser.add_argument("--input", required=True, help="24 kHz mono user WAV")
    parser.add_argument("--output", required=True, help="output WAV")
    parser.add_argument("--mimi-checkpoint")
    parser.add_argument("--model-checkpoint")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--hf-repository", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    if args.official:
        bundle = load_official_models(args.hf_repository, device=args.device)
        mimi, model, tokenizer = bundle.mimi, bundle.moshi, bundle.tokenizer
    else:
        if not args.mimi_checkpoint or not args.model_checkpoint:
            raise ValueError("provide both teaching checkpoints, or use --official")
        mimi, _ = load_mimi_checkpoint(args.mimi_checkpoint, args.device)
        model, _ = load_moshi_checkpoint(args.model_checkpoint, args.device)
        tokenizer = ByteTokenizer()
    waveform, _ = load_wav(args.input, expected_sample_rate=mimi.sample_rate)
    dialogue = MoshiDialogue(mimi, model, tokenizer, use_sampling=not args.greedy)
    output = dialogue.respond(waveform[None].to(args.device))
    save_wav(args.output, output.waveform.cpu(), mimi.sample_rate)
    print(json.dumps({"text": output.text, "wav": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
