import argparse

from teaching_moshi.checkpoint import load_mimi_checkpoint
from teaching_moshi.config import MoshiConfig
from teaching_moshi.data import StereoDialogueDataset, SyntheticDialogueDataset
from teaching_moshi.text import ByteTokenizer, SentencePieceTokenizer
from teaching_moshi.training import TrainingConfig


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", help="JSONL stereo dataset; omit for synthetic tokens")
    parser.add_argument("--mimi-checkpoint", help="teaching Mimi checkpoint for --manifest")
    parser.add_argument("--tokenizer", help="SentencePiece model; omit for the byte tokenizer")
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--dataset-size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=8)


def dataset_from_args(args: argparse.Namespace, config: MoshiConfig):
    if args.manifest is None:
        return SyntheticDialogueDataset(
            config, size=args.dataset_size, frames=args.frames
        )
    if args.mimi_checkpoint is None:
        raise ValueError("--manifest requires --mimi-checkpoint")
    mimi, _ = load_mimi_checkpoint(args.mimi_checkpoint, args.device)
    tokenizer = (
        SentencePieceTokenizer(args.tokenizer) if args.tokenizer else ByteTokenizer()
    )
    if tokenizer.vocab_size != config.text_card:
        raise ValueError("tokenizer vocabulary does not match Moshi text_card")
    return StereoDialogueDataset(
        args.manifest,
        mimi,
        tokenizer,
        config,
        duration_sec=args.duration_sec,
    )


def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        save_adapters=getattr(args, "save_adapters", False),
    )


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
