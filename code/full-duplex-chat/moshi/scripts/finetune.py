import argparse

from teaching_moshi.checkpoint import load_moshi_checkpoint
from teaching_moshi.lora import apply_lora
from teaching_moshi.training import run_training

from common import (
    add_data_arguments,
    add_training_arguments,
    dataset_from_args,
    training_config_from_args,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full or LoRA fine-tuning for teaching Moshi")
    add_data_arguments(parser)
    add_training_arguments(parser)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--full", action="store_true", help="full fine-tuning instead of LoRA")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scaling", type=float, default=2.0)
    parser.add_argument("--finetune-embeddings", action="store_true")
    parser.add_argument("--save-adapters", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    model, _ = load_moshi_checkpoint(args.base_checkpoint, args.device)
    if not args.full:
        count = apply_lora(
            model, args.lora_rank, args.lora_scaling, args.finetune_embeddings
        )
        print(f"enabled {count} LoRA adapters")
    save_adapters = not args.full if args.save_adapters is None else args.save_adapters
    if args.full and save_adapters:
        raise ValueError("full fine-tuning cannot save adapter-only checkpoints")
    config = training_config_from_args(args)
    config.save_adapters = save_adapters
    dataset = dataset_from_args(args, model.config)
    run_training(model, dataset, config, args.output_dir, args.device)


if __name__ == "__main__":
    main()
