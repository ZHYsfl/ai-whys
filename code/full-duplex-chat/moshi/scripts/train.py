import argparse

from teaching_moshi.checkpoint import load_moshi_checkpoint
from teaching_moshi.model import MoshiLM
from teaching_moshi.training import run_training

from common import (
    add_data_arguments,
    add_training_arguments,
    dataset_from_args,
    training_config_from_args,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the teaching Moshi RQ-Transformer")
    add_data_arguments(parser)
    add_training_arguments(parser)
    parser.add_argument("--init-checkpoint", help="optional full teaching checkpoint")
    args = parser.parse_args()
    if args.init_checkpoint:
        model, _ = load_moshi_checkpoint(args.init_checkpoint, args.device)
    else:
        model = MoshiLM()
    dataset = dataset_from_args(args, model.config)
    run_training(
        model,
        dataset,
        training_config_from_args(args),
        args.output_dir,
        args.device,
    )


if __name__ == "__main__":
    main()
