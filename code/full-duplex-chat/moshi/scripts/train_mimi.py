import argparse

from teaching_moshi.config import MimiConfig
from teaching_moshi.mimi import MimiModel
from teaching_moshi.mimi_training import (
    MimiTrainingConfig,
    SyntheticMimiDataset,
    run_mimi_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train teaching Mimi with RVQ, GAN and distillation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-gan", action="store_true")
    parser.add_argument("--with-reconstruction", action="store_true")
    args = parser.parse_args()
    config = MimiConfig()
    model = MimiModel(config)
    dataset = SyntheticMimiDataset(config)
    training = MimiTrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_gan=not args.no_gan,
        adversarial_only=not args.with_reconstruction,
    )
    run_mimi_training(model, dataset, training, args.output_dir, args.device)


if __name__ == "__main__":
    main()
