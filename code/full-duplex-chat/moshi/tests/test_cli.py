import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT = Path(__file__).parents[1]


def run_script(name, *args):
    result = subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / name), *map(str, args)],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_train_finetune_and_generate_cli(tmp_path):
    run_script("inspect_shapes.py")
    train_dir = tmp_path / "train"
    run_script(
        "train.py",
        "--max-steps",
        1,
        "--batch-size",
        2,
        "--output-dir",
        train_dir,
    )
    assert (train_dir / "checkpoint-last.pt").exists()

    adapter_dir = tmp_path / "adapter"
    run_script(
        "finetune.py",
        "--base-checkpoint",
        train_dir / "checkpoint-last.pt",
        "--max-steps",
        1,
        "--batch-size",
        2,
        "--output-dir",
        adapter_dir,
    )
    assert (adapter_dir / "checkpoint-last.pt").exists()

    full_dir = tmp_path / "full"
    run_script(
        "finetune.py",
        "--base-checkpoint",
        train_dir / "checkpoint-last.pt",
        "--full",
        "--max-steps",
        1,
        "--batch-size",
        2,
        "--output-dir",
        full_dir,
    )
    assert (full_dir / "checkpoint-last.pt").exists()

    mimi_dir = tmp_path / "mimi"
    run_script(
        "train_mimi.py",
        "--max-steps",
        1,
        "--batch-size",
        1,
        "--output-dir",
        mimi_dir,
    )
    assert (mimi_dir / "checkpoint-last.pt").exists()

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sf.write(input_path, np.zeros(3840, dtype=np.float32), 24_000)
    run_script(
        "generate.py",
        "--mimi-checkpoint",
        mimi_dir / "checkpoint-last.pt",
        "--model-checkpoint",
        train_dir / "checkpoint-last.pt",
        "--input",
        input_path,
        "--output",
        output_path,
        "--greedy",
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0
