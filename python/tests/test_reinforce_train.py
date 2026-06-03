import json
import math
import subprocess
import sys

from .transformer_policy_fixtures import actionable_json


def test_reinforce_train_cli_completes_tiny_run(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reinforce_training.train",
            "--input",
            str(input_path),
            "--updates",
            "1",
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--num-workers",
            "1",
            "--hidden-dim",
            "16",
            "--num-heads",
            "4",
            "--num-layers",
            "1",
            "--mlp-dim",
            "32",
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["updates"] == 1
    assert metrics["batch_size"] == 2
    assert metrics["num_workers"] == 1
    assert isinstance(metrics["params_changed"], bool)
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["mean_reward"])
    assert math.isfinite(metrics["mean_final_log_flops"])
    assert math.isfinite(metrics["mean_sample_log_prob"])
    assert math.isfinite(metrics["mean_trajectory_log_prob"])
    assert metrics["checkpoint_out"] is None


def test_reinforce_train_cli_writes_checkpoint(tmp_path):
    input_path = tmp_path / "input.json"
    checkpoint_path = tmp_path / "checkpoint"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reinforce_training.train",
            "--input",
            str(input_path),
            "--updates",
            "1",
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--num-workers",
            "1",
            "--hidden-dim",
            "16",
            "--num-heads",
            "4",
            "--num-layers",
            "1",
            "--mlp-dim",
            "32",
            "--checkpoint-out",
            str(checkpoint_path),
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["checkpoint_out"] == str(checkpoint_path)
    assert (checkpoint_path / "metadata.json").exists()
    assert (checkpoint_path / "state").exists()
