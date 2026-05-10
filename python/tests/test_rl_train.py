import json
import subprocess
import sys

from .rl_fixtures import actionable_json


def test_train_cli_completes_tiny_run(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["episodes"] == 1
    assert metrics["replay_size"] >= 1
    assert metrics["last_total_loss"] > 0.0
    assert metrics["params_changed"]
