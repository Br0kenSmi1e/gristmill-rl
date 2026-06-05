import json
import math
import subprocess
import sys

from .transformer_policy_fixtures import actionable_json


def _tiny_train_command(input_path, *extra_args):
    return [
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
        *extra_args,
    ]


def _json_stdout_lines(result):
    return [
        json.loads(line)
        for line in result.stdout.strip().splitlines()
        if line.strip().startswith("{")
    ]


def test_reinforce_train_cli_completes_tiny_run(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        _tiny_train_command(input_path),
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
        _tiny_train_command(input_path, "--checkpoint-out", str(checkpoint_path)),
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["checkpoint_out"] == str(checkpoint_path)
    assert (checkpoint_path / "metadata.json").exists()
    assert (checkpoint_path / "state").exists()


def test_reinforce_train_cli_existing_checkpoint_fails_without_misleading_metrics(
    tmp_path,
):
    input_path = tmp_path / "input.json"
    checkpoint_path = tmp_path / "checkpoint"
    marker_path = checkpoint_path / "marker.txt"
    input_path.write_text(actionable_json())
    checkpoint_path.mkdir()
    marker_path.write_text("unchanged")

    result = subprocess.run(
        _tiny_train_command(input_path, "--checkpoint-out", str(checkpoint_path)),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "checkpoint path already exists" in result.stderr
    assert all(
        metrics.get("checkpoint_out") != str(checkpoint_path)
        for metrics in _json_stdout_lines(result)
    )
    assert marker_path.read_text() == "unchanged"
    assert not (checkpoint_path / "metadata.json").exists()
    assert not (checkpoint_path / "state").exists()


def test_reinforce_train_cli_resume_continues_update_counter(tmp_path):
    input_path = tmp_path / "input.json"
    checkpoint_in = tmp_path / "checkpoint-in"
    checkpoint_out = tmp_path / "checkpoint-out"
    input_path.write_text(actionable_json())

    subprocess.run(
        _tiny_train_command(input_path, "--checkpoint-out", str(checkpoint_in)),
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reinforce_training.train",
            "--input",
            str(input_path),
            "--updates",
            "2",
            "--batch-size",
            "2",
            "--num-workers",
            "1",
            "--checkpoint-in",
            str(checkpoint_in),
            "--checkpoint-out",
            str(checkpoint_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics_lines = _json_stdout_lines(result)
    assert len(metrics_lines) == 2
    assert metrics_lines[0]["update"] == 2
    assert metrics_lines[0]["updates"] == 3
    assert metrics_lines[0]["checkpoint_in"] == str(checkpoint_in)
    assert metrics_lines[0]["checkpoint_out"] is None
    assert metrics_lines[-1]["update"] == 3
    assert metrics_lines[-1]["updates"] == 3
    assert metrics_lines[-1]["checkpoint_in"] == str(checkpoint_in)
    assert metrics_lines[-1]["checkpoint_out"] == str(checkpoint_out)
    assert (checkpoint_out / "metadata.json").exists()
    assert (checkpoint_out / "state").exists()
