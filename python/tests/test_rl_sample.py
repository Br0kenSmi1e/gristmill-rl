import json
import subprocess
import sys

from gristmill_symbolics import TensorComputation

from .rl_fixtures import actionable_json


def make_checkpoint(tmp_path):
    input_path = tmp_path / "input.json"
    checkpoint_path = tmp_path / "checkpoint"
    input_path.write_text(actionable_json())

    subprocess.run(
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
            "--hidden-dim",
            "16",
            "--checkpoint-out",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return input_path, checkpoint_path


def test_sample_cli_writes_rewritten_outputs(tmp_path):
    input_path, checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "samples"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.sample",
            "--checkpoint",
            str(checkpoint_path),
            "--input",
            str(input_path),
            "--samples",
            "2",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--temperature",
            "0.0",
            "--output-dir",
            str(output_dir),
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["samples"] == 2
    assert summary["output_dir"] == str(output_dir)

    for index in range(2):
        sample_dir = output_dir / f"sample-{index:03d}"
        final_path = sample_dir / "final.json"
        metrics_path = sample_dir / "metrics.json"
        assert final_path.exists()
        assert metrics_path.exists()
        TensorComputation.load_json(final_path)

        metrics = json.loads(metrics_path.read_text())
        assert metrics["sample"] == index
        assert metrics["steps"] >= 0
        assert metrics["checkpoint"] == str(checkpoint_path)


def test_sample_cli_refuses_existing_sample_directory_without_overwrite(tmp_path):
    input_path, checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "samples"
    (output_dir / "sample-000").mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.sample",
            "--checkpoint",
            str(checkpoint_path),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "sample-000" in result.stderr


def test_sample_cli_preflights_existing_sample_directories_before_writing(tmp_path):
    input_path, checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "samples"
    (output_dir / "sample-001").mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.sample",
            "--checkpoint",
            str(checkpoint_path),
            "--input",
            str(input_path),
            "--samples",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "sample-001" in result.stderr
    assert not (output_dir / "sample-000").exists()
