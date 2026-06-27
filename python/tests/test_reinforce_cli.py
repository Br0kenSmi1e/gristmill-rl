import json

import pytest

from gristmill_symbolics.reinforce.checkpoint import load_checkpoint
from gristmill_symbolics.reinforce.train import main
from tests.policy_fixtures import actionable_json


COMPACT_METRIC_KEYS = {
    "update_index",
    "batch_size",
    "reward_mean",
    "reward_std",
    "objective_loss_mean",
    "surrogate_loss",
    "final_flops_best",
    "params_changed",
}


def test_train_cli_completes_one_update_and_writes_checkpoint(tmp_path, capsys):
    input_path = tmp_path / "actionable.json"
    checkpoint_path = tmp_path / "checkpoint.pkl"
    input_path.write_text(actionable_json())

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--updates",
            "1",
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--seed",
            "14",
            "--state-token-pad-to",
            "256",
            "--action-token-pad-to",
            "256",
            "--definition-pad-to",
            "4",
            "--checkpoint-out",
            str(checkpoint_path),
        ]
    )

    captured = capsys.readouterr()
    line = json.loads(captured.out.strip().splitlines()[-1])
    assert exit_code == 0
    assert set(line) == COMPACT_METRIC_KEYS
    assert line["update_index"] == 0
    assert line["batch_size"] == 2
    assert "reward_mean" in line
    assert "reward_std" in line
    assert "objective_loss_mean" in line
    assert "surrogate_loss" in line
    assert "final_flops_best" in line
    assert "params_changed" in line
    assert checkpoint_path.exists()
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint.train_state.update_index == 1
    assert checkpoint.model_config.batch_size == 2
    assert checkpoint.model_config.max_steps == 1
    assert checkpoint.trainer_config.batch_size == 2


def test_train_cli_wires_static_pads_to_model_config_checkpoint(tmp_path, capsys):
    input_path = tmp_path / "actionable.json"
    checkpoint_path = tmp_path / "checkpoint.pkl"
    input_path.write_text(actionable_json())

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--updates",
            "1",
            "--batch-size",
            "1",
            "--max-steps",
            "1",
            "--seed",
            "22",
            "--state-token-pad-to",
            "256",
            "--action-token-pad-to",
            "256",
            "--definition-pad-to",
            "4",
            "--checkpoint-out",
            str(checkpoint_path),
        ]
    )

    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    checkpoint = load_checkpoint(checkpoint_path)
    assert exit_code == 0
    assert line["batch_size"] == 1
    assert checkpoint.model_config.state_token_pad_to == 256
    assert checkpoint.model_config.action_token_pad_to == 256
    assert checkpoint.model_config.definition_pad_to == 4


def test_train_cli_can_continue_from_checkpoint(tmp_path, capsys):
    input_path = tmp_path / "actionable.json"
    checkpoint_path = tmp_path / "checkpoint.pkl"
    input_path.write_text(actionable_json())
    main(
        [
            "--input",
            str(input_path),
            "--updates",
            "1",
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--seed",
            "15",
            "--state-token-pad-to",
            "256",
            "--action-token-pad-to",
            "256",
            "--definition-pad-to",
            "4",
            "--checkpoint-out",
            str(checkpoint_path),
        ]
    )

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--updates",
            "1",
            "--checkpoint-in",
            str(checkpoint_path),
            "--batch-size",
            "1",
            "--max-steps",
            "3",
            "--learning-rate",
            "0.25",
            "--checkpoint-out",
            str(checkpoint_path),
        ]
    )

    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    checkpoint = load_checkpoint(checkpoint_path)
    assert exit_code == 0
    assert set(line) == COMPACT_METRIC_KEYS
    assert line["update_index"] == 1
    assert line["batch_size"] == 2
    assert "objective_loss_mean" in line
    assert "surrogate_loss" in line
    assert checkpoint.train_state.update_index == 2
    assert checkpoint.model_config.batch_size == 2
    assert checkpoint.model_config.max_steps == 1
    assert checkpoint.trainer_config.batch_size == 2
    assert checkpoint.trainer_config.optimizer_config.learning_rate == 1.0e-3


@pytest.mark.parametrize("updates", ["0", "-1"])
def test_train_cli_rejects_non_positive_updates(tmp_path, updates):
    input_path = tmp_path / "actionable.json"
    input_path.write_text(actionable_json())

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--input",
                str(input_path),
                "--updates",
                updates,
            ]
        )

    assert exc_info.value.code == 2


def test_train_cli_requires_static_pads_for_fresh_run(tmp_path):
    input_path = tmp_path / "actionable.json"
    input_path.write_text(actionable_json())

    with pytest.raises(SystemExit) as exc_info:
        main(["--input", str(input_path)])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--state-token-pad-to", "0"),
        ("--action-token-pad-to", "-1"),
        ("--definition-pad-to", "0"),
    ],
)
def test_train_cli_rejects_non_positive_static_pad_flags(tmp_path, flag, value):
    input_path = tmp_path / "actionable.json"
    input_path.write_text(actionable_json())

    with pytest.raises(SystemExit) as exc_info:
        main(["--input", str(input_path), flag, value])

    assert exc_info.value.code == 2
