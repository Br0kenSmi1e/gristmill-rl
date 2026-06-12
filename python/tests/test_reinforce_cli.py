import json

import pytest

from gristmill_symbolics.reinforce.checkpoint import load_checkpoint
from gristmill_symbolics.reinforce.train import main
from tests.policy_fixtures import actionable_json


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
            "--checkpoint-out",
            str(checkpoint_path),
        ]
    )

    captured = capsys.readouterr()
    line = json.loads(captured.out.strip().splitlines()[-1])
    assert exit_code == 0
    assert line["update_index"] == 0
    assert line["batch_size"] == 2
    assert "reward_std" in line
    assert "target_score_count" in line
    assert "action_score_count" in line
    assert "stop_count" in line
    assert "empty_action_space_count" in line
    assert "params_changed" in line
    assert checkpoint_path.exists()
    assert load_checkpoint(checkpoint_path).train_state.update_index == 1


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
    assert line["update_index"] == 1
    assert line["batch_size"] == 2
    assert line["max_steps"] == 1
    assert checkpoint.train_state.update_index == 2
    assert checkpoint.rollout_config.batch_size == 2
    assert checkpoint.rollout_config.max_steps == 1
    assert checkpoint.train_state.optimizer_config.learning_rate == 1.0e-3


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
