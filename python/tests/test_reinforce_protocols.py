import pytest

import gristmill_symbolics.reinforce as reinforce
from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    CurrentTransformerModelConfig,
    OptimizerConfig,
    ReinforceTrainerConfig,
    TrainingError,
    validate_model_config,
    validate_trainer_config,
)


def test_current_transformer_model_config_requires_static_positive_shapes():
    config = CurrentTransformerModelConfig(
        policy_config=PolicyConfig(d_model=8),
        batch_size=2,
        max_steps=3,
        state_token_pad_to=128,
        action_token_pad_to=256,
        definition_pad_to=8,
    )

    validate_model_config(config)
    assert config.batch_size == 2
    assert config.max_steps == 3


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"max_steps": 0}, "max_steps"),
        ({"state_token_pad_to": None}, "state_token_pad_to"),
        ({"action_token_pad_to": 0}, "action_token_pad_to"),
        ({"definition_pad_to": True}, "definition_pad_to"),
    ],
)
def test_current_transformer_model_config_rejects_invalid_static_shapes(
    kwargs, field_name
):
    values = {
        "policy_config": PolicyConfig(d_model=8),
        "batch_size": 1,
        "max_steps": 1,
        "state_token_pad_to": 128,
        "action_token_pad_to": 128,
        "definition_pad_to": 4,
    }
    values.update(kwargs)

    with pytest.raises(TrainingError, match=field_name):
        validate_model_config(CurrentTransformerModelConfig(**values))


def test_reinforce_trainer_config_owns_batch_reward_baseline_and_optimizer():
    config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
    )

    validate_trainer_config(config)
    assert config.batch_size == 2
    assert config.reward_config.kind == "log_flops_improvement"
    assert config.baseline_config.standardize is False
    assert config.optimizer_config.learning_rate == pytest.approx(1.0e-2)


def test_reinforce_package_exports_protocol_boundary_names():
    expected = {
        "BaselineConfig",
        "CheckpointData",
        "CurrentTransformerModel",
        "CurrentTransformerModelConfig",
        "ExpressionModel",
        "ReinforceTrainer",
        "ReinforceTrainerConfig",
        "RewardConfig",
        "TrainState",
        "Trainer",
        "TrainingError",
        "UpdateMetrics",
        "advance_train_state",
        "compute_advantages",
        "compute_rewards",
        "init_train_state",
        "load_checkpoint",
        "make_optimizer",
        "save_checkpoint",
        "validate_model_config",
        "validate_trainer_config",
    }

    assert expected.issubset(set(reinforce.__all__))
