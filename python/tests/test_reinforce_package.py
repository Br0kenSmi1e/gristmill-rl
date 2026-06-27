import pytest

import gristmill_symbolics.reinforce as reinforce
from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce.model import (
    CurrentTransformerModel as ImplCurrentTransformerModel,
)
from gristmill_symbolics.reinforce.trainer import (
    ReinforceTrainer as ImplReinforceTrainer,
)
from gristmill_symbolics.reinforce import (
    BaselineConfig,
    CheckpointData,
    CurrentTransformerModelConfig,
    ExpressionModel,
    FinalColumnMetrics,
    LossConfig,
    OptimizerConfig,
    PolicyState,
    ReinforceTrainerConfig,
    RewardConfig,
    RolloutConfig,
    TrainState,
    Trainer,
    TrainingError,
    UpdateMetrics,
    validate_model_config,
    validate_trainer_config,
)
from gristmill_symbolics.reinforce.types import (
    CHECKPOINT_SCHEMA_VERSION,
    DECISION_ACTION,
    DECISION_TARGET,
    TOKENIZER_SCHEMA_VERSION,
    validate_policy_state,
    validate_rollout_config,
)


def test_reinforce_package_exports_streamed_training_contracts():
    config = PolicyConfig(d_model=8)
    state = PolicyState(config=config, params={})
    expected_exports = {
        "advance_train_state",
        "BaselineConfig",
        "CheckpointData",
        "compute_advantages",
        "compute_rewards",
        "CurrentTransformerModel",
        "CurrentTransformerModelConfig",
        "ExpressionModel",
        "FinalColumnMetrics",
        "init_train_state",
        "load_checkpoint",
        "LossConfig",
        "make_optimizer",
        "make_rng_grid",
        "OptimizerConfig",
        "PolicyState",
        "ReinforceTrainer",
        "ReinforceTrainerConfig",
        "RewardConfig",
        "RolloutConfig",
        "save_checkpoint",
        "train_update",
        "TrainState",
        "Trainer",
        "TrainingError",
        "UpdateMetrics",
        "validate_model_config",
        "validate_trainer_config",
    }

    assert state.config is config
    assert state.params == {}
    assert RolloutConfig(batch_size=2, max_steps=3).seed == 0
    assert RolloutConfig(batch_size=2, max_steps=3).state_token_pad_to is None
    assert RolloutConfig(batch_size=2, max_steps=3).action_token_pad_to is None
    assert RolloutConfig(batch_size=2, max_steps=3).definition_pad_to is None
    assert RolloutConfig(batch_size=2, max_steps=3).static_policy_batch is False
    assert RewardConfig().kind == "log_flops_improvement"
    assert BaselineConfig().standardize is False
    assert LossConfig().require_scored_terms is True
    assert OptimizerConfig().learning_rate == pytest.approx(1.0e-3)
    assert issubclass(TrainingError, RuntimeError)
    assert reinforce.CheckpointData is CheckpointData
    assert reinforce.CurrentTransformerModel is ImplCurrentTransformerModel
    assert reinforce.CurrentTransformerModelConfig is CurrentTransformerModelConfig
    assert reinforce.ExpressionModel is ExpressionModel
    assert reinforce.FinalColumnMetrics is FinalColumnMetrics
    assert reinforce.ReinforceTrainer is ImplReinforceTrainer
    assert reinforce.ReinforceTrainerConfig is ReinforceTrainerConfig
    assert reinforce.Trainer is Trainer
    assert reinforce.TrainState is TrainState
    assert reinforce.UpdateMetrics is UpdateMetrics
    assert reinforce.validate_model_config is validate_model_config
    assert reinforce.validate_trainer_config is validate_trainer_config
    assert set(reinforce.__all__) == expected_exports


def test_reinforce_rng_and_schema_constants_are_stable():
    assert DECISION_TARGET == 0
    assert DECISION_ACTION == 1
    assert CHECKPOINT_SCHEMA_VERSION == 1
    assert TOKENIZER_SCHEMA_VERSION == 1


def test_rollout_config_validation_rejects_non_positive_values():
    with pytest.raises(TrainingError, match="batch_size"):
        validate_rollout_config(RolloutConfig(batch_size=0, max_steps=1))
    with pytest.raises(TrainingError, match="max_steps"):
        validate_rollout_config(RolloutConfig(batch_size=1, max_steps=0))


def test_rollout_config_validation_requires_integer_values():
    with pytest.raises(TrainingError, match="batch_size"):
        validate_rollout_config(RolloutConfig(batch_size=True, max_steps=1))
    with pytest.raises(TrainingError, match="max_steps"):
        validate_rollout_config(RolloutConfig(batch_size=1, max_steps=1.5))
    with pytest.raises(TrainingError, match="seed"):
        validate_rollout_config(RolloutConfig(batch_size=1, max_steps=1, seed=False))


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"state_token_pad_to": 0}, "state_token_pad_to"),
        ({"state_token_pad_to": True}, "state_token_pad_to"),
        ({"action_token_pad_to": 0}, "action_token_pad_to"),
        ({"action_token_pad_to": False}, "action_token_pad_to"),
        ({"definition_pad_to": 0}, "definition_pad_to"),
        ({"definition_pad_to": 1.5}, "definition_pad_to"),
        ({"static_policy_batch": 1}, "static_policy_batch"),
    ],
)
def test_rollout_config_validation_rejects_invalid_static_shape_fields(
    kwargs, field_name
):
    config = RolloutConfig(batch_size=1, max_steps=1, **kwargs)

    with pytest.raises(TrainingError, match=field_name):
        validate_rollout_config(config)


@pytest.mark.parametrize(
    "missing_field",
    [
        "state_token_pad_to",
        "action_token_pad_to",
        "definition_pad_to",
    ],
)
def test_rollout_config_validation_requires_all_static_pads(missing_field):
    kwargs = {
        "state_token_pad_to": 64,
        "action_token_pad_to": 64,
        "definition_pad_to": 4,
        "static_policy_batch": True,
    }
    kwargs[missing_field] = None

    with pytest.raises(TrainingError, match=missing_field):
        validate_rollout_config(RolloutConfig(batch_size=1, max_steps=1, **kwargs))


def test_policy_state_validation_requires_config_and_params_dict():
    with pytest.raises(TrainingError, match="PolicyState"):
        validate_policy_state(None)
    with pytest.raises(TrainingError, match="PolicyConfig"):
        validate_policy_state(PolicyState(config=object(), params={}))
    with pytest.raises(TrainingError, match="params"):
        validate_policy_state(PolicyState(config=PolicyConfig(), params=[]))
