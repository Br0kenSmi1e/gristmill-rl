import pytest

import gristmill_symbolics.reinforce as reinforce
from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    BaselineConfig,
    CheckpointData,
    FinalColumnMetrics,
    LossConfig,
    OptimizerConfig,
    PolicyState,
    RewardConfig,
    RolloutConfig,
    TrainState,
    TrainingError,
    UpdateMetrics,
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
        "BaselineConfig",
        "CheckpointData",
        "compute_advantages",
        "compute_rewards",
        "FinalColumnMetrics",
        "init_train_state",
        "load_checkpoint",
        "LossConfig",
        "make_optimizer",
        "make_rng_grid",
        "OptimizerConfig",
        "PolicyState",
        "RewardConfig",
        "RolloutConfig",
        "save_checkpoint",
        "train_update",
        "TrainState",
        "TrainingError",
        "UpdateMetrics",
    }

    assert state.config is config
    assert state.params == {}
    assert RolloutConfig(batch_size=2, max_steps=3).seed == 0
    assert RewardConfig().kind == "log_flops_improvement"
    assert BaselineConfig().standardize is False
    assert LossConfig().require_scored_terms is True
    assert OptimizerConfig().learning_rate == pytest.approx(1.0e-3)
    assert issubclass(TrainingError, RuntimeError)
    assert reinforce.CheckpointData is CheckpointData
    assert reinforce.FinalColumnMetrics is FinalColumnMetrics
    assert reinforce.TrainState is TrainState
    assert reinforce.UpdateMetrics is UpdateMetrics
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


def test_policy_state_validation_requires_config_and_params_dict():
    with pytest.raises(TrainingError, match="PolicyState"):
        validate_policy_state(None)
    with pytest.raises(TrainingError, match="PolicyConfig"):
        validate_policy_state(PolicyState(config=object(), params={}))
    with pytest.raises(TrainingError, match="params"):
        validate_policy_state(PolicyState(config=PolicyConfig(), params=[]))
