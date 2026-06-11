import pytest

from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    BaselineConfig,
    LossConfig,
    OptimizerConfig,
    PolicyState,
    RewardConfig,
    RolloutConfig,
    TrainingError,
)
from gristmill_symbolics.reinforce.types import (
    CASE_ALREADY_FINISHED,
    CASE_EMPTY_ACTION_SPACE,
    CASE_STOP,
    CASE_VALID_ACTION,
    DECISION_ACTION,
    DECISION_TARGET,
    validate_policy_state,
    validate_rollout_config,
)


def test_reinforce_package_exports_phase3_contracts():
    config = PolicyConfig(d_model=8)
    state = PolicyState(config=config, params={})

    assert state.config is config
    assert state.params == {}
    assert RolloutConfig(batch_size=2, max_steps=3).seed == 0
    assert RewardConfig().kind == "log_flops_improvement"
    assert BaselineConfig().standardize is False
    assert LossConfig().require_scored_terms is True
    assert OptimizerConfig().learning_rate == pytest.approx(1.0e-3)
    assert issubclass(TrainingError, RuntimeError)


def test_reinforce_case_and_rng_constants_are_stable():
    assert CASE_ALREADY_FINISHED == 0
    assert CASE_STOP == 1
    assert CASE_EMPTY_ACTION_SPACE == 2
    assert CASE_VALID_ACTION == 3
    assert DECISION_TARGET == 0
    assert DECISION_ACTION == 1


def test_rollout_config_validation_rejects_non_positive_values():
    with pytest.raises(TrainingError, match="batch_size"):
        validate_rollout_config(RolloutConfig(batch_size=0, max_steps=1))
    with pytest.raises(TrainingError, match="max_steps"):
        validate_rollout_config(RolloutConfig(batch_size=1, max_steps=0))


def test_policy_state_validation_requires_config_and_params_dict():
    with pytest.raises(TrainingError, match="PolicyConfig"):
        validate_policy_state(PolicyState(config=object(), params={}))
    with pytest.raises(TrainingError, match="params"):
        validate_policy_state(PolicyState(config=PolicyConfig(), params=[]))
