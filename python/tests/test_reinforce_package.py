import pytest

import gristmill_symbolics.reinforce as reinforce
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
    OptimizerConfig,
    ReinforceTrainerConfig,
    RewardConfig,
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
)


def test_reinforce_package_exports_protocol_training_contracts():
    expected_exports = {
        "advance_train_state",
        "BaselineConfig",
        "CheckpointData",
        "compute_advantages",
        "compute_rewards",
        "CurrentTransformerModel",
        "CurrentTransformerModelConfig",
        "ExpressionModel",
        "init_train_state",
        "load_checkpoint",
        "make_optimizer",
        "OptimizerConfig",
        "ReinforceTrainer",
        "ReinforceTrainerConfig",
        "RewardConfig",
        "save_checkpoint",
        "train_update",
        "TrainState",
        "Trainer",
        "TrainingError",
        "UpdateMetrics",
        "validate_model_config",
        "validate_trainer_config",
    }

    assert RewardConfig().kind == "log_flops_improvement"
    assert BaselineConfig().standardize is False
    assert OptimizerConfig().learning_rate == pytest.approx(1.0e-3)
    assert issubclass(TrainingError, RuntimeError)
    assert reinforce.CheckpointData is CheckpointData
    assert reinforce.CurrentTransformerModel is ImplCurrentTransformerModel
    assert reinforce.CurrentTransformerModelConfig is CurrentTransformerModelConfig
    assert reinforce.ExpressionModel is ExpressionModel
    assert reinforce.ReinforceTrainer is ImplReinforceTrainer
    assert reinforce.ReinforceTrainerConfig is ReinforceTrainerConfig
    assert reinforce.Trainer is Trainer
    assert reinforce.TrainState is TrainState
    assert reinforce.UpdateMetrics is UpdateMetrics
    assert reinforce.validate_model_config is validate_model_config
    assert reinforce.validate_trainer_config is validate_trainer_config
    assert set(reinforce.__all__) == expected_exports
    removed_exports = {
        "Rollout" + "Config",
        "Loss" + "Config",
        "Policy" + "State",
        "Final" + "Column" + "Metrics",
    }
    assert removed_exports.isdisjoint(reinforce.__all__)
    for name in removed_exports:
        assert not hasattr(reinforce, name)


def test_reinforce_rng_and_schema_constants_are_stable():
    assert DECISION_TARGET == 0
    assert DECISION_ACTION == 1
    assert CHECKPOINT_SCHEMA_VERSION == 2
    assert TOKENIZER_SCHEMA_VERSION == 1
