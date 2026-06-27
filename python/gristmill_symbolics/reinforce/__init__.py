"""On-policy REINFORCE trainer over the row rewrite environment."""

from .checkpoint import load_checkpoint, save_checkpoint
from .model import CurrentTransformerModel
from .protocols import ExpressionModel, Trainer
from .trainer import ReinforceTrainer
from .types import (
    BaselineConfig,
    CheckpointData,
    CurrentTransformerModelConfig,
    FinalColumnMetrics,
    LossConfig,
    OptimizerConfig,
    PolicyState,
    ReinforceTrainerConfig,
    RewardConfig,
    RolloutConfig,
    TrainState,
    TrainingError,
    UpdateMetrics,
    validate_model_config,
    validate_trainer_config,
)


def compute_advantages(*args, **kwargs):
    from .objective import compute_advantages as _compute_advantages

    return _compute_advantages(*args, **kwargs)


def compute_rewards(*args, **kwargs):
    from .objective import compute_rewards as _compute_rewards

    return _compute_rewards(*args, **kwargs)


def init_train_state(*args, **kwargs):
    from .train_state import init_train_state as _init_train_state

    return _init_train_state(*args, **kwargs)


def advance_train_state(*args, **kwargs):
    from .train_state import advance_train_state as _advance_train_state

    return _advance_train_state(*args, **kwargs)


def make_rng_grid(*args, **kwargs):
    from .rollout import make_rng_grid as _make_rng_grid

    return _make_rng_grid(*args, **kwargs)


def make_optimizer(*args, **kwargs):
    from .train_state import make_optimizer as _make_optimizer

    return _make_optimizer(*args, **kwargs)


def train_update(*args, **kwargs):
    from .train_state import train_update as _train_update

    return _train_update(*args, **kwargs)


__all__ = (
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
    "make_rng_grid",
    "make_optimizer",
    "FinalColumnMetrics",
    "LossConfig",
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
)
