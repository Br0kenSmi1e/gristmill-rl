"""On-policy REINFORCE trainer over the row rewrite environment."""

from .checkpoint import load_checkpoint, save_checkpoint
from .types import (
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


def compute_advantages(*args, **kwargs):
    from .objective import compute_advantages as _compute_advantages

    return _compute_advantages(*args, **kwargs)


def compute_rewards(*args, **kwargs):
    from .objective import compute_rewards as _compute_rewards

    return _compute_rewards(*args, **kwargs)


def init_train_state(*args, **kwargs):
    from .train_state import init_train_state as _init_train_state

    return _init_train_state(*args, **kwargs)


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
    "BaselineConfig",
    "CheckpointData",
    "compute_advantages",
    "compute_rewards",
    "init_train_state",
    "load_checkpoint",
    "make_rng_grid",
    "make_optimizer",
    "FinalColumnMetrics",
    "LossConfig",
    "OptimizerConfig",
    "PolicyState",
    "RewardConfig",
    "RolloutConfig",
    "save_checkpoint",
    "train_update",
    "TrainState",
    "TrainingError",
    "UpdateMetrics",
)
