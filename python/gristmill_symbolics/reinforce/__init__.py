"""On-policy REINFORCE trainer over the row rewrite environment."""

from .objective import compute_advantages, compute_rewards, reinforce_loss, score_rollout
from .rollout import collect_rollout_batch, make_rng_grid
from .train_state import init_train_state, make_optimizer, train_update
from .types import (
    BaselineConfig,
    CheckpointData,
    FinalColumnMetrics,
    LossConfig,
    LossDiagnostics,
    OptimizerConfig,
    PolicyState,
    RewardConfig,
    RolloutConfig,
    RolloutTable,
    ScoreOutputs,
    TrainState,
    TrainingError,
    UpdateMetrics,
)

__all__ = (
    "BaselineConfig",
    "CheckpointData",
    "collect_rollout_batch",
    "compute_advantages",
    "compute_rewards",
    "init_train_state",
    "make_rng_grid",
    "make_optimizer",
    "FinalColumnMetrics",
    "LossConfig",
    "LossDiagnostics",
    "OptimizerConfig",
    "PolicyState",
    "RewardConfig",
    "RolloutConfig",
    "RolloutTable",
    "ScoreOutputs",
    "reinforce_loss",
    "score_rollout",
    "train_update",
    "TrainState",
    "TrainingError",
    "UpdateMetrics",
)
