"""On-policy REINFORCE trainer over the row rewrite environment."""

from .objective import compute_advantages, compute_rewards, reinforce_loss, score_rollout
from .rollout import collect_rollout_batch, make_rng_grid
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
    "make_rng_grid",
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
    "TrainState",
    "TrainingError",
    "UpdateMetrics",
)
