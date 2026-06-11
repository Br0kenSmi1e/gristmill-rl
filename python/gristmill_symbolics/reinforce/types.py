from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np

from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.policy.types import ActionChoiceTree, TokenTree

CASE_ALREADY_FINISHED = 0
CASE_STOP = 1
CASE_EMPTY_ACTION_SPACE = 2
CASE_VALID_ACTION = 3

DECISION_TARGET = 0
DECISION_ACTION = 1

CHECKPOINT_SCHEMA_VERSION = 1
TOKENIZER_SCHEMA_VERSION = 1


class TrainingError(RuntimeError):
    """Raised when Phase 3 trainer contracts are violated."""


@dataclass(frozen=True)
class PolicyState:
    config: PolicyConfig
    params: dict[str, object]


@dataclass(frozen=True)
class RolloutConfig:
    batch_size: int
    max_steps: int
    seed: int = 0


@dataclass(frozen=True)
class RewardConfig:
    kind: Literal["log_flops_improvement"] = "log_flops_improvement"


@dataclass(frozen=True)
class BaselineConfig:
    standardize: bool = False
    epsilon: float = 1.0e-8


@dataclass(frozen=True)
class LossConfig:
    require_scored_terms: bool = True


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1.0e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1.0e-8


@dataclass(frozen=True)
class TrainState:
    policy: PolicyState
    optimizer_config: OptimizerConfig
    opt_state: object
    root_key: jax.Array
    update_index: int


@dataclass(frozen=True)
class FinalColumnMetrics:
    initial_log_flops: np.ndarray
    final_log_flops: np.ndarray
    stopped: np.ndarray
    max_steps: np.ndarray


@dataclass(frozen=True)
class RolloutTable:
    state_tokens: TokenTree
    state_token_mask: jax.Array
    target_def_mask: jax.Array
    target_choice: jax.Array
    target_score_mask: jax.Array
    selected_def_index: jax.Array
    action_space_tokens: TokenTree
    action_space_token_mask: jax.Array
    action_choice: ActionChoiceTree
    action_score_mask: jax.Array
    step_case: jax.Array
    sampled_target_logp: jax.Array
    sampled_action_logp: jax.Array


@dataclass(frozen=True)
class ScoreOutputs:
    target_logp: jax.Array
    action_logp: jax.Array


@dataclass(frozen=True)
class LossDiagnostics:
    column_logp_sum: jax.Array
    target_score_count: int
    action_score_count: int
    target_logp_mean: float
    action_logp_mean: float


@dataclass(frozen=True)
class UpdateMetrics:
    update_index: int
    batch_size: int
    max_steps: int
    initial_log_flops_mean: float
    final_log_flops_mean: float
    final_log_flops_best: float
    reward_mean: float
    reward_std: float
    advantage_mean: float
    advantage_std: float
    valid_action_count: int
    stop_count: int
    empty_action_space_count: int
    finished_count: int
    max_steps_count: int
    target_score_count: int
    action_score_count: int
    loss: float
    target_logp_mean: float
    action_logp_mean: float
    params_changed: bool


@dataclass(frozen=True)
class CheckpointData:
    train_state: TrainState
    rollout_config: RolloutConfig
    reward_config: RewardConfig
    baseline_config: BaselineConfig
    loss_config: LossConfig
    recent_metrics: tuple[UpdateMetrics, ...]


def validate_rollout_config(config: RolloutConfig) -> None:
    if config.batch_size <= 0:
        raise TrainingError("batch_size must be positive")
    if config.max_steps <= 0:
        raise TrainingError("max_steps must be positive")


def validate_policy_state(policy: PolicyState) -> None:
    if not isinstance(policy.config, PolicyConfig):
        raise TrainingError("policy.config must be a PolicyConfig")
    if not isinstance(policy.params, dict):
        raise TrainingError("policy.params must be a dict pytree")
