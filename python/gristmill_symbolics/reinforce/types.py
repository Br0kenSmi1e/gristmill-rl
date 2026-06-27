from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np

from gristmill_symbolics.policy import PolicyConfig

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
    state_token_pad_to: int | None = None
    action_token_pad_to: int | None = None
    definition_pad_to: int | None = None
    static_policy_batch: bool = False


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
class CurrentTransformerModelConfig:
    policy_config: PolicyConfig
    batch_size: int
    max_steps: int
    state_token_pad_to: int
    action_token_pad_to: int
    definition_pad_to: int


@dataclass(frozen=True)
class ReinforceTrainerConfig:
    batch_size: int
    optimizer_config: OptimizerConfig
    reward_config: RewardConfig = RewardConfig()
    baseline_config: BaselineConfig = BaselineConfig()


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
class UpdateMetrics:
    update_index: int
    batch_size: int
    max_steps: int
    initial_log_flops_mean: float
    final_log_flops_mean: float
    final_log_flops_best: float
    reward_mean: float
    reward_std: float
    reward_stderr: float
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
    objective_loss_mean: float
    objective_loss_stderr: float
    surrogate_loss: float
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


def _validate_optional_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if type(value) is not int:
        raise TrainingError(f"{name} must be an int or None")
    if value <= 0:
        raise TrainingError(f"{name} must be positive")


def _validate_positive_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TrainingError(f"{name} must be an int")
    if value <= 0:
        raise TrainingError(f"{name} must be positive")


def validate_model_config(config: CurrentTransformerModelConfig) -> None:
    if not isinstance(config.policy_config, PolicyConfig):
        raise TrainingError("policy_config must be a PolicyConfig")
    _validate_positive_int("batch_size", config.batch_size)
    _validate_positive_int("max_steps", config.max_steps)
    _validate_positive_int("state_token_pad_to", config.state_token_pad_to)
    _validate_positive_int("action_token_pad_to", config.action_token_pad_to)
    _validate_positive_int("definition_pad_to", config.definition_pad_to)


def validate_trainer_config(config: ReinforceTrainerConfig) -> None:
    _validate_positive_int("batch_size", config.batch_size)
    if not isinstance(config.optimizer_config, OptimizerConfig):
        raise TrainingError("optimizer_config must be an OptimizerConfig")
    if not isinstance(config.reward_config, RewardConfig):
        raise TrainingError("reward_config must be a RewardConfig")
    if not isinstance(config.baseline_config, BaselineConfig):
        raise TrainingError("baseline_config must be a BaselineConfig")


def validate_rollout_config(config: RolloutConfig) -> None:
    if type(config.batch_size) is not int:
        raise TrainingError("batch_size must be an int")
    if config.batch_size <= 0:
        raise TrainingError("batch_size must be positive")
    if type(config.max_steps) is not int:
        raise TrainingError("max_steps must be an int")
    if config.max_steps <= 0:
        raise TrainingError("max_steps must be positive")
    if type(config.seed) is not int:
        raise TrainingError("seed must be an int")
    _validate_optional_positive_int("state_token_pad_to", config.state_token_pad_to)
    _validate_optional_positive_int("action_token_pad_to", config.action_token_pad_to)
    _validate_optional_positive_int("definition_pad_to", config.definition_pad_to)
    if type(config.static_policy_batch) is not bool:
        raise TrainingError("static_policy_batch must be a bool")
    if config.static_policy_batch:
        if config.state_token_pad_to is None:
            raise TrainingError("static_policy_batch requires state_token_pad_to")
        if config.action_token_pad_to is None:
            raise TrainingError("static_policy_batch requires action_token_pad_to")
        if config.definition_pad_to is None:
            raise TrainingError("static_policy_batch requires definition_pad_to")


def validate_policy_state(policy: PolicyState) -> None:
    if not isinstance(policy, PolicyState):
        raise TrainingError("policy must be a PolicyState")
    if not isinstance(policy.config, PolicyConfig):
        raise TrainingError("policy.config must be a PolicyConfig")
    if not isinstance(policy.params, dict):
        raise TrainingError("policy.params must be a dict pytree")
