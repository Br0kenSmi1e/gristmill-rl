from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax

from gristmill_symbolics.policy import PolicyConfig

DECISION_TARGET = 0
DECISION_ACTION = 1

CHECKPOINT_SCHEMA_VERSION = 2
TOKENIZER_SCHEMA_VERSION = 1


class TrainingError(RuntimeError):
    """Raised when Phase 3 trainer contracts are violated."""


@dataclass(frozen=True)
class RewardConfig:
    kind: Literal["log_flops_improvement"] = "log_flops_improvement"


@dataclass(frozen=True)
class BaselineConfig:
    standardize: bool = False
    epsilon: float = 1.0e-8


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
    params: object
    opt_state: object
    root_key: jax.Array
    update_index: int


@dataclass(frozen=True)
class UpdateMetrics:
    update_index: int
    batch_size: int
    reward_mean: float
    reward_std: float
    objective_loss_mean: float
    surrogate_loss: float
    final_flops_best: float
    params_changed: bool


@dataclass(frozen=True)
class CheckpointData:
    train_state: TrainState
    model_config: CurrentTransformerModelConfig
    trainer_config: ReinforceTrainerConfig
    recent_metrics: tuple[UpdateMetrics, ...]


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


def validate_training_configs(
    model_config: CurrentTransformerModelConfig,
    trainer_config: ReinforceTrainerConfig,
) -> None:
    validate_model_config(model_config)
    validate_trainer_config(trainer_config)
    if model_config.batch_size != trainer_config.batch_size:
        raise TrainingError(
            "model_config.batch_size must match trainer_config.batch_size"
        )
