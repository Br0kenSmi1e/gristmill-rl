from __future__ import annotations

from dataclasses import asdict
import pickle

import jax.numpy as jnp
import numpy as np

from gristmill_symbolics.policy import PolicyConfig

from .types import (
    CHECKPOINT_SCHEMA_VERSION,
    TOKENIZER_SCHEMA_VERSION,
    BaselineConfig,
    CheckpointData,
    LossConfig,
    OptimizerConfig,
    PolicyState,
    RewardConfig,
    RolloutConfig,
    TrainState,
    TrainingError,
    UpdateMetrics,
)


def save_checkpoint(
    path,
    train_state: TrainState,
    *,
    rollout_config: RolloutConfig,
    reward_config: RewardConfig,
    baseline_config: BaselineConfig,
    loss_config: LossConfig,
    recent_metrics: tuple[UpdateMetrics, ...],
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tokenizer_schema_version": TOKENIZER_SCHEMA_VERSION,
        "policy_config": asdict(train_state.policy.config),
        "policy_params": train_state.policy.params,
        "optimizer_config": asdict(train_state.optimizer_config),
        "optimizer_state": train_state.opt_state,
        "rollout_config": asdict(rollout_config),
        "reward_config": asdict(reward_config),
        "baseline_config": asdict(baseline_config),
        "loss_config": asdict(loss_config),
        "update_index": int(train_state.update_index),
        "root_key": np.asarray(train_state.root_key, dtype=np.uint32),
        "recent_metrics": tuple(asdict(metrics) for metrics in recent_metrics),
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path) -> CheckpointData:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise TrainingError("checkpoint payload must be a dict")

    schema_version = payload.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise TrainingError(
            "unsupported checkpoint schema "
            f"{schema_version}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )

    tokenizer_schema_version = payload.get("tokenizer_schema_version")
    if tokenizer_schema_version != TOKENIZER_SCHEMA_VERSION:
        raise TrainingError(
            "unsupported tokenizer schema "
            f"{tokenizer_schema_version}; expected {TOKENIZER_SCHEMA_VERSION}"
        )

    try:
        policy_config = PolicyConfig(**payload["policy_config"])
        optimizer_config = OptimizerConfig(**payload["optimizer_config"])
        train_state = TrainState(
            policy=PolicyState(
                config=policy_config,
                params=payload["policy_params"],
            ),
            optimizer_config=optimizer_config,
            opt_state=payload["optimizer_state"],
            root_key=jnp.asarray(payload["root_key"], dtype=jnp.uint32),
            update_index=int(payload["update_index"]),
        )
        return CheckpointData(
            train_state=train_state,
            rollout_config=RolloutConfig(**payload["rollout_config"]),
            reward_config=RewardConfig(**payload["reward_config"]),
            baseline_config=BaselineConfig(**payload["baseline_config"]),
            loss_config=LossConfig(**payload["loss_config"]),
            recent_metrics=tuple(
                UpdateMetrics(**metrics) for metrics in payload["recent_metrics"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingError(f"invalid checkpoint payload: {exc}") from exc
