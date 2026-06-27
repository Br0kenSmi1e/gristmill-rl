from __future__ import annotations

from dataclasses import asdict
import pickle

import jax.numpy as jnp
import numpy as np

from gristmill_symbolics.policy import PolicyConfig

from .types import (
    CHECKPOINT_SCHEMA_VERSION,
    BaselineConfig,
    CheckpointData,
    CurrentTransformerModelConfig,
    OptimizerConfig,
    ReinforceTrainerConfig,
    RewardConfig,
    TrainState,
    TrainingError,
    UpdateMetrics,
    validate_training_configs,
)


def save_checkpoint(
    path,
    train_state: TrainState,
    *,
    model_config: CurrentTransformerModelConfig,
    trainer_config: ReinforceTrainerConfig,
    recent_metrics: tuple[UpdateMetrics, ...],
) -> None:
    validate_training_configs(model_config, trainer_config)
    model_config_payload = asdict(model_config)
    model_config_payload.pop("policy_config")
    trainer_config_payload = asdict(trainer_config)
    trainer_config_payload.pop("optimizer_config")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "policy_config": asdict(model_config.policy_config),
        "policy_params": train_state.params,
        "optimizer_config": asdict(trainer_config.optimizer_config),
        "optimizer_state": train_state.opt_state,
        "model_config": model_config_payload,
        "trainer_config": trainer_config_payload,
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

    try:
        policy_config = PolicyConfig(**payload["policy_config"])
        optimizer_config = OptimizerConfig(**payload["optimizer_config"])
        model_config = CurrentTransformerModelConfig(
            policy_config=policy_config,
            **payload["model_config"],
        )
        trainer_config_payload = payload["trainer_config"]
        trainer_config = ReinforceTrainerConfig(
            batch_size=trainer_config_payload["batch_size"],
            optimizer_config=optimizer_config,
            reward_config=RewardConfig(**trainer_config_payload["reward_config"]),
            baseline_config=BaselineConfig(**trainer_config_payload["baseline_config"]),
        )
        validate_training_configs(model_config, trainer_config)
        train_state = TrainState(
            params=payload["policy_params"],
            opt_state=payload["optimizer_state"],
            root_key=jnp.asarray(payload["root_key"], dtype=jnp.uint32),
            update_index=int(payload["update_index"]),
        )
        return CheckpointData(
            train_state=train_state,
            model_config=model_config,
            trainer_config=trainer_config,
            recent_metrics=tuple(
                UpdateMetrics(**metrics) for metrics in payload["recent_metrics"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingError(f"invalid checkpoint payload: {exc}") from exc
