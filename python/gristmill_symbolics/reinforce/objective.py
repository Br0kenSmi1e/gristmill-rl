from __future__ import annotations

import numpy as np

from .types import BaselineConfig, RewardConfig, TrainingError


def compute_rewards(final_metrics: object, config: RewardConfig) -> np.ndarray:
    if config.kind != "log_flops_improvement":
        raise TrainingError(f"unsupported reward kind {config.kind!r}")
    initial = np.asarray(final_metrics.initial_log_flops, dtype=np.float64)
    final = np.asarray(final_metrics.final_log_flops, dtype=np.float64)
    if initial.ndim != 1:
        raise TrainingError(
            f"initial_log_flops must be 1D, got shape {initial.shape}"
        )
    if final.ndim != 1:
        raise TrainingError(f"final_log_flops must be 1D, got shape {final.shape}")
    if initial.shape != final.shape:
        raise TrainingError(
            "initial_log_flops and final_log_flops shapes differ: "
            f"{initial.shape} != {final.shape}"
        )
    for field_name in ("stopped", "max_steps"):
        if hasattr(final_metrics, field_name):
            field_value = getattr(final_metrics, field_name)
            if field_value is None:
                continue
            field_array = np.asarray(field_value)
            if field_array.shape != initial.shape:
                raise TrainingError(
                    f"{field_name} shape {field_array.shape} does not match "
                    f"reward shape {initial.shape}"
                )
    reward = initial - final
    if not bool(np.all(np.isfinite(reward))):
        raise TrainingError("reward contains non-finite values")
    return reward.astype(np.float64, copy=False)


def compute_advantages(reward: np.ndarray, config: BaselineConfig) -> np.ndarray:
    values = np.asarray(reward, dtype=np.float64)
    if values.ndim != 1:
        raise TrainingError(f"reward must be 1D, got shape {values.shape}")
    if values.size == 0:
        raise TrainingError("reward must contain at least one sample")
    baseline = np.mean(values, dtype=np.float64)
    advantage = values - baseline
    if config.standardize:
        std = np.std(advantage, dtype=np.float64)
        advantage = (advantage - np.mean(advantage, dtype=np.float64)) / (
            std + config.epsilon
        )
    if not bool(np.all(np.isfinite(advantage))):
        raise TrainingError("advantage contains non-finite values")
    return advantage.astype(np.float64, copy=False)
