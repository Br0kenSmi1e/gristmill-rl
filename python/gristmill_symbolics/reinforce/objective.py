from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gristmill_symbolics.policy import score_action, score_target

from .types import (
    BaselineConfig,
    FinalColumnMetrics,
    LossConfig,
    LossDiagnostics,
    PolicyState,
    RewardConfig,
    RolloutTable,
    ScoreOutputs,
    TrainingError,
)


def compute_rewards(final_metrics: FinalColumnMetrics, config: RewardConfig) -> np.ndarray:
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


def _masked_mean(values, mask) -> float:
    mask_np = np.asarray(mask, dtype=bool)
    if not bool(np.any(mask_np)):
        return 0.0
    values_np = np.asarray(values)
    return float(np.mean(values_np[mask_np]))


def _reinforce_loss_value(
    rollout: RolloutTable,
    scores: ScoreOutputs,
    advantage: np.ndarray,
):
    target_mask = jnp.asarray(rollout.target_score_mask, dtype=jnp.bool_)
    action_mask = jnp.asarray(rollout.action_score_mask, dtype=jnp.bool_)
    target_terms = jnp.where(target_mask, scores.target_logp, 0.0)
    action_terms = jnp.where(action_mask, scores.action_logp, 0.0)
    column_logp_sum = jnp.sum(target_terms + action_terms, axis=0)
    advantage_array = jax.lax.stop_gradient(jnp.asarray(advantage))
    return -jnp.mean(advantage_array * column_logp_sum), column_logp_sum


def reinforce_loss(
    rollout: RolloutTable,
    scores: ScoreOutputs,
    advantage: np.ndarray,
    config: LossConfig,
) -> tuple[jax.Array, LossDiagnostics]:
    target_mask = jnp.asarray(rollout.target_score_mask, dtype=jnp.bool_)
    action_mask = jnp.asarray(rollout.action_score_mask, dtype=jnp.bool_)
    target_count = int(np.asarray(jnp.sum(target_mask)))
    action_count = int(np.asarray(jnp.sum(action_mask)))
    if config.require_scored_terms and target_count + action_count == 0:
        raise TrainingError("no scored policy terms in rollout batch")

    advantage_np = np.asarray(advantage, dtype=np.float64)
    if advantage_np.ndim != 1:
        raise TrainingError(f"advantage must be 1D, got shape {advantage_np.shape}")
    if advantage_np.shape[0] != int(target_mask.shape[1]):
        raise TrainingError(
            f"advantage length {advantage_np.shape[0]} does not match "
            f"rollout width {target_mask.shape[1]}"
        )

    loss, column_logp_sum = _reinforce_loss_value(rollout, scores, advantage_np)
    if not bool(np.isfinite(np.asarray(loss))):
        raise TrainingError("loss is non-finite")

    diagnostics = LossDiagnostics(
        column_logp_sum=column_logp_sum,
        target_score_count=target_count,
        action_score_count=action_count,
        target_logp_mean=_masked_mean(scores.target_logp, target_mask),
        action_logp_mean=_masked_mean(scores.action_logp, action_mask),
    )
    return loss, diagnostics


def _slice_tree_2d(tree, step: int, sample: int):
    return jax.tree_util.tree_map(lambda value: value[step, sample], tree)


def _slice_action_choice_2d(choice, step: int, sample: int):
    return jax.tree_util.tree_map(lambda value: value[step, sample], choice)


def _finite_or_raise(name: str, value, step: int, sample: int, check_finite: bool):
    if not check_finite:
        return value
    if not bool(np.isfinite(np.asarray(value))):
        raise TrainingError(f"{name} logp is non-finite at step {step}, sample {sample}")
    return value


def score_rollout(
    policy: PolicyState,
    rollout: RolloutTable,
    *,
    check_finite: bool = True,
) -> ScoreOutputs:
    target_rows = []
    action_rows = []
    steps, samples = rollout.target_score_mask.shape
    if rollout.action_score_mask.shape != (steps, samples):
        raise TrainingError("target_score_mask and action_score_mask shapes differ")

    for step in range(int(steps)):
        target_values = []
        action_values = []
        for sample in range(int(samples)):
            target_scored = bool(np.asarray(rollout.target_score_mask[step, sample]))
            action_scored = bool(np.asarray(rollout.action_score_mask[step, sample]))
            if action_scored and not target_scored:
                raise TrainingError("action_score_mask=true without target_score_mask=true")

            if target_scored:
                target_logp = score_target(
                    policy.params,
                    _slice_tree_2d(rollout.state_tokens, step, sample),
                    rollout.state_token_mask[step, sample],
                    rollout.target_def_mask[step, sample],
                    rollout.target_choice[step, sample],
                )
                target_values.append(
                    _finite_or_raise("target", target_logp, step, sample, check_finite)
                )
            else:
                target_values.append(jnp.asarray(0.0, dtype=jnp.float32))

            if action_scored:
                action_logp = score_action(
                    policy.params,
                    _slice_tree_2d(rollout.state_tokens, step, sample),
                    rollout.state_token_mask[step, sample],
                    rollout.selected_def_index[step, sample],
                    _slice_tree_2d(rollout.action_space_tokens, step, sample),
                    rollout.action_space_token_mask[step, sample],
                    _slice_action_choice_2d(rollout.action_choice, step, sample),
                )
                action_values.append(
                    _finite_or_raise("action", action_logp, step, sample, check_finite)
                )
            else:
                action_values.append(jnp.asarray(0.0, dtype=jnp.float32))
        target_rows.append(jnp.stack(target_values, axis=0))
        action_rows.append(jnp.stack(action_values, axis=0))

    return ScoreOutputs(
        target_logp=jnp.stack(target_rows, axis=0),
        action_logp=jnp.stack(action_rows, axis=0),
    )
