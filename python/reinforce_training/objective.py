from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from transformer_policy.batch import (
    PaddedTokenChoiceBatch,
    score_event_batch,
    trajectory_log_probs,
)


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-3


def create_optimizer(scorer, config: TrainConfig) -> nnx.Optimizer:
    if config.learning_rate <= 0.0 or not np.isfinite(config.learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    return nnx.Optimizer(
        scorer,
        optax.adam(config.learning_rate),
        wrt=nnx.Param,
    )


def rewards_and_advantages(final_log_flops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(final_log_flops, dtype=np.float32)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("final_log_flops must be a nonempty 1-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("final_log_flops must be finite")
    rewards = -values
    advantages = rewards - np.mean(rewards, dtype=np.float32)
    return rewards.astype(np.float32), advantages.astype(np.float32)


def _chosen_event_log_probs_for_gradient(
    logits: jax.Array,
    legal_mask: np.ndarray,
    chosen_index: np.ndarray,
) -> jax.Array:
    # Keep this tracer-safe for nnx.value_and_grad; the shared validator is NumPy-based.
    masked_logits = jnp.where(
        jnp.asarray(legal_mask, dtype=bool),
        logits,
        -jnp.inf,
    )
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    return jnp.take_along_axis(
        log_probs,
        jnp.asarray(chosen_index, dtype=jnp.int32)[:, None],
        axis=-1,
    ).squeeze(-1)


def reinforce_loss(
    scorer,
    batch: PaddedTokenChoiceBatch,
    *,
    advantages: np.ndarray,
    episode_count: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    advantage_values = jax.lax.stop_gradient(jnp.asarray(advantages, dtype=jnp.float32))
    logits = score_event_batch(scorer, batch)
    chosen_log_probs = _chosen_event_log_probs_for_gradient(
        logits,
        batch.legal_mask,
        batch.chosen_index,
    )
    per_episode = trajectory_log_probs(
        chosen_log_probs,
        jnp.asarray(batch.episode_id, dtype=jnp.int32),
        episode_count=episode_count,
    )
    loss = -jnp.mean(advantage_values * per_episode)
    return loss, {
        "loss": loss,
        "mean_trajectory_log_prob": jnp.mean(per_episode),
        "mean_event_log_prob": jnp.mean(chosen_log_probs),
    }


def _flat_param_values(module) -> list[np.ndarray]:
    state = nnx.state(module, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    values = []
    for leaf in leaves:
        value = getattr(leaf, "value", leaf)
        values.append(np.asarray(value).copy())
    return values


def _reinforce_loss_for_grad(
    scorer,
    batch: PaddedTokenChoiceBatch,
    advantages: np.ndarray,
    episode_count: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    return reinforce_loss(
        scorer,
        batch,
        advantages=advantages,
        episode_count=episode_count,
    )


def train_step(
    scorer,
    *,
    optimizer: nnx.Optimizer,
    batch: PaddedTokenChoiceBatch,
    advantages: np.ndarray,
    episode_count: int,
) -> dict[str, float | bool]:
    before = _flat_param_values(scorer)
    grad_fn = nnx.value_and_grad(_reinforce_loss_for_grad, has_aux=True)
    (loss, aux), grads = grad_fn(
        scorer,
        batch,
        advantages,
        episode_count,
    )
    optimizer.update(scorer, grads)
    after = _flat_param_values(scorer)
    params_changed = any(
        not np.array_equal(left, right)
        for left, right in zip(before, after, strict=True)
    )
    return {
        "loss": float(loss),
        "mean_trajectory_log_prob": float(aux["mean_trajectory_log_prob"]),
        "mean_event_log_prob": float(aux["mean_event_log_prob"]),
        "params_changed": params_changed,
    }
