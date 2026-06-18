from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gristmill_symbolics import RewriteState
from gristmill_symbolics.policy import PolicyConfig, init_policy_params

from .objective import compute_advantages, compute_rewards
from .rollout import _collect_streamed_rollout_gradients
from .types import (
    BaselineConfig,
    LossConfig,
    OptimizerConfig,
    PolicyState,
    RewardConfig,
    RolloutConfig,
    TrainState,
    TrainingError,
    UpdateMetrics,
)


def make_optimizer(config: OptimizerConfig) -> optax.GradientTransformation:
    if not (np.isfinite(config.learning_rate) and config.learning_rate > 0.0):
        raise TrainingError("learning_rate must be finite and positive")
    if not (np.isfinite(config.b1) and 0.0 <= config.b1 < 1.0):
        raise TrainingError("b1 must be finite and satisfy 0.0 <= b1 < 1.0")
    if not (np.isfinite(config.b2) and 0.0 <= config.b2 < 1.0):
        raise TrainingError("b2 must be finite and satisfy 0.0 <= b2 < 1.0")
    if not (np.isfinite(config.eps) and config.eps > 0.0):
        raise TrainingError("eps must be finite and positive")
    return optax.adam(
        learning_rate=config.learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
    )


def init_train_state(
    policy_config: PolicyConfig,
    optimizer_config: OptimizerConfig,
    *,
    seed: int,
    update_index: int = 0,
) -> TrainState:
    root_key = jax.random.PRNGKey(int(seed))
    # fold_in data is uint32; use the unsigned representation of -1.
    params_key = jax.random.fold_in(root_key, np.uint32(0xFFFFFFFF))
    policy = PolicyState(
        config=policy_config,
        params=init_policy_params(policy_config, params_key),
    )
    optimizer = make_optimizer(optimizer_config)
    return TrainState(
        policy=policy,
        optimizer_config=optimizer_config,
        opt_state=optimizer.init(policy.params),
        root_key=root_key,
        update_index=int(update_index),
    )


def _params_changed(before, after) -> bool:
    before_leaves = jax.tree_util.tree_leaves(before)
    after_leaves = jax.tree_util.tree_leaves(after)
    for left, right in zip(before_leaves, after_leaves, strict=True):
        if hasattr(left, "dtype") and jnp.issubdtype(left.dtype, jnp.floating):
            if not bool(jnp.array_equal(left, right)):
                return True
    return False


def _validate_finite_params(params) -> None:
    for leaf in jax.tree_util.tree_leaves(params):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            if not bool(jnp.all(jnp.isfinite(leaf))):
                raise TrainingError(
                    "updated policy parameters contain non-finite values"
                )


def _reinforce_grad_loss(trajectory_grad_logp, advantage):
    stopped_advantage = jax.lax.stop_gradient(
        jnp.asarray(advantage, dtype=jnp.float32)
    )

    def reduce_leaf(grad_leaf):
        scale = stopped_advantage.reshape(
            (stopped_advantage.shape[0],) + (1,) * (grad_leaf.ndim - 1)
        )
        return -jnp.mean(scale * grad_leaf, axis=0)

    return jax.tree_util.tree_map(reduce_leaf, trajectory_grad_logp)


def _surrogate_loss(trajectory_logp, advantage):
    stopped_advantage = jax.lax.stop_gradient(
        jnp.asarray(advantage, dtype=jnp.float32)
    )
    return -jnp.mean(stopped_advantage * trajectory_logp)


def train_update(
    state: TrainState,
    initial_states: Sequence[RewriteState],
    rollout_config: RolloutConfig,
    reward_config: RewardConfig = RewardConfig(),
    baseline_config: BaselineConfig = BaselineConfig(),
    loss_config: LossConfig = LossConfig(),
):
    # Stateful training owns rollout RNG through TrainState.root_key and
    # update_index; RolloutConfig.seed is retained for standalone rollout and
    # serialization compatibility.
    streamed = _collect_streamed_rollout_gradients(
        state.policy,
        initial_states,
        rollout_config,
        update_index=state.update_index,
        root_key=state.root_key,
    )
    reward = compute_rewards(streamed.final, reward_config)
    advantage = compute_advantages(reward, baseline_config)
    if (
        loss_config.require_scored_terms
        and streamed.target_score_count + streamed.action_score_count == 0
    ):
        raise TrainingError("no scored policy terms in rollout batch")

    grads = _reinforce_grad_loss(streamed.trajectory_grad_logp, advantage)
    surrogate_loss = _surrogate_loss(streamed.trajectory_logp, advantage)
    if not bool(np.isfinite(np.asarray(surrogate_loss))):
        raise TrainingError("loss is non-finite")

    optimizer = make_optimizer(state.optimizer_config)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.policy.params)
    new_params = optax.apply_updates(state.policy.params, updates)
    _validate_finite_params(new_params)
    params_changed = _params_changed(state.policy.params, new_params)

    reward_mean = float(np.mean(reward))
    reward_std = float(np.std(reward))
    reward_stderr = reward_std / float(np.sqrt(rollout_config.batch_size))
    objective_loss_mean = -reward_mean
    metrics = UpdateMetrics(
        update_index=state.update_index,
        batch_size=rollout_config.batch_size,
        max_steps=rollout_config.max_steps,
        initial_log_flops_mean=float(np.mean(streamed.final.initial_log_flops)),
        final_log_flops_mean=float(np.mean(streamed.final.final_log_flops)),
        final_log_flops_best=float(np.min(streamed.final.final_log_flops)),
        reward_mean=reward_mean,
        reward_std=reward_std,
        reward_stderr=reward_stderr,
        advantage_mean=float(np.mean(advantage)),
        advantage_std=float(np.std(advantage)),
        valid_action_count=streamed.valid_action_count,
        stop_count=streamed.stop_count,
        empty_action_space_count=streamed.empty_action_space_count,
        finished_count=streamed.finished_count,
        max_steps_count=int(np.sum(streamed.final.max_steps)),
        target_score_count=streamed.target_score_count,
        action_score_count=streamed.action_score_count,
        loss=objective_loss_mean,
        objective_loss_mean=objective_loss_mean,
        objective_loss_stderr=reward_stderr,
        surrogate_loss=float(np.asarray(surrogate_loss)),
        target_logp_mean=(
            streamed.target_logp_sum / streamed.target_score_count
            if streamed.target_score_count
            else 0.0
        ),
        action_logp_mean=(
            streamed.action_logp_sum / streamed.action_score_count
            if streamed.action_score_count
            else 0.0
        ),
        params_changed=params_changed,
    )
    new_state = TrainState(
        policy=PolicyState(config=state.policy.config, params=new_params),
        optimizer_config=state.optimizer_config,
        opt_state=opt_state,
        root_key=state.root_key,
        update_index=state.update_index + 1,
    )
    return new_state, metrics
