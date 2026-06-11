from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gristmill_symbolics import RewriteState
from gristmill_symbolics.policy import PolicyConfig, init_policy_params

from .objective import (
    _reinforce_loss_value,
    compute_advantages,
    compute_rewards,
    reinforce_loss,
    score_rollout,
)
from .rollout import collect_rollout_batch
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
        raise TrainingError("learning_rate must be positive")
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
            if not bool(jnp.allclose(left, right)):
                return True
    return False


def _metric_counts(table):
    return {
        "valid_action_count": int(np.asarray(jnp.sum(table.step_case == 3))),
        "stop_count": int(np.asarray(jnp.sum(table.step_case == 1))),
        "empty_action_space_count": int(np.asarray(jnp.sum(table.step_case == 2))),
        "finished_count": int(np.asarray(jnp.sum(table.step_case == 0))),
        "target_score_count": int(np.asarray(jnp.sum(table.target_score_mask))),
        "action_score_count": int(np.asarray(jnp.sum(table.action_score_mask))),
    }


def train_update(
    state: TrainState,
    initial_states: Sequence[RewriteState],
    rollout_config: RolloutConfig,
    reward_config: RewardConfig = RewardConfig(),
    baseline_config: BaselineConfig = BaselineConfig(),
    loss_config: LossConfig = LossConfig(),
):
    table, final = collect_rollout_batch(
        state.policy,
        initial_states,
        rollout_config,
        update_index=state.update_index,
        root_key=state.root_key,
    )
    reward = compute_rewards(final, reward_config)
    advantage = compute_advantages(reward, baseline_config)

    def objective(params):
        scored_policy = replace(state.policy, params=params)
        scores = score_rollout(scored_policy, table, check_finite=False)
        loss, _column_logp_sum = _reinforce_loss_value(table, scores, advantage)
        return loss

    loss, grads = jax.value_and_grad(objective)(state.policy.params)
    if not bool(np.isfinite(np.asarray(loss))):
        raise TrainingError("loss is non-finite")
    concrete_scores = score_rollout(state.policy, table, check_finite=True)
    _concrete_loss, diagnostics = reinforce_loss(
        table,
        concrete_scores,
        advantage,
        loss_config,
    )
    optimizer = make_optimizer(state.optimizer_config)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.policy.params)
    new_params = optax.apply_updates(state.policy.params, updates)
    params_changed = _params_changed(state.policy.params, new_params)

    counts = _metric_counts(table)
    metrics = UpdateMetrics(
        update_index=state.update_index,
        batch_size=rollout_config.batch_size,
        max_steps=rollout_config.max_steps,
        initial_log_flops_mean=float(np.mean(final.initial_log_flops)),
        final_log_flops_mean=float(np.mean(final.final_log_flops)),
        final_log_flops_best=float(np.min(final.final_log_flops)),
        reward_mean=float(np.mean(reward)),
        reward_std=float(np.std(reward)),
        advantage_mean=float(np.mean(advantage)),
        advantage_std=float(np.std(advantage)),
        valid_action_count=counts["valid_action_count"],
        stop_count=counts["stop_count"],
        empty_action_space_count=counts["empty_action_space_count"],
        finished_count=counts["finished_count"],
        max_steps_count=int(np.sum(final.max_steps)),
        target_score_count=diagnostics.target_score_count,
        action_score_count=diagnostics.action_score_count,
        loss=float(np.asarray(loss)),
        target_logp_mean=diagnostics.target_logp_mean,
        action_logp_mean=diagnostics.action_logp_mean,
        params_changed=params_changed,
    )
    new_state = TrainState(
        policy=PolicyState(config=state.policy.config, params=new_params),
        optimizer_config=state.optimizer_config,
        opt_state=opt_state,
        root_key=state.root_key,
        update_index=state.update_index + 1,
    )
    return new_state, metrics, table
