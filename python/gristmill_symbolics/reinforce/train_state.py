from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gristmill_symbolics import RewriteState
from gristmill_symbolics.policy import PolicyConfig, init_policy_params

from .types import (
    CurrentTransformerModelConfig,
    OptimizerConfig,
    ReinforceTrainerConfig,
    TrainState,
    TrainingError,
    UpdateMetrics,
    validate_training_configs,
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
    params = init_policy_params(policy_config, params_key)
    optimizer = make_optimizer(optimizer_config)
    return TrainState(
        params=params,
        opt_state=optimizer.init(params),
        root_key=root_key,
        update_index=int(update_index),
    )


class _ConfiguredModel:
    def __init__(self, model, model_config):
        self._model = model
        self._model_config = model_config

    def sample_with_logp_grad(self, params, rng, row, _trainer_config):
        return self._model.sample_with_logp_grad(
            params,
            rng,
            row,
            self._model_config,
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


def advance_train_state(
    state: TrainState,
    initial_states: Sequence[RewriteState],
    *,
    model_config: CurrentTransformerModelConfig,
    trainer_config: ReinforceTrainerConfig,
    model=None,
    trainer=None,
):
    validate_training_configs(model_config, trainer_config)
    if model is None:
        from .model import CurrentTransformerModel

        model = CurrentTransformerModel()
    if trainer is None:
        from .trainer import ReinforceTrainer

        trainer = ReinforceTrainer()

    rng = jax.random.fold_in(state.root_key, int(state.update_index))
    new_params, new_opt_state, trainer_metrics = trainer.update(
        state.params,
        state.opt_state,
        list(initial_states),
        _ConfiguredModel(model, model_config),
        rng,
        trainer_config,
    )
    metrics = UpdateMetrics(
        update_index=state.update_index,
        batch_size=trainer_config.batch_size,
        reward_mean=float(trainer_metrics["reward_mean"]),
        reward_std=float(trainer_metrics["reward_std"]),
        objective_loss_mean=float(trainer_metrics["objective_loss_mean"]),
        surrogate_loss=float(trainer_metrics["surrogate_loss"]),
        final_flops_best=float(trainer_metrics["final_flops_best"]),
        params_changed=bool(trainer_metrics["params_changed"]),
    )
    return (
        TrainState(
            params=new_params,
            opt_state=new_opt_state,
            root_key=state.root_key,
            update_index=state.update_index + 1,
        ),
        metrics,
    )


def train_update(
    state: TrainState,
    initial_states: Sequence[RewriteState],
    *,
    model_config: CurrentTransformerModelConfig,
    trainer_config: ReinforceTrainerConfig,
    model=None,
    trainer=None,
):
    return advance_train_state(
        state,
        initial_states,
        model_config=model_config,
        trainer_config=trainer_config,
        model=model,
        trainer=trainer,
    )
