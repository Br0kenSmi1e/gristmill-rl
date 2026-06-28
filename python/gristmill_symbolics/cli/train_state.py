from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from gristmill_symbolics._training import TrainingError
from gristmill_symbolics import RewriteState


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


def init_train_state(
    model,
    trainer,
    *,
    seed: int,
    update_index: int = 0,
) -> TrainState:
    _validate_matching_batch_sizes(model, trainer)
    root_key = jax.random.PRNGKey(int(seed))
    params_key = jax.random.fold_in(root_key, np.uint32(0xFFFFFFFF))
    params = model.init_params(params_key)
    return TrainState(
        params=params,
        opt_state=trainer.init_opt_state(params),
        root_key=root_key,
        update_index=int(update_index),
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
    model,
    trainer,
):
    _validate_matching_batch_sizes(model, trainer)
    rng = jax.random.fold_in(state.root_key, int(state.update_index))
    new_params, new_opt_state, trainer_metrics = trainer.update(
        state.params,
        state.opt_state,
        list(initial_states),
        model,
        rng,
    )
    metrics = UpdateMetrics(
        update_index=state.update_index,
        batch_size=trainer.batch_size,
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


def _validate_matching_batch_sizes(model, trainer) -> None:
    if model.batch_size != trainer.batch_size:
        raise TrainingError("model batch_size must match trainer batch_size")
