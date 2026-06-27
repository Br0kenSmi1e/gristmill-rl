from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gristmill_symbolics import RewriteState, RewriteStateRow

from .objective import compute_advantages
from .train_state import (
    _params_changed,
    _reinforce_grad_loss,
    _surrogate_loss,
    _validate_finite_params,
    make_optimizer,
)
from .types import (
    ReinforceTrainerConfig,
    TrainingError,
    validate_trainer_config,
)


def _validate_logp(logp, batch_size: int):
    values = jnp.asarray(logp)
    if values.shape != (batch_size,):
        raise TrainingError(f"logp must have shape {(batch_size,)}, got {values.shape}")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise TrainingError("logp contains non-finite values")
    return values


def _validate_grad_logp(params, grad_logp, batch_size: int):
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(grad_logp):
        raise TrainingError("grad_logp pytree must match params pytree")
    for param_leaf, grad_leaf in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(grad_logp),
        strict=True,
    ):
        param_leaf = jnp.asarray(param_leaf)
        grad_leaf = jnp.asarray(grad_leaf)
        if grad_leaf.shape[0] != batch_size:
            raise TrainingError(
                "grad_logp floating leaves must have leading dimension "
                f"{batch_size}, got {grad_leaf.shape}"
            )
        if grad_leaf.shape[1:] != param_leaf.shape:
            raise TrainingError(
                "grad_logp leaf shape after the sample axis must match params leaf "
                f"shape {param_leaf.shape}, got {grad_leaf.shape[1:]}"
            )
        if jnp.issubdtype(grad_leaf.dtype, jnp.floating):
            if not bool(jnp.all(jnp.isfinite(grad_leaf))):
                raise TrainingError("grad_logp contains non-finite values")
    return grad_logp


class ReinforceTrainer:
    def update(
        self,
        params,
        opt_state,
        batch: Sequence[RewriteState],
        model,
        rng,
        config: ReinforceTrainerConfig,
    ):
        validate_trainer_config(config)
        initial_states = list(batch)
        if len(initial_states) != config.batch_size:
            raise TrainingError(
                f"batch length {len(initial_states)} differs from "
                f"batch_size {config.batch_size}"
            )

        initial_log_flops = np.asarray(
            [state.log_total_flops() for state in initial_states],
            dtype=np.float64,
        )
        row = RewriteStateRow.from_states(initial_states)
        out_row, logp, grad_logp, _model_metrics = model.sample_with_logp_grad(
            params,
            rng,
            row,
            config,
        )
        logp = _validate_logp(logp, config.batch_size)
        grad_logp = _validate_grad_logp(params, grad_logp, config.batch_size)

        final_log_flops = np.asarray(out_row.log_total_flops(), dtype=np.float64)
        if final_log_flops.shape != initial_log_flops.shape:
            raise TrainingError(
                "final_log_flops shape does not match initial_log_flops shape: "
                f"{final_log_flops.shape} != {initial_log_flops.shape}"
            )
        reward = initial_log_flops - final_log_flops
        if not bool(np.all(np.isfinite(reward))):
            raise TrainingError("reward contains non-finite values")
        advantage = compute_advantages(reward, config.baseline_config)

        grads = _reinforce_grad_loss(grad_logp, advantage)
        surrogate_loss = _surrogate_loss(logp, advantage)
        if not bool(np.isfinite(np.asarray(surrogate_loss))):
            raise TrainingError("surrogate_loss is non-finite")

        optimizer = make_optimizer(config.optimizer_config)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        _validate_finite_params(new_params)

        return new_params, new_opt_state, {
            "reward_mean": float(np.mean(reward)),
            "reward_std": float(np.std(reward)),
            "objective_loss_mean": float(-np.mean(reward)),
            "surrogate_loss": float(np.asarray(surrogate_loss)),
            "final_flops_best": float(np.min(final_log_flops)),
            "params_changed": _params_changed(params, new_params),
        }
