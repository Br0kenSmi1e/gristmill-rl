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
    RewardConfig,
    TrainingError,
    validate_trainer_config,
)


def _as_jax_array(name: str, value):
    try:
        return jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"{name} must be array-like") from exc


def _validate_real_floating_dtype(name: str, values) -> None:
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TrainingError(
            f"{name} must have real floating dtype, got {values.dtype}"
        )


def _validate_logp(logp, batch_size: int):
    values = _as_jax_array("logp", logp)
    _validate_real_floating_dtype("logp", values)
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
        param_leaf = _as_jax_array("params leaf", param_leaf)
        grad_leaf = _as_jax_array("grad_logp leaf", grad_leaf)
        _validate_real_floating_dtype("params leaves", param_leaf)
        _validate_real_floating_dtype("grad_logp leaves", grad_leaf)
        if grad_leaf.ndim == 0:
            raise TrainingError(
                "grad_logp floating leaves must have leading dimension "
                f"{batch_size}, got {grad_leaf.shape}"
            )
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


def _as_float64_array(name: str, values):
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"{name} must be numeric") from exc


def _compute_reward(
    initial_log_flops,
    final_log_flops,
    config: RewardConfig,
) -> np.ndarray:
    if config.kind != "log_flops_improvement":
        raise TrainingError(f"unsupported reward kind {config.kind!r}")
    initial = _as_float64_array("initial_log_flops", initial_log_flops)
    final = _as_float64_array("final_log_flops", final_log_flops)
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
    reward = initial - final
    if not bool(np.all(np.isfinite(reward))):
        raise TrainingError("reward contains non-finite values")
    return reward.astype(np.float64, copy=False)


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

        initial_log_flops = [state.log_total_flops() for state in initial_states]
        row = RewriteStateRow.from_states(initial_states)
        out_row, logp, grad_logp, _model_metrics = model.sample_with_logp_grad(
            params,
            rng,
            row,
            config,
        )
        logp = _validate_logp(logp, config.batch_size)
        grad_logp = _validate_grad_logp(params, grad_logp, config.batch_size)

        raw_final_log_flops = out_row.log_total_flops()
        reward = _compute_reward(
            initial_log_flops,
            raw_final_log_flops,
            config.reward_config,
        )
        final_log_flops = _as_float64_array("final_log_flops", raw_final_log_flops)
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
