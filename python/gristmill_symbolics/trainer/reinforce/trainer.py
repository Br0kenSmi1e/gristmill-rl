from __future__ import annotations

from collections.abc import Sequence
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gristmill_symbolics import RewriteState, RewriteStateRow
from gristmill_symbolics._training import TrainingError

from .objective import compute_advantages


def _make_optimizer(
    *,
    learning_rate: float,
    b1: float,
    b2: float,
    eps: float,
) -> optax.GradientTransformation:
    if not (math.isfinite(learning_rate) and learning_rate > 0.0):
        raise TrainingError("learning_rate must be finite and positive")
    if not (math.isfinite(b1) and 0.0 <= b1 < 1.0):
        raise TrainingError("b1 must be finite and satisfy 0.0 <= b1 < 1.0")
    if not (math.isfinite(b2) and 0.0 <= b2 < 1.0):
        raise TrainingError("b2 must be finite and satisfy 0.0 <= b2 < 1.0")
    if not (math.isfinite(eps) and eps > 0.0):
        raise TrainingError("eps must be finite and positive")
    return optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)


def _finite_float(name: str, value, *, error_message: str | None = None) -> float:
    message = error_message or f"{name} must be a finite float"
    if isinstance(value, (bool, np.bool_)):
        raise TrainingError(message)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingError(message) from exc
    if not math.isfinite(result):
        raise TrainingError(message)
    return result


def _positive_finite_float(name: str, value, *, error_message: str) -> float:
    result = _finite_float(name, value, error_message=error_message)
    if result <= 0.0:
        raise TrainingError(error_message)
    return result


def _optimizer_float(name: str, value, *, error_message: str) -> float:
    return _finite_float(name, value, error_message=error_message)


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int:
        raise TrainingError(f"{name} must be an int")
    if value <= 0:
        raise TrainingError(f"{name} must be positive")
    return value


def _exact_bool(name: str, value) -> bool:
    if type(value) is not bool:
        raise TrainingError(f"{name} must be a bool")
    return value


def _validate_reward_kind(reward_kind: str) -> str:
    if reward_kind != "log_flops_improvement":
        raise TrainingError(f"unsupported reward kind {reward_kind!r}")
    return reward_kind


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
    reward_kind: str,
) -> np.ndarray:
    if reward_kind != "log_flops_improvement":
        raise TrainingError(f"unsupported reward kind {reward_kind!r}")
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
                    "updated model parameters contain non-finite values"
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


class ReinforceTrainer:
    def __init__(
        self,
        *,
        batch_size: int,
        learning_rate: float = 1.0e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1.0e-8,
        reward_kind: str = "log_flops_improvement",
        standardize_baseline: bool = False,
        baseline_epsilon: float = 1.0e-8,
    ):
        self._batch_size = _positive_int("batch_size", batch_size)
        self._learning_rate = _optimizer_float(
            "learning_rate",
            learning_rate,
            error_message="learning_rate must be finite and positive",
        )
        self._b1 = _optimizer_float(
            "b1",
            b1,
            error_message="b1 must be finite and satisfy 0.0 <= b1 < 1.0",
        )
        self._b2 = _optimizer_float(
            "b2",
            b2,
            error_message="b2 must be finite and satisfy 0.0 <= b2 < 1.0",
        )
        self._eps = _optimizer_float(
            "eps",
            eps,
            error_message="eps must be finite and positive",
        )
        self._reward_kind = _validate_reward_kind(reward_kind)
        self._standardize_baseline = _exact_bool(
            "standardize_baseline",
            standardize_baseline,
        )
        self._baseline_epsilon = _positive_finite_float(
            "baseline_epsilon",
            baseline_epsilon,
            error_message="baseline_epsilon must be finite and positive",
        )
        self._optimizer()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def constructor_kwargs(self) -> dict[str, object]:
        return {
            "batch_size": self._batch_size,
            "learning_rate": self._learning_rate,
            "b1": self._b1,
            "b2": self._b2,
            "eps": self._eps,
            "reward_kind": self._reward_kind,
            "standardize_baseline": self._standardize_baseline,
            "baseline_epsilon": self._baseline_epsilon,
        }

    def _optimizer(self) -> optax.GradientTransformation:
        return _make_optimizer(
            learning_rate=self._learning_rate,
            b1=self._b1,
            b2=self._b2,
            eps=self._eps,
        )

    def init_opt_state(self, params):
        return self._optimizer().init(params)

    def update(
        self,
        params,
        opt_state,
        batch: Sequence[RewriteState],
        model,
        rng,
    ):
        initial_states = list(batch)
        if len(initial_states) != self._batch_size:
            raise TrainingError(
                f"batch length {len(initial_states)} differs from "
                f"batch_size {self._batch_size}"
            )
        if getattr(model, "batch_size", self._batch_size) != self._batch_size:
            raise TrainingError("model batch_size must match trainer batch_size")

        initial_log_flops = [state.log_total_flops() for state in initial_states]
        row = RewriteStateRow.from_states(initial_states)
        out_row, logp, grad_logp, _model_metrics = model.sample_with_logp_grad(
            params,
            rng,
            row,
        )
        logp = _validate_logp(logp, self._batch_size)
        grad_logp = _validate_grad_logp(params, grad_logp, self._batch_size)

        raw_final_log_flops = out_row.log_total_flops()
        reward = _compute_reward(
            initial_log_flops,
            raw_final_log_flops,
            self._reward_kind,
        )
        final_log_flops = _as_float64_array("final_log_flops", raw_final_log_flops)
        advantage = compute_advantages(
            reward,
            standardize=self._standardize_baseline,
            epsilon=self._baseline_epsilon,
        )

        grads = _reinforce_grad_loss(grad_logp, advantage)
        surrogate_loss = _surrogate_loss(logp, advantage)
        if not bool(np.isfinite(np.asarray(surrogate_loss))):
            raise TrainingError("surrogate_loss is non-finite")

        updates, new_opt_state = self._optimizer().update(grads, opt_state, params)
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
