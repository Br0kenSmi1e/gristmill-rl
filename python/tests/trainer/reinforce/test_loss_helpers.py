import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics._training import TrainingError
from gristmill_symbolics.trainer.reinforce.trainer import (
    _validate_finite_params,
    _reinforce_grad_loss,
    _surrogate_loss,
)


def test_reinforce_grad_loss_is_negative_mean_advantage_times_trajectory_grad():
    trajectory_grad = {
        "leaf": jnp.asarray(
            [
                [1.0, 2.0],
                [3.0, 5.0],
                [-7.0, 11.0],
            ],
            dtype=jnp.float32,
        )
    }
    advantage = np.asarray([2.0, -1.0, 0.5], dtype=np.float64)

    grad_loss = _reinforce_grad_loss(trajectory_grad, advantage)

    expected = -jnp.mean(
        jnp.asarray(advantage, dtype=jnp.float32)[:, None] * trajectory_grad["leaf"],
        axis=0,
    )
    assert jnp.allclose(grad_loss["leaf"], expected)


def test_surrogate_loss_uses_trajectory_logp_diagnostic_only():
    logp = jnp.asarray([-1.0, -2.0, -4.0], dtype=jnp.float32)
    advantage = np.asarray([2.0, -1.0, 0.5], dtype=np.float64)

    assert float(_surrogate_loss(logp, advantage)) == pytest.approx(
        float(-jnp.mean(jnp.asarray(advantage, dtype=jnp.float32) * logp))
    )


def test_validate_finite_params_rejects_non_finite_updated_param_leaf():
    with pytest.raises(TrainingError, match="updated model parameters"):
        _validate_finite_params({"w": jnp.asarray([1.0, jnp.nan])})
