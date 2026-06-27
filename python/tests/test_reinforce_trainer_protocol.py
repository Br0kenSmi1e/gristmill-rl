import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig, init_policy_params
from gristmill_symbolics.reinforce import (
    BaselineConfig,
    OptimizerConfig,
    ReinforceTrainer,
    ReinforceTrainerConfig,
    TrainingError,
    make_optimizer,
)
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _batch():
    return [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]


class FakeOutRow:
    def __init__(self, final_log_flops):
        self._final_log_flops = np.asarray(final_log_flops, dtype=np.float64)

    def log_total_flops(self):
        return self._final_log_flops


class FakeModel:
    def __init__(self, *, final_log_flops, logp, grad_logp, metrics=None):
        self.final_log_flops = final_log_flops
        self.logp = logp
        self.grad_logp = grad_logp
        self.metrics = metrics or {"stopped": np.asarray([False, True], dtype=bool)}
        self.calls = []

    def sample_with_logp_grad(self, params, rng, row, config):
        self.calls.append((params, rng, row, config))
        return FakeOutRow(self.final_log_flops), self.logp, self.grad_logp, self.metrics


def _simple_params():
    return {"w": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}


def _zero_grad_logp(params, batch_size):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((batch_size, *leaf.shape), dtype=leaf.dtype),
        params,
    )


def test_reinforce_trainer_calls_model_and_updates_from_model_outputs():
    params = _simple_params()
    optimizer = make_optimizer(OptimizerConfig(learning_rate=1.0e-2))
    opt_state = optimizer.init(params)
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([1.0, -1.0], dtype=np.float64)
    grad_logp = {"w": jnp.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=jnp.float32)}
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([-0.5, -1.5], dtype=jnp.float32),
        grad_logp=grad_logp,
    )
    config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
    )

    new_params, new_opt_state, metrics = ReinforceTrainer().update(
        params,
        opt_state,
        batch,
        model,
        jax.random.PRNGKey(0),
        config,
    )

    assert model.calls
    assert new_opt_state is not opt_state
    assert metrics["reward_mean"] == pytest.approx(0.0)
    assert metrics["reward_std"] == pytest.approx(1.0)
    assert metrics["objective_loss_mean"] == pytest.approx(-metrics["reward_mean"])
    assert np.isfinite(metrics["surrogate_loss"])
    assert metrics["final_flops_best"] == pytest.approx(float(np.min(final)))
    assert metrics["params_changed"] is True
    assert not jnp.array_equal(new_params["w"], params["w"])


def test_reinforce_trainer_standardizes_advantage_when_configured():
    params = _simple_params()
    optimizer = make_optimizer(OptimizerConfig(learning_rate=1.0e-2))
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([2.0, 4.0], dtype=np.float64)
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        grad_logp=_zero_grad_logp(params, 2),
    )
    config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
        baseline_config=BaselineConfig(standardize=True, epsilon=1.0e-12),
    )

    _new_params, _new_opt_state, metrics = ReinforceTrainer().update(
        params,
        optimizer.init(params),
        batch,
        model,
        jax.random.PRNGKey(0),
        config,
    )

    assert metrics["reward_mean"] == pytest.approx(3.0)
    assert np.isfinite(metrics["surrogate_loss"])


def test_reinforce_trainer_validates_batch_length_before_model_call():
    params = _simple_params()
    optimizer = make_optimizer(OptimizerConfig(learning_rate=1.0e-2))
    model = FakeModel(
        final_log_flops=np.asarray([0.0]),
        logp=jnp.asarray([0.0], dtype=jnp.float32),
        grad_logp=_zero_grad_logp(params, 1),
    )

    with pytest.raises(TrainingError, match="batch length"):
        ReinforceTrainer().update(
            params,
            optimizer.init(params),
            [_state_from_json(actionable_json())],
            model,
            jax.random.PRNGKey(0),
            ReinforceTrainerConfig(
                batch_size=2,
                optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
            ),
        )
    assert model.calls == []


@pytest.mark.parametrize(
    ("logp", "grad_logp", "message"),
    [
        (
            jnp.asarray([[0.0]], dtype=jnp.float32),
            _zero_grad_logp(_simple_params(), 2),
            "logp",
        ),
        (
            jnp.asarray([0.0, jnp.nan], dtype=jnp.float32),
            _zero_grad_logp(_simple_params(), 2),
            "logp",
        ),
        (
            jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            {"w": jnp.zeros((1, 2), dtype=jnp.float32)},
            "leading dimension",
        ),
        (
            jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            {"w": jnp.asarray([[0.0, 0.0], [jnp.inf, 0.0]], dtype=jnp.float32)},
            "grad_logp",
        ),
    ],
)
def test_reinforce_trainer_validates_model_output_protocol(logp, grad_logp, message):
    params = _simple_params()
    optimizer = make_optimizer(OptimizerConfig(learning_rate=1.0e-2))
    batch = _batch()
    final = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    model = FakeModel(final_log_flops=final, logp=logp, grad_logp=grad_logp)

    with pytest.raises(TrainingError, match=message):
        ReinforceTrainer().update(
            params,
            optimizer.init(params),
            batch,
            model,
            jax.random.PRNGKey(0),
            ReinforceTrainerConfig(
                batch_size=2,
                optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
            ),
        )
