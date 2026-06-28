import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, RewriteStateRow, TensorComputation
from gristmill_symbolics._training import TrainingError
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer
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
    def __init__(self, *, final_log_flops, logp, grad_logp, batch_size=2, metrics=None):
        self.batch_size = batch_size
        self.final_log_flops = final_log_flops
        self.logp = logp
        self.grad_logp = grad_logp
        self.metrics = metrics or {"stopped": np.asarray([False, True], dtype=bool)}
        self.calls = []

    def sample_with_logp_grad(self, params, rng, row):
        self.calls.append((params, rng, row))
        return FakeOutRow(self.final_log_flops), self.logp, self.grad_logp, self.metrics


def _simple_params():
    return {"w": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}


def _zero_grad_logp(params, batch_size):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((batch_size, *leaf.shape), dtype=leaf.dtype),
        params,
    )


def _trainer(**overrides):
    values = {"batch_size": 2, "learning_rate": 1.0e-2}
    values.update(overrides)
    return ReinforceTrainer(**values)


def test_reinforce_trainer_calls_model_and_updates_from_model_outputs():
    params = _simple_params()
    trainer = _trainer()
    opt_state = trainer.init_opt_state(params)
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([1.0, -1.0], dtype=np.float64)
    grad_logp = {"w": jnp.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=jnp.float32)}
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([-0.5, -1.5], dtype=jnp.float32),
        grad_logp=grad_logp,
    )

    new_params, new_opt_state, metrics = trainer.update(
        params,
        opt_state,
        batch,
        model,
        jax.random.PRNGKey(0),
    )

    assert len(model.calls) == 1
    _called_params, _called_rng, called_row = model.calls[0]
    assert isinstance(called_row, RewriteStateRow)
    assert np.asarray(called_row.log_total_flops()).shape == (2,)
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
    trainer = _trainer(standardize_baseline=True, baseline_epsilon=1.0e-12)
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([2.0, 8.0], dtype=np.float64)
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([1.0, 3.0], dtype=jnp.float32),
        grad_logp=_zero_grad_logp(params, 2),
    )

    _new_params, _new_opt_state, metrics = trainer.update(
        params,
        trainer.init_opt_state(params),
        batch,
        model,
        jax.random.PRNGKey(0),
    )

    assert metrics["reward_mean"] == pytest.approx(5.0)
    assert metrics["reward_std"] == pytest.approx(3.0)
    assert metrics["surrogate_loss"] == pytest.approx(-1.0)


def test_reinforce_trainer_validates_batch_length_before_model_call():
    params = _simple_params()
    trainer = _trainer()
    model = FakeModel(
        final_log_flops=np.asarray([0.0]),
        logp=jnp.asarray([0.0], dtype=jnp.float32),
        grad_logp=_zero_grad_logp(params, 1),
    )

    with pytest.raises(TrainingError, match="batch length"):
        trainer.update(
            params,
            trainer.init_opt_state(params),
            [_state_from_json(actionable_json())],
            model,
            jax.random.PRNGKey(0),
        )
    assert model.calls == []


def test_reinforce_trainer_rejects_unsupported_reward_kind():
    with pytest.raises(TrainingError, match="unsupported reward kind"):
        _trainer(reward_kind="unsupported")


@pytest.mark.parametrize(
    "logp",
    [
        jnp.asarray([0, 0], dtype=jnp.int32),
        jnp.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex64),
    ],
)
def test_reinforce_trainer_rejects_non_floating_logp(logp):
    params = _simple_params()
    trainer = _trainer()
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([1.0, -1.0], dtype=np.float64)
    model = FakeModel(
        final_log_flops=final,
        logp=logp,
        grad_logp=_zero_grad_logp(params, 2),
    )

    with pytest.raises(TrainingError, match="logp"):
        trainer.update(
            params,
            trainer.init_opt_state(params),
            batch,
            model,
            jax.random.PRNGKey(0),
        )


@pytest.mark.parametrize(
    "grad_logp",
    [
        {"w": jnp.zeros((2, 2), dtype=jnp.int32)},
        {"w": jnp.zeros((2, 2), dtype=jnp.complex64)},
    ],
)
def test_reinforce_trainer_rejects_non_floating_grad_logp(grad_logp):
    params = _simple_params()
    trainer = _trainer()
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([1.0, -1.0], dtype=np.float64)
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        grad_logp=grad_logp,
    )

    with pytest.raises(TrainingError, match="grad_logp"):
        trainer.update(
            params,
            trainer.init_opt_state(params),
            batch,
            model,
            jax.random.PRNGKey(0),
        )


def test_reinforce_trainer_rejects_non_floating_params():
    params = {"w": jnp.asarray([1, -2], dtype=jnp.int32)}
    trainer = _trainer()
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([1.0, -1.0], dtype=np.float64)
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        grad_logp={"w": jnp.zeros((2, 2), dtype=jnp.float32)},
    )

    with pytest.raises(TrainingError, match="params"):
        trainer.update(
            params,
            trainer.init_opt_state(_simple_params()),
            batch,
            model,
            jax.random.PRNGKey(0),
        )


def test_reinforce_trainer_accepts_scalar_floating_param_leaf():
    params = {"w": jnp.asarray(1.0, dtype=jnp.float32)}
    trainer = _trainer()
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([1.0, -1.0], dtype=np.float64)
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        grad_logp={"w": jnp.asarray([1.0, -1.0], dtype=jnp.float32)},
    )

    new_params, _new_opt_state, metrics = trainer.update(
        params,
        trainer.init_opt_state(params),
        batch,
        model,
        jax.random.PRNGKey(0),
    )

    assert new_params["w"].shape == ()
    assert metrics["params_changed"] is True


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
            {"w": jnp.asarray(0.0, dtype=jnp.float32)},
            "leading dimension|grad_logp",
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
    trainer = _trainer()
    batch = _batch()
    final = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    model = FakeModel(final_log_flops=final, logp=logp, grad_logp=grad_logp)

    with pytest.raises(TrainingError, match=message):
        trainer.update(
            params,
            trainer.init_opt_state(params),
            batch,
            model,
            jax.random.PRNGKey(0),
        )
