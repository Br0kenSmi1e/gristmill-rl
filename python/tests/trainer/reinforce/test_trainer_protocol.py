import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, RewriteStateRow, TensorComputation
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer
from gristmill_symbolics.trainer.reinforce.objective import compute_rewards
from gristmill_symbolics._training import TrainingError
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
    batch_size = 2

    def __init__(self, *, final_log_flops, logp, grad_logp):
        self.final_log_flops = final_log_flops
        self.logp = logp
        self.grad_logp = grad_logp
        self.calls = []

    def sample_with_logp_grad(self, params, rng, row):
        self.calls.append((params, rng, row))
        return (
            FakeOutRow(self.final_log_flops),
            self.logp,
            self.grad_logp,
            {"stopped": np.asarray([False, True], dtype=bool)},
        )


def _simple_params():
    return {"w": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}


def _zero_grad_logp(params, batch_size):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((batch_size, *leaf.shape), dtype=leaf.dtype),
        params,
    )


def test_reinforce_trainer_protocol_is_config_free():
    trainer = ReinforceTrainer(batch_size=2, learning_rate=1.0e-2)

    assert trainer.batch_size == 2
    assert list(inspect.signature(trainer.init_opt_state).parameters) == ["params"]
    assert list(inspect.signature(trainer.update).parameters) == [
        "params",
        "opt_state",
        "batch",
        "model",
        "rng",
    ]


def test_reinforce_trainer_constructor_kwargs_round_trip():
    trainer = ReinforceTrainer(
        batch_size=2,
        learning_rate=2.0e-3,
        b1=0.8,
        b2=0.99,
        eps=1.0e-7,
        reward_kind="log_flops_improvement",
        standardize_baseline=True,
        baseline_epsilon=1.0e-6,
    )

    round_tripped = ReinforceTrainer(**trainer.constructor_kwargs())

    assert round_tripped.constructor_kwargs() == trainer.constructor_kwargs()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": float("nan")},
        {"learning_rate": None},
        {"learning_rate": np.bool_(True)},
        {"reward_kind": "unsupported"},
        {"standardize_baseline": "yes"},
        {"baseline_epsilon": 0.0},
    ],
)
def test_reinforce_trainer_constructor_rejects_invalid_values(kwargs):
    with pytest.raises(TrainingError):
        ReinforceTrainer(batch_size=2, **kwargs)


def test_reinforce_trainer_calls_model_without_config_and_updates():
    params = _simple_params()
    trainer = ReinforceTrainer(batch_size=2, learning_rate=1.0e-2)
    opt_state = trainer.init_opt_state(params)
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([1.0, -1.0], dtype=np.float64)
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([-0.5, -1.5], dtype=jnp.float32),
        grad_logp={"w": jnp.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=jnp.float32)},
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


def test_reinforce_trainer_validates_batch_length_before_model_call():
    params = _simple_params()
    trainer = ReinforceTrainer(batch_size=2, learning_rate=1.0e-2)
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


def test_reinforce_reward_rejects_empty_metric_batch():
    final_metrics = type(
        "FinalMetrics",
        (),
        {
            "initial_log_flops": np.asarray([], dtype=np.float64),
            "final_log_flops": np.asarray([], dtype=np.float64),
        },
    )()

    with pytest.raises(TrainingError, match="reward must contain at least one sample"):
        compute_rewards(final_metrics, reward_kind="log_flops_improvement")
