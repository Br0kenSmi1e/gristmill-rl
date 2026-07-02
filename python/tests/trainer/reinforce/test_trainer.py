from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics._training import TrainingError
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer


@dataclass(frozen=True)
class FakeComp:
    log_flops: float

    def log_total_flops(self):
        return self.log_flops


@dataclass(frozen=True)
class FakeState:
    comp: FakeComp
    target_mask: jax.Array


class FakeStepwiseModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def sample_step(self, params, rng, states):
        self.calls.append((params, rng, states))
        return self.outputs[len(self.calls) - 1]


def _simple_params():
    return {"w": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}


def _states(values, stopped):
    return [
        FakeState(
            comp=FakeComp(value),
            target_mask=_target_mask(is_stopped),
        )
        for value, is_stopped in zip(values, stopped, strict=True)
    ]


def _target_mask(stopped):
    return jnp.asarray([False, False] if stopped else [True, True])


def _zero_grad_logp(params, batch_size):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((batch_size, *leaf.shape), dtype=leaf.dtype),
        params,
    )


def _trainer(**overrides):
    values = {"batch_size": 2, "max_steps": 3, "learning_rate": 1.0e-2}
    values.update(overrides)
    return ReinforceTrainer(**values)


def test_reinforce_trainer_rolls_until_all_states_stop_and_updates():
    params = _simple_params()
    trainer = _trainer()
    opt_state = trainer.init_opt_state(params)
    batch = _states([10.0, 20.0], [False, False])
    grad_1 = {"w": jnp.asarray([[1.0, 0.0], [0.0, 2.0]])}
    grad_2 = {"w": jnp.asarray([[3.0, 0.0], [0.0, 0.0]])}
    model = FakeStepwiseModel(
        [
            (
                _states([9.0, 25.0], [False, True]),
                jnp.asarray([0.1, 0.2], dtype=jnp.float32),
                grad_1,
            ),
            (
                _states([8.0, 25.0], [True, True]),
                jnp.asarray([0.3, 0.0], dtype=jnp.float32),
                grad_2,
            ),
        ]
    )

    new_params, new_opt_state, metrics = trainer.update(
        params,
        opt_state,
        batch,
        model,
        jax.random.PRNGKey(0),
    )

    assert len(model.calls) == 2
    assert model.calls[1][2] == model.outputs[0][0]
    assert new_opt_state is not opt_state
    assert metrics["reward_mean"] == pytest.approx(-1.5)
    assert metrics["reward_std"] == pytest.approx(3.5)
    assert metrics["surrogate_loss"] == pytest.approx(-0.35, abs=1.0e-6)
    assert metrics["final_flops_best"] == pytest.approx(8.0)
    assert metrics["params_changed"] is True
    assert new_params["w"][0] > params["w"][0]
    assert new_params["w"][1] < params["w"][1]


def test_reinforce_trainer_skips_model_when_initial_batch_is_terminal():
    params = _simple_params()
    trainer = _trainer()
    batch = _states([10.0, 20.0], [True, True])
    model = FakeStepwiseModel([])

    new_params, _new_opt_state, metrics = trainer.update(
        params,
        trainer.init_opt_state(params),
        batch,
        model,
        jax.random.PRNGKey(1),
    )

    assert model.calls == []
    assert jnp.array_equal(new_params["w"], params["w"])
    assert metrics["reward_mean"] == pytest.approx(0.0)
    assert metrics["surrogate_loss"] == pytest.approx(0.0)
    assert metrics["params_changed"] is False


def test_reinforce_trainer_stops_at_max_steps():
    params = _simple_params()
    trainer = _trainer(max_steps=1)
    batch = _states([10.0, 20.0], [False, False])
    model = FakeStepwiseModel(
        [
            (
                _states([9.0, 21.0], [False, False]),
                jnp.asarray([0.0, 0.0], dtype=jnp.float32),
                _zero_grad_logp(params, 2),
            )
        ]
    )

    _new_params, _new_opt_state, metrics = trainer.update(
        params,
        trainer.init_opt_state(params),
        batch,
        model,
        jax.random.PRNGKey(2),
    )

    assert len(model.calls) == 1
    assert metrics["final_flops_best"] == pytest.approx(9.0)


def test_reinforce_trainer_validates_batch_length_before_model_call():
    params = _simple_params()
    trainer = _trainer()
    model = FakeStepwiseModel([])

    with pytest.raises(TrainingError, match="batch length"):
        trainer.update(
            params,
            trainer.init_opt_state(params),
            _states([10.0], [False]),
            model,
            jax.random.PRNGKey(3),
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
    model = FakeStepwiseModel(
        [(_states([10.0, 20.0], [True, True]), logp, _zero_grad_logp(params, 2))]
    )

    with pytest.raises(TrainingError, match="logp"):
        trainer.update(
            params,
            trainer.init_opt_state(params),
            _states([10.0, 20.0], [False, False]),
            model,
            jax.random.PRNGKey(4),
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
    model = FakeStepwiseModel(
        [
            (
                _states([10.0, 20.0], [True, True]),
                jnp.asarray([0.0, 0.0], dtype=jnp.float32),
                grad_logp,
            )
        ]
    )

    with pytest.raises(TrainingError, match="grad_logp"):
        trainer.update(
            params,
            trainer.init_opt_state(params),
            _states([10.0, 20.0], [False, False]),
            model,
            jax.random.PRNGKey(5),
        )


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
            {"w": jnp.asarray([[0.0, 0.0], [jnp.inf, 0.0]])},
            "grad_logp",
        ),
    ],
)
def test_reinforce_trainer_validates_model_step_outputs(
    logp,
    grad_logp,
    message,
):
    params = _simple_params()
    trainer = _trainer()
    model = FakeStepwiseModel(
        [(_states([10.0, 20.0], [True, True]), logp, grad_logp)]
    )

    with pytest.raises(TrainingError, match=message):
        trainer.update(
            params,
            trainer.init_opt_state(params),
            _states([10.0, 20.0], [False, False]),
            model,
            jax.random.PRNGKey(6),
        )
