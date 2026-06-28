from dataclasses import fields

import jax
import jax.numpy as jnp

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.cli.train_state import (
    advance_train_state,
    init_train_state,
)
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _batch():
    return [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]


class RecordingModel:
    batch_size = 2

    def __init__(self):
        self.init_rng = None

    def init_params(self, rng):
        self.init_rng = rng
        return {"w": jnp.asarray([1.0], dtype=jnp.float32)}


class RecordingTrainer:
    batch_size = 2

    def __init__(self):
        self.init_params = None
        self.calls = []

    def init_opt_state(self, params):
        self.init_params = params
        return {"step": 0}

    def update(self, params, opt_state, batch, model, rng):
        self.calls.append((params, opt_state, batch, model, rng))
        return params, opt_state, {
            "reward_mean": 1.0,
            "reward_std": 0.0,
            "objective_loss_mean": -1.0,
            "surrogate_loss": 0.25,
            "final_flops_best": 3.0,
            "params_changed": False,
        }


def test_init_train_state_asks_model_and_trainer_to_initialize_owned_state():
    model = RecordingModel()
    trainer = RecordingTrainer()

    state = init_train_state(model, trainer, seed=11)

    assert [field.name for field in fields(type(state))] == [
        "params",
        "opt_state",
        "root_key",
        "update_index",
    ]
    assert state.update_index == 0
    assert model.init_rng is not None
    assert trainer.init_params is state.params


def test_advance_train_state_calls_trainer_directly_without_adapter_or_config():
    model = RecordingModel()
    trainer = RecordingTrainer()
    state = init_train_state(model, trainer, seed=31, update_index=7)

    new_state, metrics = advance_train_state(
        state,
        _batch(),
        model=model,
        trainer=trainer,
    )

    assert len(trainer.calls) == 1
    params, opt_state, batch, called_model, rng = trainer.calls[0]
    assert params is state.params
    assert opt_state is state.opt_state
    assert len(batch) == 2
    assert called_model is model
    assert jnp.array_equal(rng, jax.random.fold_in(state.root_key, 7))
    assert jnp.array_equal(new_state.root_key, state.root_key)
    assert new_state.update_index == 8
    assert metrics.update_index == 7
    assert metrics.params_changed is False
