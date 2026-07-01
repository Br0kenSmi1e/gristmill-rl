from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np

from gristmill_symbolics import TensorComputation
from gristmill_symbolics.cli.train_state import advance_train_state
from gristmill_symbolics.cli.train_state import init_train_state
from gristmill_symbolics.model.transformer_action_selector import (
    SelectorState,
    TransformerActionSelectorModel,
)
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _mixed_initial_states():
    return [
        _state_from_json(actionable_json()),
        _state_from_json(exact_empty_json()),
    ]


def _state_from_json(text):
    return SelectorState(
        comp=TensorComputation.from_json_string(text),
    )


def _floating_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def _model():
    return TransformerActionSelectorModel(
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
        candidate_pad_to=16,
        side_term_pad_to=16,
        d_model=8,
    )


def _trainer(*, batch_size=2, max_steps=2):
    return ReinforceTrainer(
        batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=1.0e-2,
    )


def test_init_train_state_creates_model_params_and_trainer_opt_state():
    model = _model()
    trainer = _trainer()

    state = init_train_state(model, trainer, seed=11)

    assert state.update_index == 0
    assert [field.name for field in fields(type(state))] == [
        "params",
        "opt_state",
        "root_key",
        "update_index",
    ]
    assert not hasattr(state, "policy")
    assert not hasattr(state, "optimizer_config")
    assert _floating_leaves(state.params)
    assert state.opt_state is not None


def test_advance_train_state_uses_stepwise_trainer_and_increments_index():
    model = _model()
    trainer = _trainer(max_steps=1)
    state = init_train_state(model, trainer, seed=29)

    new_state, metrics = advance_train_state(
        state,
        _mixed_initial_states(),
        model=model,
        trainer=trainer,
    )

    assert new_state.update_index == 1
    assert metrics.update_index == 0
    assert metrics.batch_size == 2
    assert np.isfinite(metrics.objective_loss_mean)
    assert np.isfinite(metrics.surrogate_loss)
    assert np.isfinite(metrics.final_flops_best)


def test_advance_train_state_folds_update_index_into_trainer_rng():
    state = init_train_state(_model(), _trainer(max_steps=1), seed=31)
    state = type(state)(
        params=state.params,
        opt_state=state.opt_state,
        root_key=state.root_key,
        update_index=7,
    )

    class RecordingTrainer:
        batch_size = 2

        def __init__(self):
            self.rng = None
            self.model = None

        def init_opt_state(self, params):
            return {"params": params}

        def update(self, params, opt_state, batch, model, rng):
            self.rng = rng
            self.model = model
            return params, opt_state, {
                "reward_mean": 1.0,
                "reward_std": 0.0,
                "objective_loss_mean": -1.0,
                "surrogate_loss": 0.25,
                "final_flops_best": 3.0,
                "params_changed": False,
            }

    model = _model()
    trainer = RecordingTrainer()

    new_state, metrics = advance_train_state(
        state,
        _mixed_initial_states(),
        model=model,
        trainer=trainer,
    )

    assert jnp.array_equal(trainer.rng, jax.random.fold_in(state.root_key, 7))
    assert trainer.model is model
    assert jnp.array_equal(new_state.root_key, state.root_key)
    assert new_state.update_index == 8
    assert metrics.update_index == 7
    assert metrics.params_changed is False
