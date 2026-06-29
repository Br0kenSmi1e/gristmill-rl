import jax
import jax.numpy as jnp

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.cli.train_state import advance_train_state, init_train_state
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _batch():
    return [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]


def _assert_pytrees_equal(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        if hasattr(left_leaf, "dtype") or hasattr(right_leaf, "dtype"):
            assert bool(jnp.array_equal(left_leaf, right_leaf))
        else:
            assert left_leaf == right_leaf


def test_object_composed_training_is_deterministic_for_same_seed_and_kwargs():
    model_kwargs = {
        "batch_size": 2,
        "max_steps": 2,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
        "d_model": 8,
        "stop_bias_init": -20.0,
    }
    trainer_kwargs = {"batch_size": 2, "learning_rate": 1.0e-2}

    left_model = TransformerActionSelectorModel(**model_kwargs)
    left_trainer = ReinforceTrainer(**trainer_kwargs)
    right_model = TransformerActionSelectorModel(**model_kwargs)
    right_trainer = ReinforceTrainer(**trainer_kwargs)

    left_state = init_train_state(left_model, left_trainer, seed=29)
    right_state = init_train_state(right_model, right_trainer, seed=29)

    left_next, left_metrics = advance_train_state(
        left_state,
        _batch(),
        model=left_model,
        trainer=left_trainer,
    )
    right_next, right_metrics = advance_train_state(
        right_state,
        _batch(),
        model=right_model,
        trainer=right_trainer,
    )

    _assert_pytrees_equal(left_next.params, right_next.params)
    _assert_pytrees_equal(left_next.opt_state, right_next.opt_state)
    assert left_next.update_index == right_next.update_index == 1
    assert left_metrics == right_metrics
    assert left_metrics.params_changed is True
