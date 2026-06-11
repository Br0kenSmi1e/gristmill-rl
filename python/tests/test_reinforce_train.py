import jax
import jax.numpy as jnp
import numpy as np

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    OptimizerConfig,
    RolloutConfig,
    init_train_state,
    make_optimizer,
    train_update,
)
from tests.policy_fixtures import actionable_state
from tests.test_bindings import exact_empty_json


def _floating_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def test_make_optimizer_returns_optax_gradient_transformation():
    optimizer = make_optimizer(OptimizerConfig(learning_rate=1.0e-2))

    assert hasattr(optimizer, "init")
    assert hasattr(optimizer, "update")


def test_init_train_state_creates_policy_params_and_opt_state():
    state = init_train_state(
        PolicyConfig(d_model=8, max_candidates=8, max_side_terms=4),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=11,
    )

    assert state.update_index == 0
    assert state.policy.config.d_model == 8
    assert _floating_leaves(state.policy.params)
    assert state.opt_state is not None


def test_train_update_collects_fresh_rollout_and_changes_params_for_nonzero_advantage():
    state = init_train_state(
        PolicyConfig(d_model=8, max_candidates=8, max_side_terms=4, stop_bias_init=-20.0),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=12,
    )

    new_state, metrics, table = train_update(
        state,
        [
            actionable_state(),
            RewriteState.from_computation(TensorComputation.from_json_string(exact_empty_json())),
        ],
        RolloutConfig(batch_size=2, max_steps=1, seed=12),
    )

    assert new_state.update_index == 1
    assert metrics.update_index == 0
    assert metrics.batch_size == 2
    assert metrics.max_steps == 1
    assert np.isfinite(metrics.loss)
    assert metrics.target_score_count >= 1
    assert metrics.valid_action_count >= 1
    assert metrics.params_changed is True
    assert table.target_score_mask.shape == (1, 2)
