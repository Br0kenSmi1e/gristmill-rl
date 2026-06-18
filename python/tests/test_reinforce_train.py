import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gristmill_symbolics.reinforce.train_state as train_state_module
from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    OptimizerConfig,
    RolloutConfig,
    TrainingError,
    init_train_state,
    make_optimizer,
    train_update,
)
from tests.policy_fixtures import actionable_state
from tests.test_bindings import exact_empty_json


def _mixed_initial_states():
    return [
        actionable_state(),
        RewriteState.from_computation(TensorComputation.from_json_string(exact_empty_json())),
    ]


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


@pytest.mark.parametrize("learning_rate", [np.nan, 0.0, -1.0e-3])
def test_make_optimizer_rejects_non_finite_or_non_positive_learning_rate(
    learning_rate,
):
    with pytest.raises(TrainingError, match="learning_rate"):
        make_optimizer(OptimizerConfig(learning_rate=learning_rate))


@pytest.mark.parametrize(
    ("config", "field_name"),
    [
        (OptimizerConfig(b1=1.0), "b1"),
        (OptimizerConfig(b2=float("nan")), "b2"),
        (OptimizerConfig(eps=0.0), "eps"),
    ],
)
def test_make_optimizer_rejects_invalid_adam_hyperparameters(config, field_name):
    with pytest.raises(TrainingError, match=field_name):
        make_optimizer(config)


def test_init_train_state_creates_policy_params_and_opt_state():
    state = init_train_state(
        PolicyConfig(d_model=8),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=11,
    )

    assert state.update_index == 0
    assert state.policy.config.d_model == 8
    assert _floating_leaves(state.policy.params)
    assert state.opt_state is not None


def test_train_update_collects_fresh_rollout_and_changes_params_for_nonzero_advantage():
    state = init_train_state(
        PolicyConfig(d_model=8, stop_bias_init=-20.0),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=12,
    )

    new_state, metrics = train_update(
        state,
        _mixed_initial_states(),
        RolloutConfig(batch_size=2, max_steps=1, seed=12),
    )

    assert new_state.update_index == 1
    assert metrics.update_index == 0
    assert metrics.batch_size == 2
    assert metrics.max_steps == 1
    assert np.isfinite(metrics.loss)
    assert metrics.loss == pytest.approx(metrics.objective_loss_mean)
    assert metrics.objective_loss_mean == pytest.approx(-metrics.reward_mean)
    assert metrics.objective_loss_stderr == pytest.approx(metrics.reward_stderr)
    assert np.isfinite(metrics.surrogate_loss)
    assert metrics.target_score_count >= 1
    assert metrics.valid_action_count >= 1
    assert metrics.params_changed is True


def test_train_update_reports_tiny_exact_parameter_change():
    state = init_train_state(
        PolicyConfig(d_model=8, stop_bias_init=-20.0),
        OptimizerConfig(learning_rate=1.0e-8),
        seed=12,
    )

    _new_state, metrics = train_update(
        state,
        _mixed_initial_states(),
        RolloutConfig(batch_size=2, max_steps=1, seed=12),
    )

    assert metrics.params_changed is True


def test_train_update_rejects_non_finite_updated_params(monkeypatch):
    state = init_train_state(
        PolicyConfig(d_model=8, stop_bias_init=-20.0),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=12,
    )

    class NonFiniteUpdateOptimizer:
        def update(self, grads, opt_state, params):
            updates = jax.tree_util.tree_map(
                lambda value: jnp.full_like(value, jnp.nan),
                params,
            )
            return updates, opt_state

    monkeypatch.setattr(
        train_state_module,
        "make_optimizer",
        lambda _config: NonFiniteUpdateOptimizer(),
    )

    with pytest.raises(TrainingError, match="updated policy parameters"):
        train_update(
            state,
            _mixed_initial_states(),
            RolloutConfig(batch_size=2, max_steps=1, seed=12),
        )


def test_multi_sample_update_reports_finite_loss_and_core_metrics():
    state = init_train_state(
        PolicyConfig(d_model=8, stop_bias_init=-20.0),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=17,
    )

    new_state, metrics = train_update(
        state,
        [
            actionable_state(),
            actionable_state(),
            RewriteState.from_computation(
                TensorComputation.from_json_string(exact_empty_json())
            ),
        ],
        RolloutConfig(batch_size=3, max_steps=2, seed=17),
    )

    assert new_state.update_index == 1
    core_float_metrics = [
        metrics.loss,
        metrics.reward_stderr,
        metrics.objective_loss_mean,
        metrics.objective_loss_stderr,
        metrics.surrogate_loss,
        metrics.reward_mean,
        metrics.reward_std,
        metrics.advantage_mean,
        metrics.advantage_std,
        metrics.initial_log_flops_mean,
        metrics.final_log_flops_mean,
        metrics.final_log_flops_best,
        metrics.target_logp_mean,
        metrics.action_logp_mean,
    ]
    assert np.all(np.isfinite(core_float_metrics))
    assert metrics.loss == pytest.approx(metrics.objective_loss_mean)
    assert metrics.objective_loss_mean == pytest.approx(-metrics.reward_mean)
    assert metrics.objective_loss_stderr == pytest.approx(metrics.reward_stderr)
    assert metrics.valid_action_count >= 1
    assert metrics.empty_action_space_count >= 1
    assert metrics.target_score_count >= metrics.action_score_count


def test_train_update_reports_reward_stderr_from_batch_reward_std():
    state = init_train_state(
        PolicyConfig(d_model=8, stop_bias_init=-20.0),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=23,
    )

    _new_state, metrics = train_update(
        state,
        [
            actionable_state(),
            actionable_state(),
            RewriteState.from_computation(TensorComputation.from_json_string(exact_empty_json())),
        ],
        RolloutConfig(batch_size=3, max_steps=2, seed=23),
    )

    assert metrics.reward_stderr == pytest.approx(
        metrics.reward_std / np.sqrt(metrics.batch_size)
    )
