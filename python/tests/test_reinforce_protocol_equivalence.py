import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    CurrentTransformerModel,
    CurrentTransformerModelConfig,
    OptimizerConfig,
    ReinforceTrainer,
    ReinforceTrainerConfig,
    init_train_state,
    train_update,
)
from gristmill_symbolics.reinforce.train_state import _ConfiguredModel
from gristmill_symbolics.reinforce.types import RolloutConfig
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _batch():
    return [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]


def _tree_allclose(left, right, *, atol=1.0e-5):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        if hasattr(left_leaf, "dtype") and jnp.issubdtype(left_leaf.dtype, jnp.floating):
            assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=atol)


def test_new_trainer_model_path_matches_legacy_static_train_update():
    policy_config = PolicyConfig(d_model=8, stop_bias_init=-20.0)
    optimizer_config = OptimizerConfig(learning_rate=1.0e-2)
    state = init_train_state(policy_config, optimizer_config, seed=29)
    legacy_config = RolloutConfig(
        batch_size=2,
        max_steps=2,
        seed=29,
        static_policy_batch=True,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
    )
    legacy_state, legacy_metrics = train_update(state, _batch(), legacy_config)

    model_config = CurrentTransformerModelConfig(
        policy_config=policy_config,
        batch_size=2,
        max_steps=2,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
    )
    trainer_config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=optimizer_config,
    )
    rng = jax.random.fold_in(state.root_key, state.update_index)
    new_params, new_opt_state, new_metrics = ReinforceTrainer().update(
        state.policy.params,
        state.opt_state,
        _batch(),
        _ConfiguredModel(CurrentTransformerModel(), model_config),
        rng,
        trainer_config,
    )

    _tree_allclose(new_params, legacy_state.policy.params)
    _tree_allclose(new_opt_state, legacy_state.opt_state)
    assert new_metrics["reward_mean"] == pytest.approx(legacy_metrics.reward_mean)
    assert new_metrics["reward_std"] == pytest.approx(legacy_metrics.reward_std)
    assert new_metrics["objective_loss_mean"] == pytest.approx(
        legacy_metrics.objective_loss_mean
    )
    assert new_metrics["surrogate_loss"] == pytest.approx(
        legacy_metrics.surrogate_loss,
        abs=1.0e-5,
    )
    assert new_metrics["final_flops_best"] == pytest.approx(
        legacy_metrics.final_log_flops_best
    )
    assert new_metrics["params_changed"] is legacy_metrics.params_changed
