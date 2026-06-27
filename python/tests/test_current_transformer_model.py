import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, RewriteStateRow, TensorComputation
from gristmill_symbolics.policy import PolicyConfig, init_policy_params
from gristmill_symbolics.reinforce import (
    CurrentTransformerModel,
    CurrentTransformerModelConfig,
    TrainingError,
)
from gristmill_symbolics.reinforce.rollout import _collect_streamed_rollout_gradients
from gristmill_symbolics.reinforce.types import PolicyState, RolloutConfig
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _floating_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def _tree_allclose(left, right, *, atol=1.0e-5):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        _floating_leaves(left), _floating_leaves(right), strict=True
    ):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=atol)


def _model_config(**overrides):
    values = {
        "policy_config": PolicyConfig(d_model=8, stop_bias_init=-20.0),
        "batch_size": 2,
        "max_steps": 2,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
    }
    values.update(overrides)
    return CurrentTransformerModelConfig(**values)


def test_current_transformer_model_matches_legacy_static_rollout():
    config = _model_config()
    params = init_policy_params(config.policy_config, jax.random.PRNGKey(0))
    initial_json = [actionable_json(), exact_empty_json()]
    legacy_policy = PolicyState(config=config.policy_config, params=params)
    root_key = jax.random.PRNGKey(23)
    update_index = 4
    legacy = _collect_streamed_rollout_gradients(
        legacy_policy,
        [_state_from_json(text) for text in initial_json],
        RolloutConfig(
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            seed=23,
            static_policy_batch=True,
            state_token_pad_to=config.state_token_pad_to,
            action_token_pad_to=config.action_token_pad_to,
            definition_pad_to=config.definition_pad_to,
        ),
        update_index=update_index,
        root_key=root_key,
    )
    row = RewriteStateRow.from_states([_state_from_json(text) for text in initial_json])
    model = CurrentTransformerModel()

    out_row, logp, grad_logp, metrics = model.sample_with_logp_grad(
        params,
        jax.random.fold_in(root_key, update_index),
        row,
        config,
    )

    assert out_row is row
    assert np.allclose(out_row.log_total_flops(), legacy.final.final_log_flops)
    assert jnp.allclose(logp, legacy.trajectory_logp, atol=1.0e-5)
    _tree_allclose(grad_logp, legacy.trajectory_grad_logp)
    assert set(metrics) == {"stopped"}
    assert metrics["stopped"].tolist() == legacy.final.stopped.tolist()


def test_current_transformer_model_rejects_batch_size_mismatch():
    config = _model_config(batch_size=2)
    params = init_policy_params(config.policy_config, jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(TrainingError, match="row batch size|batch_size"):
        CurrentTransformerModel().sample_with_logp_grad(
            params,
            jax.random.PRNGKey(0),
            row,
            config,
        )


def test_current_transformer_model_static_pad_errors_name_dimension():
    config = _model_config(batch_size=1, state_token_pad_to=1)
    params = init_policy_params(config.policy_config, jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(
        TrainingError,
        match="state token length .* exceeds state_token_pad_to 1",
    ):
        CurrentTransformerModel().sample_with_logp_grad(
            params,
            jax.random.PRNGKey(0),
            row,
            config,
        )
