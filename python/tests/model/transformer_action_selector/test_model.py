import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics._training import TrainingError
from gristmill_symbolics import RewriteState, RewriteStateRow, TensorComputation
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
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


def _model(**overrides):
    values = {
        "batch_size": 2,
        "max_steps": 2,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
        "d_model": 8,
        "stop_bias_init": -20.0,
    }
    values.update(overrides)
    return TransformerActionSelectorModel(**values)


def test_transformer_action_selector_model_returns_passthrough_shape_and_metrics():
    model = _model()
    params = model.init_params(jax.random.PRNGKey(0))
    initial_json = [actionable_json(), exact_empty_json()]
    root_key = jax.random.PRNGKey(23)
    row = RewriteStateRow.from_states([_state_from_json(text) for text in initial_json])

    out_row, logp, grad_logp, metrics = model.sample_with_logp_grad(
        params,
        root_key,
        row,
    )

    assert out_row is row
    assert logp.shape == (model.batch_size,)
    assert set(metrics) == {"stopped"}
    assert metrics["stopped"].shape == (model.batch_size,)
    assert metrics["stopped"].dtype == bool
    for leaf in _floating_leaves(grad_logp):
        assert leaf.shape[0] == model.batch_size
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_transformer_action_selector_model_rejects_batch_size_mismatch():
    model = _model(batch_size=2)
    params = model.init_params(jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(TrainingError, match="row batch size|batch_size"):
        model.sample_with_logp_grad(
            params,
            jax.random.PRNGKey(0),
            row,
        )


def test_transformer_action_selector_model_static_pad_errors_name_dimension():
    model = _model(batch_size=1, state_token_pad_to=1)
    params = model.init_params(jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(
        TrainingError,
        match="state token length .* exceeds state_token_pad_to 1",
    ):
        model.sample_with_logp_grad(
            params,
            jax.random.PRNGKey(0),
            row,
        )
