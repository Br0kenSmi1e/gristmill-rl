import inspect
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, RewriteStateRow, TensorComputation
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics._training import TrainingError
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


def test_transformer_action_selector_model_protocol_is_config_free():
    model = _model()

    assert model.batch_size == 2
    assert list(inspect.signature(model.init_params).parameters) == ["rng"]
    assert list(inspect.signature(model.sample_with_logp_grad).parameters) == [
        "params",
        "rng",
        "row",
    ]


def test_transformer_action_selector_constructor_kwargs_round_trip():
    model = _model(init_scale=0.03, stop_bias_init=-7.0)

    kwargs = model.constructor_kwargs()
    restored = TransformerActionSelectorModel(**kwargs)

    assert restored.constructor_kwargs() == kwargs


@pytest.mark.parametrize(
    "overrides",
    [
        {"init_scale": float("nan")},
        {"stop_bias_init": None},
        {"init_scale": True},
        {"init_scale": np.bool_(True)},
    ],
)
def test_transformer_action_selector_model_rejects_invalid_float_settings(overrides):
    with pytest.raises(TrainingError, match="must be a finite float"):
        _model(**overrides)


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"max_steps": 0}, "max_steps"),
        ({"state_token_pad_to": None}, "state_token_pad_to"),
        ({"action_token_pad_to": 0}, "action_token_pad_to"),
        ({"definition_pad_to": True}, "definition_pad_to"),
    ],
)
def test_transformer_action_selector_model_rejects_invalid_static_shapes(
    overrides,
    field_name,
):
    with pytest.raises(TrainingError, match=field_name):
        _model(**overrides)


def test_transformer_action_selector_model_module_does_not_export_old_param_helper():
    module = importlib.import_module(
        "gristmill_symbolics.model.transformer_action_selector.model"
    )

    assert not hasattr(module, "init_policy_params")


def test_transformer_action_selector_model_initializes_params_and_samples():
    model = _model()
    params = model.init_params(jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states(
        [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]
    )

    out_row, logp, grad_logp, metrics = model.sample_with_logp_grad(
        params,
        jax.random.PRNGKey(23),
        row,
    )

    assert out_row is row
    assert logp.shape == (2,)
    assert set(metrics) == {"stopped"}
    assert metrics["stopped"].shape == (2,)
    assert metrics["stopped"].dtype == bool
    assert np.isfinite(np.asarray(logp)).all()
    assert jax.tree_util.tree_structure(grad_logp) == jax.tree_util.tree_structure(
        params
    )
    floating_leaves = _floating_leaves(grad_logp)
    assert floating_leaves
    for leaf in floating_leaves:
        assert leaf.shape[0] == 2
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_transformer_action_selector_model_rejects_batch_size_mismatch():
    model = _model(batch_size=2)
    params = model.init_params(jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(TrainingError, match="row batch size|batch_size"):
        model.sample_with_logp_grad(params, jax.random.PRNGKey(0), row)
