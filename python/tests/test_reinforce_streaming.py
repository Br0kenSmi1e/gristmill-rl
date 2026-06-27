import jax
import jax.numpy as jnp
import json
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, RewriteStateRow, TensorComputation
from gristmill_symbolics.policy import (
    PolicyConfig,
    init_policy_params,
    sample_action,
    score_action,
)
from gristmill_symbolics.reinforce import (
    CurrentTransformerModel,
    CurrentTransformerModelConfig,
)
from gristmill_symbolics.reinforce.rollout import (
    _dummy_action_policy_item,
    _dummy_state_policy_item,
    _mask_tree_rows,
    _sample_static_model_rollout,
    _stack_bool_masks,
)
from gristmill_symbolics.reinforce.train_state import (
    _reinforce_grad_loss,
    _surrogate_loss,
)
from gristmill_symbolics.reinforce.types import TrainingError
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


def _model_config(**overrides):
    values = {
        "policy_config": PolicyConfig(d_model=8, stop_bias_init=-20.0),
        "batch_size": 1,
        "max_steps": 1,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
    }
    values.update(overrides)
    return CurrentTransformerModelConfig(**values)


def _two_definition_json():
    data = json.loads(actionable_json())
    data["tensors"].append({"id": 4, "symmetry": []})
    data["definitions"].append({**data["definitions"][0], "base": 4})
    return json.dumps(data)


def _params(config):
    return init_policy_params(config.policy_config, jax.random.PRNGKey(0))


def test_stack_bool_masks_can_pad_to_static_width():
    stacked = _stack_bool_masks(
        [
            jnp.asarray([True], dtype=jnp.bool_),
            jnp.asarray([False, True], dtype=jnp.bool_),
        ],
        pad_to=3,
    )

    assert stacked.shape == (2, 3)
    assert stacked.tolist() == [[True, False, False], [False, True, False]]


def test_stack_bool_masks_rejects_static_width_that_is_too_small():
    with pytest.raises(
        TrainingError,
        match="definition mask length 2 exceeds definition_pad_to 1",
    ):
        _stack_bool_masks(
            [jnp.asarray([True, False], dtype=jnp.bool_)],
            pad_to=1,
        )


def test_mask_tree_rows_zeroes_inactive_rows_without_changing_active_rows():
    grads = {
        "leaf": jnp.asarray(
            [
                [1.0, 2.0],
                [3.0, 5.0],
                [7.0, 11.0],
            ],
            dtype=jnp.float32,
        )
    }

    masked = _mask_tree_rows(
        grads,
        jnp.asarray([True, False, True], dtype=jnp.bool_),
    )

    assert masked["leaf"].tolist() == [[1.0, 2.0], [0.0, 0.0], [7.0, 11.0]]


def test_dummy_action_policy_inputs_score_finite_values():
    config = _model_config()
    params = _params(config)
    state_tokens, state_mask = _dummy_state_policy_item()
    action_tokens, action_mask = _dummy_action_policy_item()
    selected = jnp.asarray(0, dtype=jnp.int32)
    action_choice = sample_action(
        params,
        state_tokens,
        state_mask,
        selected,
        action_tokens,
        action_mask,
        jax.random.PRNGKey(99),
    )

    logp, grad = jax.value_and_grad(score_action, argnums=0)(
        params,
        state_tokens,
        state_mask,
        selected,
        action_tokens,
        action_mask,
        action_choice,
    )

    assert np.isfinite(float(np.asarray(logp)))
    for leaf in _floating_leaves(grad):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_reinforce_grad_loss_is_negative_mean_advantage_times_trajectory_grad():
    trajectory_grad = {
        "leaf": jnp.asarray(
            [
                [1.0, 2.0],
                [3.0, 5.0],
                [-7.0, 11.0],
            ],
            dtype=jnp.float32,
        )
    }
    advantage = np.asarray([2.0, -1.0, 0.5], dtype=np.float64)

    grad_loss = _reinforce_grad_loss(trajectory_grad, advantage)

    expected = -jnp.mean(
        jnp.asarray(advantage, dtype=jnp.float32)[:, None] * trajectory_grad["leaf"],
        axis=0,
    )
    assert jnp.allclose(grad_loss["leaf"], expected)


def test_surrogate_loss_uses_trajectory_logp_diagnostic_only():
    logp = jnp.asarray([-1.0, -2.0, -4.0], dtype=jnp.float32)
    advantage = np.asarray([2.0, -1.0, 0.5], dtype=np.float64)

    assert float(_surrogate_loss(logp, advantage)) == pytest.approx(
        float(-jnp.mean(jnp.asarray(advantage, dtype=jnp.float32) * logp))
    )


def test_model_rollout_rejects_non_floating_param_leaves():
    config = _model_config()
    params = {**_params(config), "unused_integer_leaf": jnp.asarray(1, dtype=jnp.int32)}
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(TrainingError, match="floating|streamed gradients"):
        _sample_static_model_rollout(
            params,
            jax.random.PRNGKey(5),
            row,
            config,
        )


def test_model_rollout_rejects_too_small_state_token_pad():
    config = _model_config(state_token_pad_to=1)
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(
        TrainingError,
        match="state token length .* exceeds state_token_pad_to 1",
    ):
        _sample_static_model_rollout(_params(config), jax.random.PRNGKey(5), row, config)


def test_model_rollout_rejects_too_small_definition_pad():
    config = _model_config(definition_pad_to=1)
    row = RewriteStateRow.from_states([_state_from_json(_two_definition_json())])

    with pytest.raises(
        TrainingError,
        match="definition mask length 2 exceeds definition_pad_to 1",
    ):
        _sample_static_model_rollout(_params(config), jax.random.PRNGKey(5), row, config)


def test_model_rollout_rejects_too_small_action_token_pad():
    config = _model_config(action_token_pad_to=1)
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(
        TrainingError,
        match="action token length .* exceeds action_token_pad_to 1",
    ):
        _sample_static_model_rollout(_params(config), jax.random.PRNGKey(5), row, config)


def test_current_model_returns_finite_batched_rollout_tensors():
    config = _model_config(batch_size=2, max_steps=2)
    params = _params(config)
    row = RewriteStateRow.from_states(
        [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]
    )

    out_row, logp, grad_logp, metrics = CurrentTransformerModel().sample_with_logp_grad(
        params,
        jax.random.PRNGKey(8),
        row,
        config,
    )

    assert out_row is row
    assert logp.shape == (2,)
    assert metrics["stopped"].shape == (2,)
    assert np.isfinite(np.asarray(logp)).all()
    assert np.isfinite(np.asarray(out_row.log_total_flops())).all()
    for leaf in _floating_leaves(grad_logp):
        assert leaf.shape[0] == 2
        assert bool(jnp.all(jnp.isfinite(leaf)))
