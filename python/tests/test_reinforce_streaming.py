import jax
import jax.numpy as jnp
import json
import numpy as np
import pytest

from gristmill_symbolics import (
    RewriteState,
    RewriteStateRow,
    TensorComputation,
    validate_decision,
)
from gristmill_symbolics.policy import (
    PolicyConfig,
    action_choice_to_python,
    init_policy_params,
    sample_action,
    sample_target,
    score_action,
    score_target,
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from gristmill_symbolics.reinforce import (
    CurrentTransformerModel,
    CurrentTransformerModelConfig,
)
from gristmill_symbolics.reinforce.rollout import (
    _dummy_action_policy_item,
    _dummy_state_policy_item,
    _make_decision_rng_grid,
    _mask_tree_rows,
    _sample_static_model_rollout,
    _stack_bool_masks,
)
from gristmill_symbolics.reinforce.train_state import (
    _reinforce_grad_loss,
    _surrogate_loss,
)
from gristmill_symbolics.reinforce.types import (
    DECISION_ACTION,
    DECISION_TARGET,
    TrainingError,
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


def _no_target_json():
    return json.dumps(
        {
            "ranges": [{"id": 0, "size": 3}],
            "tensors": [
                {
                    "id": 0,
                    "symmetry": [{"perm": [0], "action": "Identity"}],
                }
            ],
            "definitions": [
                {
                    "base": 0,
                    "ext_indices": [{"id": 0, "range": 0}],
                    "terms": [
                        {
                            "coeff": [1, 1],
                            "sum_indices": [],
                            "factors": [{"tensor": 0, "indices": [0]}],
                        }
                    ],
                }
            ],
        }
    )


def _params(config):
    return init_policy_params(config.policy_config, jax.random.PRNGKey(0))


def _tree_allclose(left, right, *, atol=1.0e-5):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        _floating_leaves(left), _floating_leaves(right), strict=True
    ):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=atol)


def _tree_add(left, right):
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _tree_row(tree, index):
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def _trim_choice(choice):
    py_choice = action_choice_to_python(choice)
    return {
        "candidate_index": py_choice["candidate_index"],
        "left_mask": [
            keep
            for keep, valid in zip(
                py_choice["left_mask"], py_choice["left_valid_mask"], strict=True
            )
            if valid
        ],
        "right_mask": [
            keep
            for keep, valid in zip(
                py_choice["right_mask"], py_choice["right_valid_mask"], strict=True
            )
            if valid
        ],
    }


def _manual_decision_keys(rng, config):
    flat_keys = jax.random.split(rng, config.max_steps * config.batch_size * 2)
    return flat_keys.reshape((config.max_steps, config.batch_size, 2, 2))


def _scalar_oracle(params, rng, state, config, *, sample_index=0):
    decision_keys = _manual_decision_keys(rng, config)
    logp = jnp.asarray(0.0, dtype=jnp.float32)
    grad_logp = jax.tree_util.tree_map(jnp.zeros_like, params)
    exact_empty_def_mask = None
    stopped = False

    for step in range(config.max_steps):
        state_tokens, state_mask = tokenize_state_snapshot(state.snapshot())
        def_mask = jnp.asarray(state.definition_mask(), dtype=jnp.bool_)
        if exact_empty_def_mask is not None and not bool(np.asarray(jnp.any(def_mask))):
            def_mask = exact_empty_def_mask
        target_key = decision_keys[step, sample_index, DECISION_TARGET]
        target_choice = sample_target(
            params,
            state_tokens,
            state_mask,
            def_mask,
            target_key,
        )
        target_logp, target_grad = jax.value_and_grad(score_target, argnums=0)(
            params,
            state_tokens,
            state_mask,
            def_mask,
            target_choice,
        )
        logp = logp + target_logp
        grad_logp = _tree_add(grad_logp, target_grad)

        target_index = int(np.asarray(target_choice))
        if target_index == -1:
            stopped = True
            break

        space = state.action_space_for_def(target_index)
        if space is None:
            exact_empty_def_mask = jnp.zeros_like(def_mask).at[target_index].set(True)
            continue

        action_tokens, action_mask = tokenize_action_space_snapshot(space.snapshot())
        action_key = decision_keys[step, sample_index, DECISION_ACTION]
        action_choice = sample_action(
            params,
            state_tokens,
            state_mask,
            target_choice,
            action_tokens,
            action_mask,
            action_key,
        )
        action_logp, action_grad = jax.value_and_grad(score_action, argnums=0)(
            params,
            state_tokens,
            state_mask,
            target_choice,
            action_tokens,
            action_mask,
            action_choice,
        )
        logp = logp + action_logp
        grad_logp = _tree_add(grad_logp, action_grad)
        decision = _trim_choice(action_choice)
        validate_decision(space, decision)
        state.apply_validated_decision(space, decision)
        exact_empty_def_mask = None

    return logp, grad_logp, stopped, state


def test_model_rollout_matches_scalar_oracle_for_sampled_score_accumulation():
    config = _model_config(max_steps=2)
    params = _params(config)
    rng = jax.random.PRNGKey(17)
    row = RewriteStateRow.from_states([_state_from_json(_two_definition_json())])
    expected_logp, expected_grad, expected_stopped, expected_state = _scalar_oracle(
        params,
        rng,
        _state_from_json(_two_definition_json()),
        config,
    )

    result = _sample_static_model_rollout(params, rng, row, config)

    assert jnp.allclose(result.logp[0], expected_logp, atol=1.0e-5)
    _tree_allclose(_tree_row(result.grad_logp, 0), expected_grad)
    assert result.stopped.tolist() == [expected_stopped]
    assert float(result.out_row.log_total_flops()[0]) == pytest.approx(
        expected_state.log_total_flops()
    )


def test_model_rollout_uses_physical_sample_rng_and_masks_inactive_rows():
    config = _model_config(batch_size=2, max_steps=2)
    params = _params(config)
    rng = jax.random.PRNGKey(19)
    row = RewriteStateRow.from_states(
        [_state_from_json(_no_target_json()), _state_from_json(actionable_json())]
    )
    expected0_logp, expected0_grad, expected0_stopped, expected0_state = _scalar_oracle(
        params,
        rng,
        _state_from_json(_no_target_json()),
        config,
        sample_index=0,
    )
    expected1_logp, expected1_grad, expected1_stopped, expected1_state = _scalar_oracle(
        params,
        rng,
        _state_from_json(actionable_json()),
        config,
        sample_index=1,
    )

    result = _sample_static_model_rollout(params, rng, row, config)

    assert result.stopped.tolist() == [expected0_stopped, expected1_stopped]
    assert jnp.allclose(result.logp[0], expected0_logp, atol=1.0e-5)
    assert jnp.allclose(result.logp[1], expected1_logp, atol=1.0e-5)
    _tree_allclose(_tree_row(result.grad_logp, 0), expected0_grad)
    _tree_allclose(_tree_row(result.grad_logp, 1), expected1_grad)
    assert result.out_row.log_total_flops() == pytest.approx(
        [expected0_state.log_total_flops(), expected1_state.log_total_flops()]
    )


def test_model_rollout_replays_exact_empty_definition_with_scalar_oracle():
    config = _model_config(max_steps=2)
    params = _params(config)
    rng = jax.random.PRNGKey(23)
    row = RewriteStateRow.from_states([_state_from_json(exact_empty_json())])
    expected_logp, expected_grad, expected_stopped, expected_state = _scalar_oracle(
        params,
        rng,
        _state_from_json(exact_empty_json()),
        config,
    )

    result = _sample_static_model_rollout(params, rng, row, config)

    assert jnp.allclose(result.logp[0], expected_logp, atol=1.0e-5)
    _tree_allclose(_tree_row(result.grad_logp, 0), expected_grad)
    assert result.stopped.tolist() == [expected_stopped]
    assert float(result.out_row.log_total_flops()[0]) == pytest.approx(
        expected_state.log_total_flops()
    )


def test_decision_rng_grid_matches_manual_step_sample_decision_layout():
    config = _model_config(batch_size=2, max_steps=3)
    rng = jax.random.PRNGKey(123)

    assert jnp.array_equal(
        _make_decision_rng_grid(rng, config.max_steps, config.batch_size),
        _manual_decision_keys(rng, config),
    )


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
