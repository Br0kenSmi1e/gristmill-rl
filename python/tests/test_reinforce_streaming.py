import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gristmill_symbolics.reinforce.rollout as rollout_module
from gristmill_symbolics import RewriteState, TensorComputation, validate_decision
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
from gristmill_symbolics.reinforce.rollout import (
    _collect_streamed_rollout_gradients,
    _dummy_action_policy_item,
    _dummy_state_policy_item,
    _mask_tree_rows,
    _stack_bool_masks,
    make_rng_grid,
)
from gristmill_symbolics.reinforce.train_state import (
    _reinforce_grad_loss,
    _surrogate_loss,
)
from gristmill_symbolics.reinforce.types import (
    DECISION_ACTION,
    DECISION_TARGET,
    PolicyState,
    RolloutConfig,
    TrainingError,
)
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _policy(*, stop_bias_init=-20.0):
    config = PolicyConfig(d_model=8, stop_bias_init=stop_bias_init)
    return PolicyState(
        config=config, params=init_policy_params(config, jax.random.PRNGKey(0))
    )


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _floating_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def _tree_allclose(left, right, *, atol=1.0e-5):
    for left_leaf, right_leaf in zip(
        _floating_leaves(left), _floating_leaves(right), strict=True
    ):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=atol)


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


def _two_actionable_json():
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


def _definition_mask(state):
    return jnp.asarray(state.definition_mask(), dtype=jnp.bool_)


def _tree_add(left, right):
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _tree_row(tree, index):
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def _static_config(**overrides):
    kwargs = {
        "batch_size": 1,
        "max_steps": 1,
        "seed": 5,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
        "static_policy_batch": True,
    }
    kwargs.update(overrides)
    return RolloutConfig(**kwargs)


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
    policy = _policy()
    state_tokens, state_mask = _dummy_state_policy_item()
    action_tokens, action_mask = _dummy_action_policy_item()
    selected = jnp.asarray(0, dtype=jnp.int32)
    action_choice = sample_action(
        policy.params,
        state_tokens,
        state_mask,
        selected,
        action_tokens,
        action_mask,
        jax.random.PRNGKey(99),
    )

    logp, grad = jax.value_and_grad(score_action, argnums=0)(
        policy.params,
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


def _scalar_rollout_oracle(policy, state, config, *, update_index, root_key):
    rng_grid = make_rng_grid(
        root_key,
        update_index=update_index,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
    )
    trajectory_logp = jnp.asarray(0.0, dtype=jnp.float32)
    trajectory_grad = jax.tree_util.tree_map(jnp.zeros_like, policy.params)
    exact_empty_def_mask = None

    for step in range(config.max_steps):
        state_tokens, state_mask = tokenize_state_snapshot(state.snapshot())
        def_mask = _definition_mask(state)
        if exact_empty_def_mask is not None and not bool(np.asarray(jnp.any(def_mask))):
            def_mask = exact_empty_def_mask
        target_key = rng_grid[step, 0, DECISION_TARGET]
        target_choice = sample_target(
            policy.params, state_tokens, state_mask, def_mask, target_key
        )
        target_logp, target_grad = jax.value_and_grad(score_target, argnums=0)(
            policy.params, state_tokens, state_mask, def_mask, target_choice
        )
        trajectory_logp = trajectory_logp + target_logp
        trajectory_grad = _tree_add(trajectory_grad, target_grad)

        if int(np.asarray(target_choice)) == -1:
            break

        space = state.action_space_for_def(int(np.asarray(target_choice)))
        if space is None:
            exact_empty_def_mask = jnp.zeros_like(def_mask).at[
                int(np.asarray(target_choice))
            ].set(True)
            continue

        action_tokens, action_mask = tokenize_action_space_snapshot(space.snapshot())
        action_key = rng_grid[step, 0, DECISION_ACTION]
        action_choice = sample_action(
            policy.params,
            state_tokens,
            state_mask,
            target_choice,
            action_tokens,
            action_mask,
            action_key,
        )
        action_logp, action_grad = jax.value_and_grad(score_action, argnums=0)(
            policy.params,
            state_tokens,
            state_mask,
            target_choice,
            action_tokens,
            action_mask,
            action_choice,
        )
        trajectory_logp = trajectory_logp + action_logp
        trajectory_grad = _tree_add(trajectory_grad, action_grad)
        decision = _trim_choice(action_choice)
        validate_decision(space, decision)
        state.apply_validated_decision(space, decision)
        exact_empty_def_mask = None

    return trajectory_logp, trajectory_grad


def test_streamed_rollout_rejects_non_floating_param_leaves():
    policy = _policy()
    policy = PolicyState(
        config=policy.config,
        params={**policy.params, "unused_integer_leaf": jnp.asarray(1, dtype=jnp.int32)},
    )

    with pytest.raises(TrainingError, match="floating|streamed gradients"):
        _collect_streamed_rollout_gradients(
            policy,
            [_state_from_json(actionable_json())],
            RolloutConfig(batch_size=1, max_steps=1, seed=5),
            update_index=0,
            root_key=jax.random.PRNGKey(5),
        )


def test_static_rollout_rejects_too_small_state_token_pad():
    policy = _policy()

    with pytest.raises(
        TrainingError,
        match="state token length .* exceeds state_token_pad_to 1",
    ):
        _collect_streamed_rollout_gradients(
            policy,
            [_state_from_json(actionable_json())],
            _static_config(state_token_pad_to=1),
            update_index=0,
            root_key=jax.random.PRNGKey(5),
        )


def test_static_rollout_rejects_too_small_definition_pad():
    policy = _policy()

    with pytest.raises(
        TrainingError,
        match="definition mask length 2 exceeds definition_pad_to 1",
    ):
        _collect_streamed_rollout_gradients(
            policy,
            [_state_from_json(_two_actionable_json())],
            _static_config(definition_pad_to=1),
            update_index=0,
            root_key=jax.random.PRNGKey(5),
        )


def test_static_rollout_rejects_too_small_action_token_pad():
    policy = _policy()

    with pytest.raises(
        TrainingError,
        match="action token length .* exceeds action_token_pad_to 1",
    ):
        _collect_streamed_rollout_gradients(
            policy,
            [_state_from_json(actionable_json())],
            _static_config(action_token_pad_to=1),
            update_index=0,
            root_key=jax.random.PRNGKey(5),
        )


def test_static_rollout_skips_dummy_only_action_application(monkeypatch):
    def sample_stop(
        _params,
        _state_tokens,
        _state_token_mask,
        def_mask,
        _rng,
    ):
        return jnp.full((def_mask.shape[0],), -1, dtype=jnp.int32)

    monkeypatch.setattr(rollout_module, "batched_sample_target", sample_stop)
    policy = _policy()

    result = _collect_streamed_rollout_gradients(
        policy,
        [_state_from_json(actionable_json())],
        _static_config(max_steps=2),
        update_index=0,
        root_key=jax.random.PRNGKey(5),
    )

    assert result.valid_action_count == 0
    assert result.action_score_count == 0
    assert result.action_logp_sum == pytest.approx(0.0)
    assert bool(jnp.all(jnp.isfinite(result.trajectory_logp)))
    for leaf in _floating_leaves(result.trajectory_grad_logp):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_static_rollout_preserves_physical_target_rows_after_lower_row_stops(
    monkeypatch,
):
    def sample_first_target_when_present(
        _params,
        _state_tokens,
        _state_token_mask,
        def_mask,
        _rng,
    ):
        return jnp.where(jnp.any(def_mask, axis=1), 0, -1).astype(jnp.int32)

    monkeypatch.setattr(
        rollout_module,
        "batched_sample_target",
        sample_first_target_when_present,
    )
    policy = _policy()
    root = jax.random.PRNGKey(19)
    states_json = [_no_target_json(), exact_empty_json()]

    dynamic = _collect_streamed_rollout_gradients(
        policy,
        [_state_from_json(text) for text in states_json],
        RolloutConfig(batch_size=2, max_steps=2, seed=19),
        update_index=0,
        root_key=root,
    )
    static = _collect_streamed_rollout_gradients(
        policy,
        [_state_from_json(text) for text in states_json],
        _static_config(batch_size=2, max_steps=2, seed=19),
        update_index=0,
        root_key=root,
    )

    assert dynamic.final.stopped.tolist() == [True, False]
    assert dynamic.final.max_steps.tolist() == [False, True]
    assert static.final.stopped.tolist() == dynamic.final.stopped.tolist()
    assert static.final.max_steps.tolist() == dynamic.final.max_steps.tolist()
    assert static.stop_count == dynamic.stop_count == 1
    assert static.target_score_count == dynamic.target_score_count == 3
    assert static.empty_action_space_count == dynamic.empty_action_space_count == 2


def test_streamed_rollout_accumulates_one_step_sampled_score_gradients():
    policy = _policy()
    state = _state_from_json(actionable_json())
    root = jax.random.PRNGKey(5)
    config = RolloutConfig(batch_size=1, max_steps=1, seed=5)

    result = _collect_streamed_rollout_gradients(
        policy, [state], config, update_index=0, root_key=root
    )

    rng_grid = make_rng_grid(root, update_index=0, max_steps=1, batch_size=1)
    state_tokens, state_mask = tokenize_state_snapshot(state.snapshot())
    def_mask = _definition_mask(state)
    target_choice = sample_target(
        policy.params,
        state_tokens,
        state_mask,
        def_mask,
        rng_grid[0, 0, DECISION_TARGET],
    )
    expected_logp, expected_grad = jax.value_and_grad(score_target, argnums=0)(
        policy.params, state_tokens, state_mask, def_mask, target_choice
    )

    if int(np.asarray(target_choice)) != -1:
        space = state.action_space_for_def(int(np.asarray(target_choice)))
        if space is not None:
            action_tokens, action_mask = tokenize_action_space_snapshot(space.snapshot())
            action_choice = sample_action(
                policy.params,
                state_tokens,
                state_mask,
                target_choice,
                action_tokens,
                action_mask,
                rng_grid[0, 0, DECISION_ACTION],
            )
            action_logp, action_grad = jax.value_and_grad(score_action, argnums=0)(
                policy.params,
                state_tokens,
                state_mask,
                target_choice,
                action_tokens,
                action_mask,
                action_choice,
            )
            expected_logp = expected_logp + action_logp
            expected_grad = _tree_add(expected_grad, action_grad)

    assert jnp.allclose(result.trajectory_logp[0], expected_logp, atol=1.0e-5)
    _tree_allclose(_tree_row(result.trajectory_grad_logp, 0), expected_grad)


def test_streamed_rollout_accumulates_multi_step_scalar_oracle():
    policy = _policy()
    root = jax.random.PRNGKey(17)
    config = RolloutConfig(batch_size=1, max_steps=2, seed=17)

    result = _collect_streamed_rollout_gradients(
        policy,
        [_state_from_json(_two_actionable_json())],
        config,
        update_index=0,
        root_key=root,
    )
    expected_logp, expected_grad = _scalar_rollout_oracle(
        policy,
        _state_from_json(_two_actionable_json()),
        config,
        update_index=0,
        root_key=root,
    )

    assert jnp.allclose(result.trajectory_logp[0], expected_logp, atol=1.0e-5)
    _tree_allclose(_tree_row(result.trajectory_grad_logp, 0), expected_grad)


def test_static_rollout_matches_scalar_oracle_on_one_sample():
    policy = _policy()
    root = jax.random.PRNGKey(17)
    config = _static_config(max_steps=2, seed=17)

    result = _collect_streamed_rollout_gradients(
        policy,
        [_state_from_json(_two_actionable_json())],
        config,
        update_index=0,
        root_key=root,
    )
    expected_logp, expected_grad = _scalar_rollout_oracle(
        policy,
        _state_from_json(_two_actionable_json()),
        config,
        update_index=0,
        root_key=root,
    )

    assert jnp.allclose(result.trajectory_logp[0], expected_logp, atol=1.0e-5)
    _tree_allclose(_tree_row(result.trajectory_grad_logp, 0), expected_grad)
    assert result.target_score_count >= 1
    assert result.action_score_count >= 1


def test_streamed_rollout_supports_batched_mixed_action_counts():
    policy = _policy()
    result = _collect_streamed_rollout_gradients(
        policy,
        [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())],
        RolloutConfig(batch_size=2, max_steps=1, seed=8),
        update_index=0,
        root_key=jax.random.PRNGKey(8),
    )

    assert result.final.initial_log_flops.shape == (2,)
    assert result.final.final_log_flops.shape == (2,)
    assert result.trajectory_logp.shape == (2,)
    assert result.target_score_count >= 1
    assert result.action_score_count >= 0
    for leaf in _floating_leaves(result.trajectory_grad_logp):
        assert leaf.shape[0] == 2


def test_static_rollout_matches_dynamic_streamed_mixed_batch():
    policy = _policy()
    root = jax.random.PRNGKey(23)
    initial_states = [
        _state_from_json(actionable_json()),
        _state_from_json(exact_empty_json()),
        _state_from_json(_two_actionable_json()),
    ]
    dynamic_config = RolloutConfig(batch_size=3, max_steps=2, seed=23)
    static_config = _static_config(batch_size=3, max_steps=2, seed=23)

    dynamic = _collect_streamed_rollout_gradients(
        policy,
        [
            _state_from_json(actionable_json()),
            _state_from_json(exact_empty_json()),
            _state_from_json(_two_actionable_json()),
        ],
        dynamic_config,
        update_index=0,
        root_key=root,
    )
    static = _collect_streamed_rollout_gradients(
        policy,
        initial_states,
        static_config,
        update_index=0,
        root_key=root,
    )

    assert np.allclose(static.final.initial_log_flops, dynamic.final.initial_log_flops)
    assert np.allclose(static.final.final_log_flops, dynamic.final.final_log_flops)
    assert static.final.stopped.tolist() == dynamic.final.stopped.tolist()
    assert static.final.max_steps.tolist() == dynamic.final.max_steps.tolist()
    assert jnp.allclose(static.trajectory_logp, dynamic.trajectory_logp, atol=1.0e-5)
    _tree_allclose(static.trajectory_grad_logp, dynamic.trajectory_grad_logp)
    assert static.valid_action_count == dynamic.valid_action_count
    assert static.stop_count == dynamic.stop_count
    assert static.empty_action_space_count == dynamic.empty_action_space_count
    assert static.finished_count == dynamic.finished_count
    assert static.target_score_count == dynamic.target_score_count
    assert static.action_score_count == dynamic.action_score_count
    assert static.target_logp_sum == pytest.approx(dynamic.target_logp_sum, abs=1.0e-5)
    assert static.action_logp_sum == pytest.approx(dynamic.action_logp_sum, abs=1.0e-5)


def test_static_rollout_inactive_rows_do_not_increase_metrics_or_apply_counts():
    policy = _policy(stop_bias_init=100.0)

    result = _collect_streamed_rollout_gradients(
        policy,
        [_state_from_json(actionable_json())],
        _static_config(max_steps=3, seed=31),
        update_index=0,
        root_key=jax.random.PRNGKey(31),
    )

    assert result.stop_count == 1
    assert result.finished_count == 2
    assert result.valid_action_count == 0
    assert result.target_score_count == 1
    assert result.action_score_count == 0
    assert result.target_logp_sum == pytest.approx(float(result.trajectory_logp[0]))
    assert result.action_logp_sum == pytest.approx(0.0)
    assert np.isfinite(np.asarray(result.trajectory_logp)).all()
    for leaf in _floating_leaves(result.trajectory_grad_logp):
        assert bool(jnp.all(jnp.isfinite(leaf)))
