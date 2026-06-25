import copy

import jax
import jax.numpy as jnp

from gristmill_symbolics.policy import (
    PolicyConfig,
    batched_sample_action,
    batched_sample_target,
    batched_score_action_grad,
    batched_score_target_grad,
    init_policy_params,
    sample_action,
    sample_target,
    score_action,
    score_target,
    stack_token_trees,
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from tests.policy_fixtures import (
    actionable_action_space_snapshot,
    actionable_state_snapshot,
)


def _params():
    return init_policy_params(
        PolicyConfig(d_model=16),
        jax.random.PRNGKey(0),
    )


def _state_tree():
    return tokenize_state_snapshot(actionable_state_snapshot())


def _action_tree():
    return tokenize_action_space_snapshot(actionable_action_space_snapshot())


def _two_row_state_batch():
    return stack_token_trees([_state_tree(), _state_tree()])


def _two_row_action_batch():
    return stack_token_trees([_action_tree(), _action_tree()])


def _wide_action_space_snapshot(*, candidates=10, side_terms=6):
    snapshot = copy.deepcopy(actionable_action_space_snapshot())
    template = snapshot["candidate_templates"][0]
    widened = []
    for _ in range(candidates):
        candidate = copy.deepcopy(template)
        for side_name in ("left_definition", "right_definition"):
            terms = candidate[side_name]["terms"]
            candidate[side_name]["terms"] = [
                copy.deepcopy(terms[index % len(terms)]) for index in range(side_terms)
            ]
        widened.append(candidate)
    snapshot["candidate_templates"] = widened
    return snapshot


def _slice_tree(tree, index):
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def _floating_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def _tree_allclose(left, right, *, atol=1.0e-5):
    left_leaves = _floating_leaves(left)
    right_leaves = _floating_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=atol)


def _assert_choice_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert jnp.array_equal(left[key], right[key])


def test_batched_sample_target_matches_existing_vmap_call():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    def_mask = jnp.asarray([[True], [False]])
    keys = jax.random.split(jax.random.PRNGKey(10), 2)

    actual = batched_sample_target(
        params,
        state_tokens,
        state_mask,
        def_mask,
        keys,
    )
    expected = jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))(
        params,
        state_tokens,
        state_mask,
        def_mask,
        keys,
    )

    assert jnp.array_equal(actual, expected)


def test_batched_score_target_grad_matches_existing_vmap_call():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    def_mask = jnp.asarray([[False], [True]])
    choices = jnp.asarray([-1, 0], dtype=jnp.int32)

    actual_logps, actual_grads = batched_score_target_grad(
        params,
        state_tokens,
        state_mask,
        def_mask,
        choices,
    )
    expected_logps, expected_grads = jax.vmap(
        jax.value_and_grad(score_target, argnums=0),
        in_axes=(None, 0, 0, 0, 0),
    )(
        params,
        state_tokens,
        state_mask,
        def_mask,
        choices,
    )

    assert actual_logps.shape == (2,)
    assert jnp.allclose(actual_logps, expected_logps, atol=1.0e-5)
    _tree_allclose(actual_grads, expected_grads)
    for leaf in _floating_leaves(actual_grads):
        assert leaf.shape[0] == 2


def test_batched_sample_action_matches_existing_vmap_call():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    action_tokens, action_mask = _two_row_action_batch()
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(25), 2)

    actual = batched_sample_action(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        keys,
    )
    expected = jax.vmap(
        sample_action,
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        keys,
    )

    for index in range(2):
        _assert_choice_equal(_slice_tree(actual, index), _slice_tree(expected, index))


def test_batched_score_action_grad_matches_existing_vmap_call():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    action_tokens, action_mask = _two_row_action_batch()
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(20), 2)
    choices = batched_sample_action(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        keys,
    )

    actual_logps, actual_grads = batched_score_action_grad(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        choices,
    )
    expected_logps, expected_grads = jax.vmap(
        jax.value_and_grad(score_action, argnums=0),
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        choices,
    )

    assert actual_logps.shape == (2,)
    assert jnp.allclose(actual_logps, expected_logps, atol=1.0e-5)
    _tree_allclose(actual_grads, expected_grads)
    for leaf in _floating_leaves(actual_grads):
        assert leaf.shape[0] == 2


def test_batched_sample_action_uses_local_padding_width_not_model_config():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    small_action = _action_tree()
    wide_action = tokenize_action_space_snapshot(
        _wide_action_space_snapshot(candidates=10, side_terms=6)
    )
    action_tokens, action_mask = stack_token_trees([small_action, wide_action])
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(31), 2)

    choices = batched_sample_action(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        keys,
    )
    logp, _grad = batched_score_action_grad(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        choices,
    )

    assert choices["left_mask"].shape == choices["left_valid_mask"].shape
    assert choices["right_mask"].shape == choices["right_valid_mask"].shape
    assert int(jnp.sum(choices["left_valid_mask"][1])) == 6
    assert int(jnp.sum(choices["right_valid_mask"][1])) == 6
    assert bool(jnp.all(jnp.isfinite(logp)))
