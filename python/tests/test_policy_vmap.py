import jax
import jax.numpy as jnp

from gristmill_symbolics.policy import (
    PolicyConfig,
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
        PolicyConfig(d_model=16, max_candidates=8, max_side_terms=4),
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


def _slice_tree(tree, index):
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def _assert_choice_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert jnp.array_equal(left[key], right[key])


def test_vmap_score_target_matches_scalar_rows():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    def_mask = jnp.asarray([[False], [True]])
    choices = jnp.asarray([-1, 0], dtype=jnp.int32)

    vmapped = jax.vmap(score_target, in_axes=(None, 0, 0, 0, 0))(
        params, state_tokens, state_mask, def_mask, choices
    )
    scalar = jnp.asarray(
        [
            score_target(
                params,
                _slice_tree(state_tokens, index),
                state_mask[index],
                def_mask[index],
                choices[index],
            )
            for index in range(2)
        ]
    )

    assert vmapped.shape == (2,)
    assert jnp.allclose(vmapped, scalar)


def test_vmap_sample_target_matches_scalar_rows_with_same_keys():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    def_mask = jnp.asarray([[True], [False]])
    keys = jax.random.split(jax.random.PRNGKey(10), 2)

    vmapped_choices, vmapped_logp = jax.vmap(
        sample_target, in_axes=(None, 0, 0, 0, 0)
    )(params, state_tokens, state_mask, def_mask, keys)
    scalar = [
        sample_target(
            params,
            _slice_tree(state_tokens, index),
            state_mask[index],
            def_mask[index],
            keys[index],
        )
        for index in range(2)
    ]
    scalar_choices = jnp.asarray([choice for choice, _ in scalar])
    scalar_logp = jnp.asarray([logp for _, logp in scalar])

    assert jnp.array_equal(vmapped_choices, scalar_choices)
    assert jnp.allclose(vmapped_logp, scalar_logp)


def test_vmap_score_action_matches_scalar_rows_for_vmapped_samples():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    action_tokens, action_mask = _two_row_action_batch()
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(20), 2)

    choices, _ = jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, keys
    )
    vmapped = jax.vmap(score_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        choices,
    )
    scalar = jnp.asarray(
        [
            score_action(
                params,
                _slice_tree(state_tokens, index),
                state_mask[index],
                selected_defs[index],
                _slice_tree(action_tokens, index),
                action_mask[index],
                _slice_tree(choices, index),
            )
            for index in range(2)
        ]
    )

    assert vmapped.shape == (2,)
    assert jnp.allclose(vmapped, scalar)


def test_vmap_sample_action_matches_scalar_rows_with_same_keys():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    action_tokens, action_mask = _two_row_action_batch()
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(25), 2)

    vmapped_choices, vmapped_logp = jax.vmap(
        sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0)
    )(params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, keys)
    scalar = [
        sample_action(
            params,
            _slice_tree(state_tokens, index),
            state_mask[index],
            selected_defs[index],
            _slice_tree(action_tokens, index),
            action_mask[index],
            keys[index],
        )
        for index in range(2)
    ]
    scalar_logp = jnp.asarray([logp for _, logp in scalar])

    for index, (scalar_choice, _) in enumerate(scalar):
        _assert_choice_equal(_slice_tree(vmapped_choices, index), scalar_choice)
    assert jnp.allclose(vmapped_logp, scalar_logp)


def test_width_one_vmap_sample_action_matches_scalar_with_same_key():
    params = _params()
    state_tokens, state_mask = stack_token_trees([_state_tree()])
    action_tokens, action_mask = stack_token_trees([_action_tree()])
    selected_defs = jnp.asarray([0], dtype=jnp.int32)
    key = jax.random.PRNGKey(30)
    keys = key[None, :]

    vmapped_choice, vmapped_logp = jax.vmap(
        sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0)
    )(params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, keys)
    scalar_choice, scalar_logp = sample_action(
        params,
        _slice_tree(state_tokens, 0),
        state_mask[0],
        selected_defs[0],
        _slice_tree(action_tokens, 0),
        action_mask[0],
        key,
    )

    _assert_choice_equal(_slice_tree(vmapped_choice, 0), scalar_choice)
    assert jnp.allclose(vmapped_logp[0], scalar_logp)
