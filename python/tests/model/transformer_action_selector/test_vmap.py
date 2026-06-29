import copy

import jax
import jax.numpy as jnp

from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.model.transformer_action_selector.api import (
    sample_action,
    sample_target,
    score_action,
    score_target,
)
from gristmill_symbolics.model.transformer_action_selector.tokenize import (
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from gristmill_symbolics.model.transformer_action_selector.tree import (
    stack_token_trees,
)
from tests.policy_fixtures import (
    actionable_action_space_snapshot,
    actionable_state_snapshot,
)


def _params():
    model = TransformerActionSelectorModel(
        batch_size=2,
        max_steps=1,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
        d_model=16,
    )
    return model.init_params(jax.random.PRNGKey(0))


def _state_tree():
    return tokenize_state_snapshot(actionable_state_snapshot())


def _action_tree():
    return tokenize_action_space_snapshot(actionable_action_space_snapshot())


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

    vmapped_choices = jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))(
        params, state_tokens, state_mask, def_mask, keys
    )
    scalar_choices = jnp.asarray(
        [
            sample_target(
                params,
                _slice_tree(state_tokens, index),
                state_mask[index],
                def_mask[index],
                keys[index],
            )
            for index in range(2)
        ]
    )
    vmapped_replayed_logp = jax.vmap(score_target, in_axes=(None, 0, 0, 0, 0))(
        params, state_tokens, state_mask, def_mask, vmapped_choices
    )
    scalar_replayed_logp = jnp.asarray(
        [
            score_target(
                params,
                _slice_tree(state_tokens, index),
                state_mask[index],
                def_mask[index],
                scalar_choices[index],
            )
            for index in range(2)
        ]
    )

    assert jnp.array_equal(vmapped_choices, scalar_choices)
    assert jnp.allclose(vmapped_replayed_logp, scalar_replayed_logp)


def test_vmap_score_action_matches_scalar_rows_for_vmapped_samples():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    action_tokens, action_mask = _two_row_action_batch()
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(20), 2)

    choices = jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
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

    vmapped_choices = jax.vmap(
        sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0)
    )(params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, keys)
    scalar_choices = [
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
    vmapped_replayed_logp = jax.vmap(
        score_action, in_axes=(None, 0, 0, 0, 0, 0, 0)
    )(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        vmapped_choices,
    )
    scalar_replayed_logp = jnp.asarray(
        [
            score_action(
                params,
                _slice_tree(state_tokens, index),
                state_mask[index],
                selected_defs[index],
                _slice_tree(action_tokens, index),
                action_mask[index],
                scalar_choice,
            )
            for index, scalar_choice in enumerate(scalar_choices)
        ]
    )

    for index, scalar_choice in enumerate(scalar_choices):
        _assert_choice_equal(_slice_tree(vmapped_choices, index), scalar_choice)
    assert jnp.allclose(vmapped_replayed_logp, scalar_replayed_logp)


def test_vmap_sample_action_uses_local_padding_width_not_model_config():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    small_action = _action_tree()
    wide_action = tokenize_action_space_snapshot(
        _wide_action_space_snapshot(candidates=10, side_terms=6)
    )
    action_tokens, action_mask = stack_token_trees([small_action, wide_action])
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(31), 2)

    choices = jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, keys
    )
    logp = jax.vmap(score_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, choices
    )

    assert choices["left_mask"].shape == choices["left_valid_mask"].shape
    assert choices["right_mask"].shape == choices["right_valid_mask"].shape
    assert int(jnp.sum(choices["left_valid_mask"][1])) == 6
    assert int(jnp.sum(choices["right_valid_mask"][1])) == 6
    assert bool(jnp.all(jnp.isfinite(logp)))


def test_width_one_vmap_sample_action_matches_scalar_with_same_key():
    params = _params()
    state_tokens, state_mask = stack_token_trees([_state_tree()])
    action_tokens, action_mask = stack_token_trees([_action_tree()])
    selected_defs = jnp.asarray([0], dtype=jnp.int32)
    key = jax.random.PRNGKey(30)
    keys = key[None, :]

    vmapped_choice = jax.vmap(
        sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0)
    )(params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, keys)
    scalar_choice = sample_action(
        params,
        _slice_tree(state_tokens, 0),
        state_mask[0],
        selected_defs[0],
        _slice_tree(action_tokens, 0),
        action_mask[0],
        key,
    )
    vmapped_replayed_logp = jax.vmap(
        score_action, in_axes=(None, 0, 0, 0, 0, 0, 0)
    )(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        vmapped_choice,
    )
    scalar_replayed_logp = score_action(
        params,
        _slice_tree(state_tokens, 0),
        state_mask[0],
        selected_defs[0],
        _slice_tree(action_tokens, 0),
        action_mask[0],
        scalar_choice,
    )

    _assert_choice_equal(_slice_tree(vmapped_choice, 0), scalar_choice)
    assert jnp.allclose(vmapped_replayed_logp[0], scalar_replayed_logp)
