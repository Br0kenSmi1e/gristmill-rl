import jax
import jax.numpy as jnp

from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.model.transformer_action_selector.api import (
    sample_action,
    score_action,
    score_target,
)
from gristmill_symbolics.model.transformer_action_selector.tokenize import (
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from tests.policy_fixtures import (
    actionable_action_space_snapshot,
    actionable_state_snapshot,
)


def _params():
    model = TransformerActionSelectorModel(
        batch_size=1,
        max_steps=1,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
        d_model=16,
    )
    return model.init_params(jax.random.PRNGKey(0))


def _gradient_params():
    model = TransformerActionSelectorModel(
        batch_size=1,
        max_steps=1,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
        d_model=16,
        stop_bias_init=0.0,
    )
    return model.init_params(jax.random.PRNGKey(0))


def _state():
    return tokenize_state_snapshot(actionable_state_snapshot())


def _action_space():
    return tokenize_action_space_snapshot(actionable_action_space_snapshot())


def _sampled_action(params):
    state_tokens, state_mask = _state()
    action_tokens, action_mask = _action_space()
    choice = sample_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        jax.random.PRNGKey(1),
    )
    return state_tokens, state_mask, action_tokens, action_mask, choice


def _assert_floating_tree_finite_and_nonzero(tree):
    leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    assert leaves
    for leaf in leaves:
        assert bool(jnp.all(jnp.isfinite(leaf)))
    assert any(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in leaves)


def test_jit_score_target_returns_finite_scalar():
    params = _params()
    state_tokens, state_mask = _state()
    def_mask = jnp.asarray([True])

    logp = jax.jit(score_target)(
        params,
        state_tokens,
        state_mask,
        def_mask,
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert logp.shape == ()
    assert bool(jnp.isfinite(logp))


def test_jit_score_action_returns_finite_scalar_for_valid_sample():
    params = _params()
    state_tokens, state_mask, action_tokens, action_mask, choice = _sampled_action(params)

    logp = jax.jit(score_action)(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        choice,
    )

    assert logp.shape == ()
    assert bool(jnp.isfinite(logp))


def test_score_target_gradients_are_finite():
    params = _gradient_params()
    state_tokens, state_mask = _state()
    def_mask = jnp.asarray([True])

    grads = jax.grad(
        lambda policy_params: score_target(
            policy_params,
            state_tokens,
            state_mask,
            def_mask,
            jnp.asarray(0, dtype=jnp.int32),
        )
    )(params)

    _assert_floating_tree_finite_and_nonzero(grads)


def test_score_action_gradients_are_finite_for_valid_sample():
    params = _gradient_params()
    state_tokens, state_mask, action_tokens, action_mask, choice = _sampled_action(params)

    grads = jax.grad(
        lambda policy_params: score_action(
            policy_params,
            state_tokens,
            state_mask,
            jnp.asarray(0, dtype=jnp.int32),
            action_tokens,
            action_mask,
            choice,
        )
    )(params)

    _assert_floating_tree_finite_and_nonzero(grads)
