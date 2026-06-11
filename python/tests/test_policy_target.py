import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics.policy import (
    PolicyConfig,
    init_policy_params,
    sample_target,
    score_target,
    tokenize_state_snapshot,
)
from tests.policy_fixtures import actionable_state_snapshot


def _params():
    return init_policy_params(PolicyConfig(d_model=16), jax.random.PRNGKey(0))


def _state():
    return tokenize_state_snapshot(actionable_state_snapshot())


def test_target_score_maps_stop_minus_one_to_stop_logit():
    params = _params()
    state_tokens, state_mask = _state()
    def_mask = jnp.asarray([True])

    stop_logp = score_target(
        params, state_tokens, state_mask, def_mask, jnp.asarray(-1, dtype=jnp.int32)
    )
    def_logp = score_target(
        params, state_tokens, state_mask, def_mask, jnp.asarray(0, dtype=jnp.int32)
    )

    assert stop_logp.shape == ()
    assert def_logp.shape == ()
    assert float(stop_logp) < float(def_logp)


def test_target_all_masked_definitions_make_stop_probability_one():
    params = _params()
    state_tokens, state_mask = _state()
    def_mask = jnp.asarray([False])

    logp = score_target(
        params, state_tokens, state_mask, def_mask, jnp.asarray(-1, dtype=jnp.int32)
    )

    assert float(logp) == pytest.approx(0.0)


def test_target_sampling_never_returns_masked_definition():
    params = _params()
    state_tokens, state_mask = _state()
    def_mask = jnp.asarray([False])

    choice, logp = sample_target(
        params, state_tokens, state_mask, def_mask, jax.random.PRNGKey(1)
    )

    assert int(choice) == -1
    assert float(logp) == pytest.approx(0.0)


def test_target_scoring_rejects_masked_definition_for_concrete_input():
    params = _params()
    state_tokens, state_mask = _state()
    def_mask = jnp.asarray([False])

    with pytest.raises(ValueError, match="masked definition"):
        score_target(params, state_tokens, state_mask, def_mask, 0)


def test_target_sampling_is_deterministic_for_same_rng():
    params = _params()
    state_tokens, state_mask = _state()
    def_mask = jnp.asarray([True])

    left = sample_target(params, state_tokens, state_mask, def_mask, jax.random.PRNGKey(123))
    right = sample_target(
        params, state_tokens, state_mask, def_mask, jax.random.PRNGKey(123)
    )

    assert int(left[0]) == int(right[0])
    assert float(left[1]) == pytest.approx(float(right[1]))
