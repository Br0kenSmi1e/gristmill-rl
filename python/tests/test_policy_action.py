import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics import RewriteStateRow, validate_decision
from gristmill_symbolics.policy import (
    PolicyConfig,
    action_choice_to_python,
    init_policy_params,
    sample_action,
    score_action,
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from tests.policy_fixtures import actionable_state, actionable_state_snapshot


def _params():
    return init_policy_params(
        PolicyConfig(d_model=16, max_candidates=8, max_side_terms=4),
        jax.random.PRNGKey(0),
    )


def _state():
    return tokenize_state_snapshot(actionable_state_snapshot())


def _action_space():
    state = actionable_state()
    space = state.action_space_for_def(0)
    assert space is not None
    return space


def _action_space_tokens():
    return tokenize_action_space_snapshot(_action_space().snapshot())


def _sample():
    params = _params()
    state_tokens, state_mask = _state()
    action_tokens, action_mask = _action_space_tokens()
    return (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        sample_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(0, dtype=jnp.int32),
            action_tokens,
            action_mask,
            jax.random.PRNGKey(1),
        ),
    )


def _trimmed_decision(choice):
    py_choice = action_choice_to_python(choice)
    return {
        "candidate_index": py_choice["candidate_index"],
        "left_mask": [
            keep
            for keep, valid in zip(py_choice["left_mask"], py_choice["left_valid_mask"])
            if valid
        ],
        "right_mask": [
            keep
            for keep, valid in zip(
                py_choice["right_mask"], py_choice["right_valid_mask"]
            )
            if valid
        ],
    }


def test_action_sample_returns_padded_choice_tree_and_finite_logp():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        (choice, logp),
    ) = _sample()

    assert set(choice) == {
        "candidate_index",
        "left_mask",
        "left_valid_mask",
        "right_mask",
        "right_valid_mask",
    }
    width = params["action"]["left_position_bias"].shape[0]
    assert choice["candidate_index"].shape == ()
    assert choice["left_mask"].shape == (width,)
    assert choice["left_valid_mask"].shape == (width,)
    assert choice["right_mask"].shape == (width,)
    assert choice["right_valid_mask"].shape == (width,)
    assert bool(jnp.any(choice["left_mask"] & choice["left_valid_mask"]))
    assert bool(jnp.any(choice["right_mask"] & choice["right_valid_mask"]))
    assert logp.shape == ()
    assert bool(jnp.isfinite(logp))

    replay = score_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        choice,
    )
    assert bool(jnp.isfinite(replay))


def test_action_score_replays_sampled_logp():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        (choice, sampled_logp),
    ) = _sample()

    replayed_logp = score_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        choice,
    )

    assert float(replayed_logp) == pytest.approx(float(sampled_logp))


def test_action_final_bit_constraint_prevents_empty_side_masks():
    params = _params()
    params = {
        **params,
        "action": {
            **params["action"],
            "left_w": jnp.zeros_like(params["action"]["left_w"]),
            "right_w": jnp.zeros_like(params["action"]["right_w"]),
            "left_context_w": jnp.zeros_like(params["action"]["left_context_w"]),
            "left_position_bias": jnp.full_like(
                params["action"]["left_position_bias"], -100.0
            ),
            "right_position_bias": jnp.full_like(
                params["action"]["right_position_bias"], -100.0
            ),
        },
    }
    state_tokens, state_mask = _state()
    action_tokens, action_mask = _action_space_tokens()

    choice, _ = sample_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        jax.random.PRNGKey(2),
    )

    assert bool(jnp.any(choice["left_mask"] & choice["left_valid_mask"]))
    assert bool(jnp.any(choice["right_mask"] & choice["right_valid_mask"]))


def test_action_score_rejects_concrete_empty_side_mask():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        (choice, _),
    ) = _sample()
    empty_left = {
        **choice,
        "left_mask": jnp.zeros_like(choice["left_mask"], dtype=jnp.bool_),
    }

    with pytest.raises(ValueError, match="empty left_mask"):
        score_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(0, dtype=jnp.int32),
            action_tokens,
            action_mask,
            empty_left,
        )


def test_action_score_rejects_concrete_illegal_candidate_index():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        (choice, _),
    ) = _sample()
    bad_candidate = {
        **choice,
        "candidate_index": jnp.asarray(99, dtype=jnp.int32),
    }

    with pytest.raises(ValueError, match="candidate_index"):
        score_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(0, dtype=jnp.int32),
            action_tokens,
            action_mask,
            bad_candidate,
        )


def test_action_score_traced_invalid_candidate_index_returns_negative_infinity():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        (choice, _),
    ) = _sample()
    candidate_indices = jnp.asarray([-1, 2, 99], dtype=jnp.int32)

    scores = jax.jit(
        lambda candidates: jax.vmap(
            lambda candidate: score_action(
                params,
                state_tokens,
                state_mask,
                jnp.asarray(0, dtype=jnp.int32),
                action_tokens,
                action_mask,
                {**choice, "candidate_index": candidate},
            )
        )(candidates)
    )(candidate_indices)

    assert scores.shape == (3,)
    assert bool(jnp.all(jnp.isneginf(scores)))


def test_action_sample_validates_through_scalar_and_row_boundaries():
    (
        _params,
        _state_tokens,
        _state_mask,
        _action_tokens,
        _action_mask,
        (choice, _),
    ) = _sample()
    scalar_space = _action_space()
    validate_decision(scalar_space, _trimmed_decision(choice))

    row = RewriteStateRow.from_states([actionable_state()])
    spaces = row.query_action_spaces_for_row([0], [True])
    py_choice = action_choice_to_python(choice)
    validated = row.validate_actions_for_row(spaces, [py_choice], [True])

    assert validated.entry_kinds() == ["valid"]
