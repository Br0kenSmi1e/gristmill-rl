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
from gristmill_symbolics.policy.constants import TOKEN_KIND
from tests.policy_fixtures import actionable_state, actionable_state_snapshot


def _params():
    return init_policy_params(
        PolicyConfig(d_model=16),
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


def _with_action_space_start_def_index(action_tokens, def_index):
    start = (
        jnp.asarray(action_tokens["token_kind"])
        == jnp.asarray(int(TOKEN_KIND.ACTION_SPACE_START), dtype=jnp.int32)
    )
    return {
        **action_tokens,
        "def_index": jnp.where(
            start,
            jnp.asarray(def_index, dtype=jnp.int32),
            action_tokens["def_index"],
        ),
    }


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


def test_action_sample_returns_padded_choice_tree_and_finite_replay_logp():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        choice,
    ) = _sample()

    assert set(choice) == {
        "candidate_index",
        "left_mask",
        "left_valid_mask",
        "right_mask",
        "right_valid_mask",
    }
    width = choice["left_mask"].shape[0]
    assert choice["candidate_index"].shape == ()
    assert choice["left_mask"].shape == (width,)
    assert choice["left_valid_mask"].shape == (width,)
    assert choice["right_mask"].shape == (width,)
    assert choice["right_valid_mask"].shape == (width,)
    assert bool(jnp.any(choice["left_mask"] & choice["left_valid_mask"]))
    assert bool(jnp.any(choice["right_mask"] & choice["right_valid_mask"]))
    replay = score_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        choice,
    )
    assert replay.shape == ()
    assert bool(jnp.isfinite(replay))


def test_action_score_deterministically_replays_choice_logp():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        choice,
    ) = _sample()

    left_replay_logp = score_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        choice,
    )
    right_replay_logp = score_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        choice,
    )

    assert float(left_replay_logp) == pytest.approx(float(right_replay_logp))


def test_action_final_bit_constraint_prevents_empty_side_masks():
    params = _params()
    params = {
        **params,
        "action": {
            **params["action"],
            "left_w": jnp.zeros_like(params["action"]["left_w"]),
            "right_w": jnp.zeros_like(params["action"]["right_w"]),
            "left_context_w": jnp.zeros_like(params["action"]["left_context_w"]),
            "left_bias": jnp.asarray(-100.0, dtype=jnp.float32),
            "right_bias": jnp.asarray(-100.0, dtype=jnp.float32),
        },
    }
    state_tokens, state_mask = _state()
    action_tokens, action_mask = _action_space_tokens()

    choice = sample_action(
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
        choice,
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
        choice,
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
        choice,
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


def test_action_rejects_concrete_stop_selected_def_index():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        choice,
    ) = _sample()

    with pytest.raises(ValueError, match="selected_def_index"):
        score_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(-1, dtype=jnp.int32),
            action_tokens,
            action_mask,
            choice,
        )

    with pytest.raises(ValueError, match="selected_def_index"):
        sample_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(-1, dtype=jnp.int32),
            action_tokens,
            action_mask,
            jax.random.PRNGKey(3),
        )


def test_action_rejects_concrete_missing_selected_def_index():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        choice,
    ) = _sample()

    with pytest.raises(ValueError, match="selected_def_index"):
        score_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(1, dtype=jnp.int32),
            action_tokens,
            action_mask,
            choice,
        )

    with pytest.raises(ValueError, match="selected_def_index"):
        sample_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(1, dtype=jnp.int32),
            action_tokens,
            action_mask,
            jax.random.PRNGKey(4),
        )


def test_action_rejects_concrete_action_space_def_mismatch():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        choice,
    ) = _sample()
    mismatched_action_tokens = _with_action_space_start_def_index(action_tokens, 1)

    with pytest.raises(ValueError, match="action space"):
        score_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(0, dtype=jnp.int32),
            mismatched_action_tokens,
            action_mask,
            choice,
        )

    with pytest.raises(ValueError, match="action space"):
        sample_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(0, dtype=jnp.int32),
            mismatched_action_tokens,
            action_mask,
            jax.random.PRNGKey(5),
        )


def test_action_score_traced_invalid_selected_def_index_returns_negative_infinity():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        choice,
    ) = _sample()
    selected_defs = jnp.asarray([-1, 1, 99], dtype=jnp.int32)

    scores = jax.jit(
        lambda indices: jax.vmap(
            lambda selected_def: score_action(
                params,
                state_tokens,
                state_mask,
                selected_def,
                action_tokens,
                action_mask,
                choice,
            )
        )(indices)
    )(selected_defs)

    assert scores.shape == (3,)
    assert bool(jnp.all(jnp.isneginf(scores)))


def test_action_score_traced_valid_mask_mismatch_returns_negative_infinity():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        choice,
    ) = _sample()
    bad_left_valid = ~choice["left_valid_mask"]
    bad_right_valid = ~choice["right_valid_mask"]

    scores = jax.jit(
        lambda left_valid, right_valid: jnp.asarray(
            [
                score_action(
                    params,
                    state_tokens,
                    state_mask,
                    jnp.asarray(0, dtype=jnp.int32),
                    action_tokens,
                    action_mask,
                    {**choice, "left_valid_mask": left_valid},
                ),
                score_action(
                    params,
                    state_tokens,
                    state_mask,
                    jnp.asarray(0, dtype=jnp.int32),
                    action_tokens,
                    action_mask,
                    {**choice, "right_valid_mask": right_valid},
                ),
            ]
        )
    )(bad_left_valid, bad_right_valid)

    assert scores.shape == (2,)
    assert bool(jnp.all(jnp.isneginf(scores)))


def test_action_score_traced_empty_side_mask_returns_negative_infinity():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        choice,
    ) = _sample()
    empty_left = jnp.zeros_like(choice["left_mask"], dtype=jnp.bool_)

    score = jax.jit(
        lambda left_mask: score_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(0, dtype=jnp.int32),
            action_tokens,
            action_mask,
            {**choice, "left_mask": left_mask},
        )
    )(empty_left)

    assert bool(jnp.isneginf(score))


def test_action_score_traced_padded_slot_selection_returns_negative_infinity():
    (
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        choice,
    ) = _sample()
    padded_left = jnp.zeros_like(choice["left_mask"], dtype=jnp.bool_).at[-1].set(True)

    score = jax.jit(
        lambda left_mask: score_action(
            params,
            state_tokens,
            state_mask,
            jnp.asarray(0, dtype=jnp.int32),
            action_tokens,
            action_mask,
            {**choice, "left_mask": left_mask},
        )
    )(padded_left)

    assert bool(jnp.isneginf(score))


def test_action_sample_validates_through_scalar_and_row_boundaries():
    (
        _params,
        _state_tokens,
        _state_mask,
        _action_tokens,
        _action_mask,
        choice,
    ) = _sample()
    scalar_space = _action_space()
    validate_decision(scalar_space, _trimmed_decision(choice))

    row = RewriteStateRow.from_states([actionable_state()])
    spaces = row.query_action_spaces_for_row([0], [True])
    py_choice = action_choice_to_python(choice)
    validated = row.validate_actions_for_row(spaces, [py_choice], [True])

    assert validated.entry_kinds() == ["valid"]
