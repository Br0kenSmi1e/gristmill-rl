import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig, init_policy_params
from gristmill_symbolics.reinforce import (
    PolicyState,
    RolloutConfig,
    collect_rollout_batch,
    make_rng_grid,
)
from gristmill_symbolics.reinforce.types import (
    CASE_ALREADY_FINISHED,
    CASE_EMPTY_ACTION_SPACE,
    CASE_STOP,
    CASE_VALID_ACTION,
    DECISION_ACTION,
    DECISION_TARGET,
)
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _policy(*, stop_bias_init=-20.0):
    config = PolicyConfig(
        d_model=8,
        max_candidates=8,
        max_side_terms=4,
        stop_bias_init=stop_bias_init,
    )
    return PolicyState(config=config, params=init_policy_params(config, jax.random.PRNGKey(0)))


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def test_make_rng_grid_uses_step_sample_decision_kind_axes():
    root = jax.random.PRNGKey(123)
    grid = make_rng_grid(root, update_index=7, max_steps=3, batch_size=2)
    expected = jax.random.split(jax.random.fold_in(root, 7), 3 * 2 * 2).reshape((3, 2, 2, 2))

    assert grid.shape == (3, 2, 2, 2)
    assert jnp.array_equal(grid, expected)
    assert jnp.array_equal(grid[0, 0, DECISION_TARGET], expected[0, 0, 0])
    assert jnp.array_equal(grid[0, 0, DECISION_ACTION], expected[0, 0, 1])


def test_width_one_rollout_stores_shared_state_and_valid_action_case():
    policy = _policy(stop_bias_init=-20.0)
    table, final = collect_rollout_batch(
        policy,
        [_state_from_json(actionable_json())],
        RolloutConfig(batch_size=1, max_steps=1, seed=5),
        update_index=0,
        root_key=jax.random.PRNGKey(5),
    )

    assert table.state_token_mask.shape[:2] == (1, 1)
    assert table.target_def_mask.shape[:2] == (1, 1)
    assert table.target_choice.shape == (1, 1)
    assert table.target_score_mask.tolist() == [[True]]
    assert table.action_score_mask.tolist() == [[True]]
    assert table.step_case.tolist() == [[CASE_VALID_ACTION]]
    assert table.action_choice["left_mask"].shape == (1, 1, policy.config.max_side_terms)
    assert final.initial_log_flops.shape == (1,)
    assert final.final_log_flops.shape == (1,)
    assert final.stopped.tolist() == [False]
    assert final.max_steps.tolist() == [True]


def test_rollout_records_stop_then_already_finished_without_action_score():
    policy = _policy(stop_bias_init=100.0)
    table, final = collect_rollout_batch(
        policy,
        [_state_from_json(actionable_json())],
        RolloutConfig(batch_size=1, max_steps=2, seed=6),
        update_index=0,
        root_key=jax.random.PRNGKey(6),
    )

    assert table.step_case.tolist() == [[CASE_STOP], [CASE_ALREADY_FINISHED]]
    assert table.target_score_mask.tolist() == [[True], [False]]
    assert table.action_score_mask.tolist() == [[False], [False]]
    assert final.stopped.tolist() == [True]
    assert final.max_steps.tolist() == [False]


def test_exact_empty_rollout_scores_target_only_and_keeps_sample_active_until_max_steps():
    policy = _policy(stop_bias_init=-20.0)
    table, final = collect_rollout_batch(
        policy,
        [_state_from_json(exact_empty_json())],
        RolloutConfig(batch_size=1, max_steps=3, seed=7),
        update_index=0,
        root_key=jax.random.PRNGKey(7),
    )

    assert table.step_case.tolist() == [
        [CASE_EMPTY_ACTION_SPACE],
        [CASE_EMPTY_ACTION_SPACE],
        [CASE_EMPTY_ACTION_SPACE],
    ]
    assert table.target_score_mask.tolist() == [[True], [True], [True]]
    assert table.action_score_mask.tolist() == [[False], [False], [False]]
    assert final.stopped.tolist() == [False]
    assert final.max_steps.tolist() == [True]


def test_multi_sample_rollout_preserves_sample_axis_alignment():
    policy = _policy(stop_bias_init=-20.0)
    table, final = collect_rollout_batch(
        policy,
        [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())],
        RolloutConfig(batch_size=2, max_steps=1, seed=8),
        update_index=0,
        root_key=jax.random.PRNGKey(8),
    )

    assert table.step_case.shape == (1, 2)
    assert table.step_case[0, 0] == CASE_VALID_ACTION
    assert table.step_case[0, 1] == CASE_EMPTY_ACTION_SPACE
    assert table.action_score_mask.tolist() == [[True, False]]
    assert final.initial_log_flops.shape == (2,)
    assert final.final_log_flops.shape == (2,)
