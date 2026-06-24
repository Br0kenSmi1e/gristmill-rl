import json

import jax
import jax.numpy as jnp

from gristmill_symbolics.policy import PolicyConfig, init_policy_params
from gristmill_symbolics.reinforce.rollout import _collect_streamed_rollout_gradients
from gristmill_symbolics.reinforce.rollout import make_rng_grid
from gristmill_symbolics.reinforce.types import (
    DECISION_ACTION,
    DECISION_TARGET,
    PolicyState,
    RolloutConfig,
)
from tests.policy_fixtures import actionable_state


def test_make_rng_grid_uses_step_sample_decision_kind_axes():
    root = jax.random.PRNGKey(123)
    grid = make_rng_grid(root, update_index=7, max_steps=3, batch_size=2)
    expected = jax.random.split(jax.random.fold_in(root, 7), 3 * 2 * 2).reshape(
        (3, 2, 2, 2)
    )

    assert grid.shape == (3, 2, 2, 2)
    assert jnp.array_equal(grid, expected)
    assert jnp.array_equal(grid[0, 0, DECISION_TARGET], expected[0, 0, 0])
    assert jnp.array_equal(grid[0, 0, DECISION_ACTION], expected[0, 0, 1])


def _policy():
    config = PolicyConfig(d_model=8, stop_bias_init=-20.0)
    return PolicyState(
        config=config,
        params=init_policy_params(config, jax.random.PRNGKey(0)),
    )


def test_streamed_rollout_profiling_is_silent_by_default(capsys):
    _collect_streamed_rollout_gradients(
        _policy(),
        [actionable_state()],
        RolloutConfig(batch_size=1, max_steps=1, seed=5),
        update_index=0,
        root_key=jax.random.PRNGKey(5),
    )

    assert capsys.readouterr().err == ""


def test_streamed_rollout_profiling_emits_json_phase_events(monkeypatch, capsys):
    monkeypatch.setenv("GRISTMILL_PROFILE_ROLLOUT", "1")

    _collect_streamed_rollout_gradients(
        _policy(),
        [actionable_state()],
        RolloutConfig(batch_size=1, max_steps=1, seed=5),
        update_index=0,
        root_key=jax.random.PRNGKey(5),
    )

    events = [
        json.loads(line)
        for line in capsys.readouterr().err.splitlines()
        if line.strip()
    ]
    phase_names = {event["phase"] for event in events}

    assert events
    assert all(event["event"] == "rollout_phase" for event in events)
    assert all(event["elapsed_ms"] >= 0.0 for event in events)
    assert all(event["step"] == 0 for event in events)
    assert {
        "row_snapshots",
        "tokenize_state",
        "stack_state_tokens",
        "sample_target",
        "score_target_grad",
        "row_query_action_spaces",
        "tokenize_action_space",
        "stack_action_tokens",
        "sample_action",
        "score_action_grad",
        "action_choice_to_python",
        "row_validate_apply_actions",
    } <= phase_names
    assert any(event.get("state_token_len_max", 0) > 0 for event in events)
    assert any(event.get("action_token_len_max", 0) > 0 for event in events)
