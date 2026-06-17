import numpy as np
import pytest

from gristmill_symbolics.reinforce import (
    BaselineConfig,
    FinalColumnMetrics,
    RewardConfig,
    compute_advantages,
    compute_rewards,
)
from gristmill_symbolics.reinforce.types import TrainingError


def _policy():
    config = PolicyConfig(d_model=8, max_candidates=8, max_side_terms=4)
    return PolicyState(
        config=config,
        params=init_policy_params(config, jax.random.PRNGKey(0)),
    )


def test_compute_rewards_uses_float64_log_flops_improvement():
    final = FinalColumnMetrics(
        initial_log_flops=np.asarray([10.0, 2.0 + 1.0e-12], dtype=np.float64),
        final_log_flops=np.asarray([7.5, 2.0], dtype=np.float64),
        stopped=np.asarray([False, True], dtype=bool),
        max_steps=np.asarray([True, False], dtype=bool),
    )

    reward = compute_rewards(final, RewardConfig())

    assert reward.dtype == np.float64
    assert reward.tolist()[0] == pytest.approx(2.5)
    assert reward.tolist()[1] == pytest.approx(1.0e-12)


def test_compute_rewards_rejects_non_1d_log_flops():
    final = FinalColumnMetrics(
        initial_log_flops=np.asarray([[10.0, 9.0]], dtype=np.float64),
        final_log_flops=np.asarray([[7.5, 6.0]], dtype=np.float64),
        stopped=np.asarray([[False, True]], dtype=bool),
        max_steps=np.asarray([[True, False]], dtype=bool),
    )

    with pytest.raises(TrainingError, match="initial_log_flops.*1D"):
        compute_rewards(final, RewardConfig())


def test_compute_rewards_rejects_metric_shape_mismatch():
    final = FinalColumnMetrics(
        initial_log_flops=np.asarray([10.0, 9.0], dtype=np.float64),
        final_log_flops=np.asarray([7.5, 6.0], dtype=np.float64),
        stopped=np.asarray([False], dtype=bool),
        max_steps=np.asarray([True, False], dtype=bool),
    )

    with pytest.raises(TrainingError, match="stopped.*shape"):
        compute_rewards(final, RewardConfig())


def test_compute_advantages_batch_mean_and_optional_standardization():
    reward = np.asarray([1.0, 3.0, 5.0], dtype=np.float64)

    advantage = compute_advantages(reward, BaselineConfig())
    standardized = compute_advantages(
        reward,
        BaselineConfig(standardize=True, epsilon=1.0e-12),
    )

    assert advantage.dtype == np.float64
    assert advantage.tolist() == pytest.approx([-2.0, 0.0, 2.0])
    assert float(np.mean(standardized)) == pytest.approx(0.0)
    assert float(np.std(standardized)) == pytest.approx(1.0)


def test_reinforce_loss_is_column_normalized_not_decision_normalized():
    target_mask = jnp.asarray([[True, True], [True, False]])
    action_mask = jnp.asarray([[False, True], [False, False]])
    rollout = type(
        "Rollout",
        (),
        {"target_score_mask": target_mask, "action_score_mask": action_mask},
    )()
    scores = ScoreOutputs(
        target_logp=jnp.asarray([[-1.0, -2.0], [-3.0, 0.0]], dtype=jnp.float32),
        action_logp=jnp.asarray([[0.0, -5.0], [0.0, 0.0]], dtype=jnp.float32),
    )
    advantage = np.asarray([2.0, -1.0], dtype=np.float64)

    loss, diagnostics = reinforce_loss(rollout, scores, advantage, LossConfig())

    expected_col0 = -1.0 + -3.0
    expected_col1 = -2.0 + -5.0
    expected = -((2.0 * expected_col0) + (-1.0 * expected_col1)) / 2.0
    decision_normalized = -((2.0 * expected_col0) + (-1.0 * expected_col1)) / 4.0
    assert float(loss) == pytest.approx(expected)
    assert float(loss) != pytest.approx(decision_normalized)
    assert diagnostics.target_score_count == 3
    assert diagnostics.action_score_count == 1


def test_masked_dummy_entries_do_not_affect_loss_or_metrics():
    rollout = type(
        "Rollout",
        (),
        {
            "target_score_mask": jnp.asarray([[True, False]], dtype=jnp.bool_),
            "action_score_mask": jnp.asarray([[False, False]], dtype=jnp.bool_),
        },
    )()
    scores = ScoreOutputs(
        target_logp=jnp.asarray([[-2.0, -9999.0]], dtype=jnp.float32),
        action_logp=jnp.asarray([[0.0, -9999.0]], dtype=jnp.float32),
    )

    loss, diagnostics = reinforce_loss(
        rollout,
        scores,
        np.asarray([1.0, 1.0], dtype=np.float64),
        LossConfig(),
    )

    assert float(loss) == pytest.approx(1.0)
    assert diagnostics.column_logp_sum.tolist() == pytest.approx([-2.0, 0.0])
    assert diagnostics.target_score_count == 1
    assert diagnostics.action_score_count == 0
    assert diagnostics.target_logp_mean == pytest.approx(-2.0)
    assert diagnostics.action_logp_mean == pytest.approx(0.0)


def test_reinforce_loss_rejects_no_scored_terms_by_default():
    rollout = type(
        "Rollout",
        (),
        {
            "target_score_mask": jnp.zeros((1, 2), dtype=jnp.bool_),
            "action_score_mask": jnp.zeros((1, 2), dtype=jnp.bool_),
        },
    )()
    scores = ScoreOutputs(
        target_logp=jnp.zeros((1, 2), dtype=jnp.float32),
        action_logp=jnp.zeros((1, 2), dtype=jnp.float32),
    )

    with pytest.raises(TrainingError, match="no scored policy terms"):
        reinforce_loss(
            rollout,
            scores,
            np.asarray([1.0, -1.0], dtype=np.float64),
            LossConfig(),
        )


def test_score_rollout_recomputes_logp_and_ignores_sampled_logp():
    policy = _policy()
    table, _final = collect_rollout_batch(
        policy,
        [actionable_state()],
        RolloutConfig(batch_size=1, max_steps=1, seed=9),
        update_index=0,
        root_key=jax.random.PRNGKey(9),
    )
    table = type(table)(
        **{
            **table.__dict__,
            "sampled_target_logp": jnp.asarray([[1234.0]], dtype=jnp.float32),
        }
    )

    scores = score_rollout(policy, table)

    assert scores.target_logp.shape == table.target_score_mask.shape
    assert scores.action_logp.shape == table.action_score_mask.shape
    assert bool(jnp.isfinite(scores.target_logp[0, 0]))
    assert float(scores.target_logp[0, 0]) != pytest.approx(1234.0)


def test_score_rollout_gates_masked_invalid_action_entries_before_policy_scoring():
    policy = _policy()
    table, _final = collect_rollout_batch(
        policy,
        [actionable_state()],
        RolloutConfig(batch_size=1, max_steps=1, seed=10),
        update_index=0,
        root_key=jax.random.PRNGKey(10),
    )
    bad_choice = {
        **table.action_choice,
        "candidate_index": jnp.asarray([[999]], dtype=jnp.int32),
    }
    masked_table = type(table)(
        **{
            **table.__dict__,
            "action_choice": bad_choice,
            "action_score_mask": jnp.asarray([[False]], dtype=jnp.bool_),
        }
    )

    scores = score_rollout(policy, masked_table)

    assert float(scores.action_logp[0, 0]) == pytest.approx(0.0)


def test_score_rollout_gates_masked_invalid_target_entries_before_policy_scoring():
    policy = _policy()
    table, _final = collect_rollout_batch(
        policy,
        [actionable_state()],
        RolloutConfig(batch_size=1, max_steps=1, seed=11),
        update_index=0,
        root_key=jax.random.PRNGKey(11),
    )
    masked_table = type(table)(
        **{
            **table.__dict__,
            "target_choice": jnp.asarray([[999]], dtype=jnp.int32),
            "target_score_mask": jnp.asarray([[False]], dtype=jnp.bool_),
            "action_score_mask": jnp.asarray([[False]], dtype=jnp.bool_),
        }
    )

    scores = score_rollout(policy, masked_table)

    assert float(scores.target_logp[0, 0]) == pytest.approx(0.0)


def test_score_rollout_rejects_action_scored_without_target_scored():
    policy = _policy()
    table, _final = collect_rollout_batch(
        policy,
        [actionable_state()],
        RolloutConfig(batch_size=1, max_steps=1, seed=12),
        update_index=0,
        root_key=jax.random.PRNGKey(12),
    )
    invalid_table = type(table)(
        **{
            **table.__dict__,
            "target_score_mask": jnp.asarray([[False]], dtype=jnp.bool_),
            "action_score_mask": jnp.asarray([[True]], dtype=jnp.bool_),
        }
    )

    with pytest.raises(TrainingError, match="action_score_mask=true"):
        score_rollout(policy, invalid_table)


def test_reinforce_loss_value_preserves_score_rollout_gradients():
    policy = _policy()
    table, _final = collect_rollout_batch(
        policy,
        [actionable_state()],
        RolloutConfig(batch_size=1, max_steps=1, seed=13),
        update_index=0,
        root_key=jax.random.PRNGKey(13),
    )
    advantage = jnp.asarray([1.0], dtype=jnp.float32)

    def objective(params):
        policy_with_params = PolicyState(config=policy.config, params=params)
        scores = score_rollout(policy_with_params, table, check_finite=False)
        loss, _column_logp_sum = _reinforce_loss_value(table, scores, advantage)
        return loss

    loss, grads = jax.value_and_grad(objective)(policy.params)
    floating_grads = [
        np.asarray(leaf)
        for leaf in jax.tree_util.tree_leaves(grads)
        if np.issubdtype(np.asarray(leaf).dtype, np.floating)
    ]

    assert np.isfinite(np.asarray(loss))
    assert floating_grads
    assert all(np.all(np.isfinite(grad)) for grad in floating_grads)
    assert any(np.any(np.abs(grad) > 0.0) for grad in floating_grads)
