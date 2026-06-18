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
