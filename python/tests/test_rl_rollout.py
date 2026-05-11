from pathlib import Path

import numpy as np
import pytest

from gristmill_rl.features import FeatureConfig
from gristmill_rl.model import PolicyValueModel
from gristmill_rl.rollout import RolloutConfig, run_policy_rollout
from gristmill_symbolics import GristmillSymbolicsError, TensorComputation

from .rl_fixtures import actionable_comp


ROOT = Path(__file__).resolve().parents[2]
BASIC_FIXTURE = ROOT / "tests" / "fixtures" / "repr" / "basic.json"


def test_policy_rollout_returns_trace_and_rewritten_comp():
    comp = actionable_comp()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)

    result = run_policy_rollout(
        comp,
        model=model,
        feature_config=FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3),
        config=RolloutConfig(
            max_steps=1,
            simulations=2,
            actions_per_node=1,
            sample_attempts=4,
            temperature=0.0,
            c_puct=1.5,
        ),
        rng=np.random.default_rng(0),
    )

    assert result.steps == 1
    assert len(result.trace.records) == 1
    assert result.final_log_flops == result.comp.log_total_flops()
    assert result.comp.snapshot() != comp.snapshot()
    assert result.valid_action_counts == [1]


def test_policy_rollout_on_zero_flop_terminal_comp_raises():
    terminal = TensorComputation.from_json_string(
        '{"ranges":[],"tensors":[],"definitions":[]}'
    )
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)

    with pytest.raises(GristmillSymbolicsError, match="ZeroTotalFlops"):
        run_policy_rollout(
            terminal,
            model=model,
            feature_config=FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3),
            config=RolloutConfig(max_steps=2, simulations=1, actions_per_node=1, sample_attempts=2),
            rng=np.random.default_rng(0),
        )


def test_policy_rollout_preserves_terminal_fixture_log_flops():
    comp = TensorComputation.load_json(BASIC_FIXTURE)
    expected_log_flops = float(comp.log_total_flops())
    assert comp.next_action_space(0) is None
    assert expected_log_flops != 0.0
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)

    result = run_policy_rollout(
        comp,
        model=model,
        feature_config=FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3),
        config=RolloutConfig(max_steps=2, simulations=1, actions_per_node=1, sample_attempts=2),
        rng=np.random.default_rng(0),
    )

    assert result.steps == 0
    assert len(result.trace.records) == 0
    assert result.terminal
    assert result.initial_log_flops == expected_log_flops
    assert result.final_log_flops == expected_log_flops
    assert result.comp.snapshot() == comp.snapshot()
