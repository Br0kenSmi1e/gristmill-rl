from pathlib import Path

import numpy as np
import pytest

import gristmill_rl.rollout as rollout_module
from gristmill_rl.actions import SampledAction
from gristmill_rl.features import FeatureConfig
from gristmill_rl.model import PolicyValueModel
from gristmill_rl.rollout import RolloutConfig, run_policy_rollout
from gristmill_symbolics import GristmillSymbolicsError, TensorComputation

from .rl_fixtures import actionable_comp


ROOT = Path(__file__).resolve().parents[2]
BASIC_FIXTURE = ROOT / "tests" / "fixtures" / "repr" / "basic.json"


class _FakeActionSpace:
    def __init__(self):
        self.def_index = 0

    def snapshot(self):
        return {
            "def_index": 0,
            "candidate_templates": [
                {
                    "left_definition": {"terms": []},
                    "right_definition": {"terms": []},
                    "rewritten_definition": {"terms": []},
                }
            ],
        }


class _FakeComp:
    def __init__(self, state="root"):
        self.state = state

    def clone(self):
        return _FakeComp(self.state)

    def snapshot(self):
        return {
            "ranges": [],
            "tensors": [],
            "definitions": [
                {
                    "terms": [],
                }
            ],
            "state": self.state,
        }

    def log_total_flops(self):
        if self.state in {"zero_terminal", "zero_nonterminal"}:
            raise GristmillSymbolicsError("ZeroTotalFlops")
        return 5.0

    def next_action_space(self, start_from):
        if self.state == "zero_terminal":
            return None
        return _FakeActionSpace()

    def apply_decision_with_space(self, space, decision):
        self.state = "zero_terminal"


def _single_fake_action_proposal(*args, **kwargs):
    def proposal(snapshot):
        return [
            SampledAction(
                decision={
                    "candidate_index": 0,
                    "left_mask": [],
                    "right_mask": [],
                },
                prior=1.0,
            )
        ]

    return proposal


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


def test_policy_rollout_returns_zero_step_result_for_zero_flop_terminal_comp():
    terminal = TensorComputation.from_json_string(
        '{"ranges":[],"tensors":[],"definitions":[]}'
    )
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)

    result = run_policy_rollout(
        terminal,
        model=model,
        feature_config=FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3),
        config=RolloutConfig(max_steps=2, simulations=1, actions_per_node=1, sample_attempts=2),
        rng=np.random.default_rng(0),
    )

    assert result.steps == 0
    assert len(result.trace.records) == 0
    assert result.terminal
    assert result.initial_log_flops == 0.0
    assert result.final_log_flops == 0.0
    assert result.comp.snapshot() == terminal.snapshot()


def test_policy_rollout_allows_zero_flop_terminal_child(monkeypatch):
    monkeypatch.setattr(
        rollout_module, "_proposal_for_node", _single_fake_action_proposal
    )

    result = run_policy_rollout(
        _FakeComp(),
        model=object(),
        feature_config=FeatureConfig(
            max_candidates=1, max_left_terms=0, max_right_terms=0
        ),
        config=RolloutConfig(
            max_steps=1, simulations=1, actions_per_node=1, sample_attempts=1
        ),
        rng=np.random.default_rng(0),
    )

    assert result.steps == 1
    assert len(result.trace.records) == 1
    assert result.initial_log_flops == 5.0
    assert result.final_log_flops == 0.0
    assert result.comp.state == "zero_terminal"


def test_policy_rollout_allows_zero_flop_terminal_child_without_simulations(
    monkeypatch,
):
    monkeypatch.setattr(
        rollout_module, "_proposal_for_node", _single_fake_action_proposal
    )

    result = run_policy_rollout(
        _FakeComp(),
        model=object(),
        feature_config=FeatureConfig(
            max_candidates=1, max_left_terms=0, max_right_terms=0
        ),
        config=RolloutConfig(
            max_steps=1, simulations=0, actions_per_node=1, sample_attempts=1
        ),
        rng=np.random.default_rng(0),
    )

    assert result.steps == 1
    assert result.terminal
    assert result.final_log_flops == 0.0
    assert result.comp.state == "zero_terminal"


def test_policy_rollout_propagates_zero_flop_nonterminal_metric_error():
    with pytest.raises(GristmillSymbolicsError, match="ZeroTotalFlops"):
        run_policy_rollout(
            _FakeComp("zero_nonterminal"),
            model=object(),
            feature_config=FeatureConfig(
                max_candidates=1, max_left_terms=0, max_right_terms=0
            ),
            config=RolloutConfig(max_steps=0),
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
