import numpy as np
import pytest

from gristmill_symbolics import RewriteState, TensorComputation
from transformer_policy.decoder import sample_step, score_step
from transformer_policy.types import T

from .transformer_policy_fixtures import actionable_state


class PreferenceScorer:
    def score_next(self, context_tokens, decision_prefix, legal_next_tokens):
        scores = []
        for token in legal_next_tokens:
            payload = token.payload_dict()
            if token.kind == "DEF":
                scores.append(1.0e6)
            elif token.kind == "STOP":
                scores.append(-1.0e6)
            elif token.kind == "CAND" and payload.get("candidate_index") == 0:
                scores.append(1.0e6)
            elif token.kind.endswith("KEEP"):
                scores.append(1.0e6)
            elif token.kind == "END":
                scores.append(1.0e6)
            else:
                scores.append(-1.0e6)
        return np.asarray(scores, dtype=np.float32)


class StopScorer:
    def score_next(self, context_tokens, decision_prefix, legal_next_tokens):
        return np.asarray(
            [
                1.0e6 if token.kind == "STOP" else -1.0e6
                for token in legal_next_tokens
            ],
            dtype=np.float32,
        )


def empty_state():
    text = """
    {
      "ranges": [{"id": 0, "size": 8}],
      "tensors": [{"id": 0, "symmetry": []}],
      "definitions": [
        {
          "base": 0,
          "ext_indices": [{"id": 0, "range": 0}],
          "terms": [
            {
              "coeff": [1, 1],
              "sum_indices": [],
              "factors": [{"tensor": 0, "indices": [0]}]
            }
          ]
        }
      ]
    }
    """
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def test_sample_step_can_stop_when_mask_is_empty():
    sample = sample_step(empty_state(), StopScorer(), np.random.default_rng(0))

    assert sample.stopped
    assert sample.decision is None
    assert sample.def_attempts == ()
    assert sample.decision_tokens == (T("STOP"),)


def test_sample_step_returns_rewrite_decision_that_rust_can_apply():
    state = actionable_state()

    sample = sample_step(state, PreferenceScorer(), np.random.default_rng(0))

    assert not sample.stopped
    assert sample.def_index == 0
    assert sample.decision == {
        "candidate_index": 0,
        "left_mask": [True],
        "right_mask": [True, True],
    }
    assert sample.action_space is not None
    state.step_with_space(sample.action_space, sample.decision)


def test_score_step_replays_sample_log_probability():
    state = actionable_state()
    sample = sample_step(state, PreferenceScorer(), np.random.default_rng(0))

    rescored = score_step(actionable_state(), PreferenceScorer(), sample)

    assert rescored == pytest.approx(sample.log_prob)


def test_score_step_rejects_invalid_mask_length():
    sample = sample_step(actionable_state(), PreferenceScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=False,
        def_index=sample.def_index,
        action_space=sample.action_space,
        decision={
            "candidate_index": sample.decision["candidate_index"],
            "left_mask": [True, False],
            "right_mask": sample.decision["right_mask"],
        },
        log_prob=sample.log_prob,
        def_attempts=sample.def_attempts,
        decision_tokens=sample.decision_tokens,
    )

    with pytest.raises(ValueError, match="invalid left_mask length"):
        score_step(actionable_state(), PreferenceScorer(), invalid)
