import numpy as np
import pytest

from gristmill_symbolics import RewriteState, TensorComputation
from transformer_policy.decoder import sample_step, score_step
from transformer_policy.types import Stage1Attempt, T

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


class RejectThenPreferenceScorer:
    def score_next(self, context_tokens, decision_prefix, legal_next_tokens):
        def_indices = {
            token.payload_dict()["def_index"]
            for token in legal_next_tokens
            if token.kind == "DEF"
        }
        preferred_def = 0 if 0 in def_indices else 1 if 1 in def_indices else None
        scores = []
        for token in legal_next_tokens:
            payload = token.payload_dict()
            if token.kind == "DEF" and payload.get("def_index") == preferred_def:
                scores.append(1.0e6)
            elif token.kind in {"DEF", "STOP"}:
                scores.append(-1.0e6)
            elif token.kind == "CAND" and payload.get("candidate_index") == 0:
                scores.append(1.0e6)
            elif token.kind.endswith("KEEP"):
                scores.append(1.0e6)
            else:
                scores.append(-1.0e6)
        return np.asarray(scores, dtype=np.float32)


class WrongShapeScorer:
    def score_next(self, context_tokens, decision_prefix, legal_next_tokens):
        return np.zeros((1, len(legal_next_tokens)), dtype=np.float32)


class NonFiniteScorer:
    def score_next(self, context_tokens, decision_prefix, legal_next_tokens):
        scores = np.zeros(len(legal_next_tokens), dtype=np.float32)
        scores[0] = np.nan
        return scores


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


def reject_then_actionable_state():
    text = """
    {
      "ranges": [{"id": 0, "size": 8}],
      "tensors": [
        {"id": 0, "symmetry": []},
        {"id": 1, "symmetry": []},
        {"id": 2, "symmetry": []},
        {"id": 3, "symmetry": []},
        {"id": 4, "symmetry": []}
      ],
      "definitions": [
        {
          "base": 4,
          "ext_indices": [{"id": 0, "range": 0}],
          "terms": [
            {
              "coeff": [1, 1],
              "sum_indices": [],
              "factors": [{"tensor": 0, "indices": [0]}]
            },
            {
              "coeff": [1, 1],
              "sum_indices": [],
              "factors": [{"tensor": 0, "indices": [0]}]
            }
          ]
        },
        {
          "base": 3,
          "ext_indices": [
            {"id": 0, "range": 0},
            {"id": 1, "range": 0}
          ],
          "terms": [
            {
              "coeff": [1, 1],
              "sum_indices": [{"id": 2, "range": 0}],
              "factors": [
                {"tensor": 0, "indices": [0, 2]},
                {"tensor": 1, "indices": [2, 1]}
              ]
            },
            {
              "coeff": [1, 1],
              "sum_indices": [{"id": 3, "range": 0}],
              "factors": [
                {"tensor": 0, "indices": [0, 3]},
                {"tensor": 2, "indices": [3, 1]}
              ]
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


def test_score_step_replays_rejected_probe_without_mutating_input_state():
    state = reject_then_actionable_state()
    sample = sample_step(state, RejectThenPreferenceScorer(), np.random.default_rng(0))
    before_score = state.definition_mask()

    first = score_step(state, RejectThenPreferenceScorer(), sample)
    second = score_step(state, RejectThenPreferenceScorer(), sample)

    assert [attempt.accepted for attempt in sample.def_attempts] == [False, True]
    assert [attempt.def_index for attempt in sample.def_attempts] == [0, 1]
    assert first == pytest.approx(sample.log_prob)
    assert second == pytest.approx(sample.log_prob)
    assert state.definition_mask() == before_score


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


def test_score_step_rejects_stopped_sample_with_accepted_attempt():
    sample = sample_step(empty_state(), StopScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=True,
        log_prob=sample.log_prob,
        def_attempts=(Stage1Attempt(def_index=0, log_prob=0.0, accepted=True),),
        decision_tokens=sample.decision_tokens,
    )

    with pytest.raises(ValueError, match="stopped sample must not contain accepted"):
        score_step(actionable_state(), PreferenceScorer(), invalid)


def test_score_step_rejects_attempt_after_accepted_attempt():
    sample = sample_step(actionable_state(), PreferenceScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=False,
        def_index=sample.def_index,
        action_space=sample.action_space,
        decision=sample.decision,
        log_prob=sample.log_prob,
        def_attempts=(
            *sample.def_attempts,
            Stage1Attempt(def_index=0, log_prob=0.0, accepted=False),
        ),
        decision_tokens=sample.decision_tokens,
    )

    with pytest.raises(ValueError, match="accepted stage-1 attempt must be final"):
        score_step(actionable_state(), PreferenceScorer(), invalid)


def test_score_step_rejects_mismatched_sample_def_index():
    sample = sample_step(actionable_state(), PreferenceScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=False,
        def_index=999,
        action_space=sample.action_space,
        decision=sample.decision,
        log_prob=sample.log_prob,
        def_attempts=sample.def_attempts,
        decision_tokens=sample.decision_tokens,
    )

    with pytest.raises(ValueError, match="sample def_index must match accepted"):
        score_step(actionable_state(), PreferenceScorer(), invalid)


def test_score_step_rejects_mismatched_decision_tokens_for_stop():
    sample = sample_step(empty_state(), StopScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=True,
        log_prob=sample.log_prob,
        def_attempts=sample.def_attempts,
        decision_tokens=(T("DEF", def_index=0),),
    )

    with pytest.raises(ValueError, match="stopped sample decision_tokens must be STOP"):
        score_step(empty_state(), StopScorer(), invalid)


def test_score_step_rejects_mismatched_decision_tokens_for_rewrite():
    sample = sample_step(actionable_state(), PreferenceScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=False,
        def_index=sample.def_index,
        action_space=sample.action_space,
        decision=sample.decision,
        log_prob=sample.log_prob,
        def_attempts=sample.def_attempts,
        decision_tokens=(T("CAND", candidate_index=0), T("END")),
    )

    with pytest.raises(ValueError, match="decision_tokens must match replayed decision"):
        score_step(actionable_state(), PreferenceScorer(), invalid)


def test_sample_step_rejects_wrong_shape_logits():
    with pytest.raises(ValueError, match="scorer logits must be a 1-D vector"):
        sample_step(empty_state(), WrongShapeScorer(), np.random.default_rng(0))


def test_score_step_rejects_non_finite_logits():
    sample = sample_step(empty_state(), StopScorer(), np.random.default_rng(0))

    with pytest.raises(ValueError, match="scorer logits must be finite"):
        score_step(empty_state(), NonFiniteScorer(), sample)


def test_score_step_rejects_bool_candidate_index():
    sample = sample_step(actionable_state(), PreferenceScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=False,
        def_index=sample.def_index,
        action_space=sample.action_space,
        decision={
            "candidate_index": True,
            "left_mask": sample.decision["left_mask"],
            "right_mask": sample.decision["right_mask"],
        },
        log_prob=sample.log_prob,
        def_attempts=sample.def_attempts,
        decision_tokens=sample.decision_tokens,
    )

    with pytest.raises(ValueError, match="candidate_index must be an int"):
        score_step(actionable_state(), PreferenceScorer(), invalid)


def test_score_step_rejects_integer_mask_entry():
    sample = sample_step(actionable_state(), PreferenceScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=False,
        def_index=sample.def_index,
        action_space=sample.action_space,
        decision={
            "candidate_index": sample.decision["candidate_index"],
            "left_mask": [1],
            "right_mask": sample.decision["right_mask"],
        },
        log_prob=sample.log_prob,
        def_attempts=sample.def_attempts,
        decision_tokens=sample.decision_tokens,
    )

    with pytest.raises(ValueError, match="left_mask entries must be bool"):
        score_step(actionable_state(), PreferenceScorer(), invalid)


def test_score_step_rejects_invalid_candidate_index():
    sample = sample_step(actionable_state(), PreferenceScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=False,
        def_index=sample.def_index,
        action_space=sample.action_space,
        decision={
            "candidate_index": 999,
            "left_mask": sample.decision["left_mask"],
            "right_mask": sample.decision["right_mask"],
        },
        log_prob=sample.log_prob,
        def_attempts=sample.def_attempts,
        decision_tokens=sample.decision_tokens,
    )

    with pytest.raises(ValueError, match="invalid candidate_index"):
        score_step(actionable_state(), PreferenceScorer(), invalid)


def test_score_step_rejects_invalid_right_mask_length():
    sample = sample_step(actionable_state(), PreferenceScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=False,
        def_index=sample.def_index,
        action_space=sample.action_space,
        decision={
            "candidate_index": sample.decision["candidate_index"],
            "left_mask": sample.decision["left_mask"],
            "right_mask": [True],
        },
        log_prob=sample.log_prob,
        def_attempts=sample.def_attempts,
        decision_tokens=sample.decision_tokens,
    )

    with pytest.raises(ValueError, match="invalid right_mask length"):
        score_step(actionable_state(), PreferenceScorer(), invalid)
