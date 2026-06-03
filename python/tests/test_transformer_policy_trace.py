import numpy as np
import pytest

from transformer_policy.decoder import sample_step, sample_step_with_events
from transformer_policy.trace import (
    TokenChoiceEvent,
    TracedPolicySample,
    event_from_wire,
    event_to_wire,
    token_from_wire,
    token_to_wire,
)
from transformer_policy.types import T

from .test_transformer_policy_decoder import (
    PreferenceScorer,
    RejectThenPreferenceScorer,
    StopScorer,
    empty_state,
    reject_then_actionable_state,
)
from .transformer_policy_fixtures import actionable_state


def test_token_wire_round_trip_preserves_payload_order():
    token = T("FACTOR", position=1, tensor=3, arity=2)

    wire = token_to_wire(token)
    restored = token_from_wire(wire)

    assert wire == {
        "kind": "FACTOR",
        "payload": (("arity", 2), ("position", 1), ("tensor", 3)),
    }
    assert restored == token


def test_token_wire_rejects_invalid_shape():
    with pytest.raises(ValueError, match="token wire kind must be a string"):
        token_from_wire({"kind": 1, "payload": ()})

    with pytest.raises(ValueError, match="token wire payload must be a list or tuple"):
        token_from_wire({"kind": "STOP", "payload": None})


def test_token_wire_accepts_transport_style_lists():
    restored = token_from_wire(
        {
            "kind": "FACTOR",
            "payload": [["arity", 2], ["position", 1], ["tensor", 3]],
        }
    )

    assert restored == T("FACTOR", position=1, tensor=3, arity=2)


def test_event_wire_round_trip():
    event = TokenChoiceEvent(
        sequence_tokens=(T("STATE_START"), T("STATE_END")),
        legal_next_tokens=(T("STOP"), T("DEF", def_index=0)),
        chosen_index=1,
        phase="def",
        step_index=2,
    )

    restored = event_from_wire(event_to_wire(event))

    assert restored == event


def test_event_wire_accepts_transport_style_lists():
    restored = event_from_wire(
        {
            "sequence_tokens": [
                {"kind": "STATE_START", "payload": []},
                {"kind": "STATE_END", "payload": []},
            ],
            "legal_next_tokens": [
                {"kind": "STOP", "payload": []},
                {"kind": "DEF", "payload": [["def_index", 0]]},
            ],
            "chosen_index": 1,
            "phase": "def",
            "step_index": 3,
        }
    )

    assert restored == TokenChoiceEvent(
        sequence_tokens=(T("STATE_START"), T("STATE_END")),
        legal_next_tokens=(T("STOP"), T("DEF", def_index=0)),
        chosen_index=1,
        phase="def",
        step_index=3,
    )


def test_event_wire_rejects_missing_or_invalid_collections():
    with pytest.raises(ValueError, match="event wire sequence_tokens must be present"):
        event_from_wire(
            {
                "legal_next_tokens": [],
                "chosen_index": 0,
                "phase": "def",
                "step_index": 0,
            }
        )

    with pytest.raises(ValueError, match="event wire legal_next_tokens must be a list or tuple"):
        event_from_wire(
            {
                "sequence_tokens": [],
                "legal_next_tokens": None,
                "chosen_index": 0,
                "phase": "def",
                "step_index": 0,
            }
        )


def test_traced_sampling_matches_plain_sampling_for_same_seed():
    state = actionable_state()
    traced = sample_step_with_events(
        state,
        PreferenceScorer(),
        np.random.default_rng(0),
    )
    plain = sample_step(
        actionable_state(),
        PreferenceScorer(),
        np.random.default_rng(0),
    )

    assert isinstance(traced, TracedPolicySample)
    assert traced.sample.stopped == plain.stopped
    assert traced.sample.def_index == plain.def_index
    assert traced.sample.decision == plain.decision
    assert traced.sample.def_attempts == plain.def_attempts
    assert traced.sample.decision_tokens == plain.decision_tokens
    assert traced.sample.log_prob == pytest.approx(plain.log_prob)


def test_traced_sampling_records_one_event_per_sampled_decision():
    traced = sample_step_with_events(
        actionable_state(),
        PreferenceScorer(),
        np.random.default_rng(0),
    )

    phases = [event.phase for event in traced.events]

    assert phases == ["def", "candidate", "left_bit", "right_bit", "right_bit"]
    assert [event.step_index for event in traced.events] == [0, 0, 0, 0, 0]
    assert traced.events[0].legal_next_tokens[0] == T("STOP")
    assert traced.events[0].legal_next_tokens[1] == T("DEF", def_index=0)
    assert traced.events[0].chosen_index == 1
    assert traced.events[1].legal_next_tokens[traced.events[1].chosen_index] == T(
        "CAND",
        candidate_index=0,
    )


def test_traced_sampling_records_stage1_retry_attempts():
    traced = sample_step_with_events(
        reject_then_actionable_state(),
        RejectThenPreferenceScorer(),
        np.random.default_rng(0),
    )

    assert [event.phase for event in traced.events[:2]] == ["def", "def"]
    assert [event.chosen_index for event in traced.events[:2]] == [1, 1]
    assert [attempt.accepted for attempt in traced.sample.def_attempts] == [False, True]
    assert [attempt.def_index for attempt in traced.sample.def_attempts] == [0, 1]
    assert traced.events[0].legal_next_tokens[1] == T("DEF", def_index=0)
    assert traced.events[1].legal_next_tokens[1] == T("DEF", def_index=1)


def test_stop_only_state_records_zero_log_prob_stop_event():
    traced = sample_step_with_events(empty_state(), StopScorer(), np.random.default_rng(0))

    assert traced.sample.stopped
    assert traced.sample.log_prob == pytest.approx(0.0)
    assert len(traced.events) == 1
    event = traced.events[0]
    assert event.phase == "def"
    assert event.legal_next_tokens == (T("STOP"),)
    assert event.chosen_index == 0
