import pytest

from transformer_policy.trace import TokenChoiceEvent
from transformer_policy.types import Stage1Attempt, T
from reinforce_training.trace import (
    EpisodeTrace,
    Stage1AttemptTrace,
    StepTrace,
    step_trace_from_traced_sample,
)
from transformer_policy.decoder import sample_step_with_events

from .test_transformer_policy_decoder import PreferenceScorer
from .transformer_policy_fixtures import actionable_state


def _token_event(*, step_index: int = 0) -> TokenChoiceEvent:
    return TokenChoiceEvent(
        sequence_tokens=(T("STATE_START"), T("STATE_END")),
        legal_next_tokens=(T("STOP"),),
        chosen_index=0,
        phase="def",
        step_index=step_index,
    )


def _stopped_step(*, step_index: int = 0) -> StepTrace:
    return StepTrace(
        step_index=step_index,
        state_snapshot={"definitions": []},
        stopped=True,
        def_attempts=(),
        def_index=None,
        action_space_snapshot=None,
        decision=None,
        decision_tokens=(T("STOP"),),
        token_events=(_token_event(step_index=step_index),),
        sample_log_prob=0.0,
    )


def _rewrite_step(*, step_index: int = 0) -> StepTrace:
    return StepTrace(
        step_index=step_index,
        state_snapshot={"definitions": [{"terms": [0]}]},
        stopped=False,
        def_attempts=(),
        def_index=0,
        action_space_snapshot={"candidates": [{"left_terms": [0]}]},
        decision={
            "candidate_index": 0,
            "left_mask": [True],
            "right_mask": [True],
        },
        decision_tokens=(T("DEF", def_index=0),),
        token_events=(_token_event(step_index=step_index),),
        sample_log_prob=0.0,
    )


def test_stage1_attempt_trace_from_policy_attempt():
    trace = Stage1AttemptTrace.from_policy_attempt(
        Stage1Attempt(def_index=2, log_prob=-0.5, accepted=False)
    )

    assert trace.def_index == 2
    assert trace.log_prob == -0.5
    assert not trace.accepted


def test_stage1_attempt_trace_rejects_invalid_values():
    with pytest.raises(ValueError, match="def_index must be non-negative"):
        Stage1AttemptTrace(def_index=-1, log_prob=0.0, accepted=False)

    with pytest.raises(ValueError, match="log_prob must be finite"):
        Stage1AttemptTrace(def_index=0, log_prob=float("nan"), accepted=False)


def test_step_trace_from_traced_sample_drops_action_space_handle():
    traced = sample_step_with_events(
        actionable_state(),
        PreferenceScorer(),
        __import__("numpy").random.default_rng(0),
    )
    step = step_trace_from_traced_sample(
        step_index=0,
        state_snapshot=actionable_state().snapshot(),
        traced=traced,
    )

    assert isinstance(step, StepTrace)
    assert not step.stopped
    assert step.action_space_snapshot is not None
    assert step.decision is not None
    assert step.sample_log_prob == pytest.approx(traced.sample.log_prob)
    assert step.token_events == traced.events
    assert not hasattr(step, "action_space")


def test_step_trace_from_traced_sample_copies_mutable_inputs():
    state_snapshot = actionable_state().snapshot()
    traced = sample_step_with_events(
        actionable_state(),
        PreferenceScorer(),
        __import__("numpy").random.default_rng(0),
    )

    step = step_trace_from_traced_sample(
        step_index=0,
        state_snapshot=state_snapshot,
        traced=traced,
    )
    state_snapshot["definitions"].append({"base": 999})
    traced.sample.decision["left_mask"][0] = False

    assert len(step.state_snapshot["definitions"]) == 1
    assert step.decision["left_mask"] == [True]


def test_step_trace_copies_mutable_snapshots_on_construction():
    state_snapshot = {"definitions": [{"terms": [0]}]}
    action_space_snapshot = {"candidates": [{"left_terms": [0]}]}
    decision = {"candidate_index": 0, "left_mask": [True], "right_mask": [True]}

    step = StepTrace(
        step_index=0,
        state_snapshot=state_snapshot,
        stopped=False,
        def_attempts=(),
        def_index=0,
        action_space_snapshot=action_space_snapshot,
        decision=decision,
        decision_tokens=(T("DEF", def_index=0),),
        token_events=(_token_event(),),
        sample_log_prob=0.0,
    )
    state_snapshot["definitions"][0]["terms"].append(1)
    action_space_snapshot["candidates"][0]["left_terms"].append(1)
    decision["left_mask"][0] = False

    assert step.state_snapshot["definitions"][0]["terms"] == [0]
    assert step.action_space_snapshot["candidates"][0]["left_terms"] == [0]
    assert step.decision["left_mask"] == [True]


def test_step_trace_rejects_token_event_step_index_mismatch():
    with pytest.raises(ValueError, match="token event step_index must match"):
        StepTrace(
            step_index=1,
            state_snapshot={"definitions": []},
            stopped=True,
            def_attempts=(),
            def_index=None,
            action_space_snapshot=None,
            decision=None,
            decision_tokens=(T("STOP"),),
            token_events=(_token_event(step_index=0),),
            sample_log_prob=0.0,
        )


def test_episode_trace_validates_reward_and_steps():
    episode = EpisodeTrace(
        episode_index=0,
        episode_seed=123,
        steps=(_stopped_step(),),
        final_snapshot={"definitions": []},
        final_log_flops=7.0,
        reward=-7.0,
        terminal_reason="stop",
    )

    assert episode.reward == -episode.final_log_flops


def test_episode_trace_copies_final_snapshot_on_construction():
    final_snapshot = {"definitions": [{"terms": [0]}]}

    episode = EpisodeTrace(
        episode_index=0,
        episode_seed=123,
        steps=(_stopped_step(),),
        final_snapshot=final_snapshot,
        final_log_flops=7.0,
        reward=-7.0,
        terminal_reason="stop",
    )
    final_snapshot["definitions"][0]["terms"].append(1)

    assert episode.final_snapshot["definitions"][0]["terms"] == [0]


def test_episode_trace_rejects_invalid_terminal_reason():
    with pytest.raises(ValueError, match="terminal_reason must be"):
        EpisodeTrace(
            episode_index=0,
            episode_seed=0,
            steps=(),
            final_snapshot={},
            final_log_flops=1.0,
            reward=-1.0,
            terminal_reason="done",
        )


def test_episode_trace_rejects_empty_steps():
    with pytest.raises(ValueError, match="episode trace must contain at least one step"):
        EpisodeTrace(
            episode_index=0,
            episode_seed=0,
            steps=(),
            final_snapshot={},
            final_log_flops=1.0,
            reward=-1.0,
            terminal_reason="max_steps",
        )


def test_episode_trace_rejects_terminal_reason_inconsistency():
    with pytest.raises(ValueError, match="stop episode requires a final stopped step"):
        EpisodeTrace(
            episode_index=0,
            episode_seed=0,
            steps=(_rewrite_step(),),
            final_snapshot={},
            final_log_flops=1.0,
            reward=-1.0,
            terminal_reason="stop",
        )

    with pytest.raises(ValueError, match="max_steps episode must not end with stop"):
        EpisodeTrace(
            episode_index=0,
            episode_seed=0,
            steps=(_stopped_step(),),
            final_snapshot={},
            final_log_flops=1.0,
            reward=-1.0,
            terminal_reason="max_steps",
        )
