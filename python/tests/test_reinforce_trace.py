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


def test_stage1_attempt_trace_from_policy_attempt():
    trace = Stage1AttemptTrace.from_policy_attempt(
        Stage1Attempt(def_index=2, log_prob=-0.5, accepted=False)
    )

    assert trace.def_index == 2
    assert trace.log_prob == -0.5
    assert not trace.accepted


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


def test_episode_trace_validates_reward_and_steps():
    step = StepTrace(
        step_index=0,
        state_snapshot={"definitions": []},
        stopped=True,
        def_attempts=(),
        def_index=None,
        action_space_snapshot=None,
        decision=None,
        decision_tokens=(T("STOP"),),
        token_events=(
            TokenChoiceEvent(
                sequence_tokens=(T("STATE_START"), T("STATE_END")),
                legal_next_tokens=(T("STOP"),),
                chosen_index=0,
                phase="def",
                step_index=0,
            ),
        ),
        sample_log_prob=0.0,
    )

    episode = EpisodeTrace(
        episode_index=0,
        episode_seed=123,
        steps=(step,),
        final_snapshot={"definitions": []},
        final_log_flops=7.0,
        reward=-7.0,
        terminal_reason="stop",
    )

    assert episode.reward == -episode.final_log_flops


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
