from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from transformer_policy.trace import TokenChoiceEvent
from transformer_policy.types import Stage1Attempt, Token


TerminalReason = Literal["stop", "max_steps"]


@dataclass(frozen=True)
class Stage1AttemptTrace:
    def_index: int
    log_prob: float
    accepted: bool

    def __post_init__(self) -> None:
        if self.def_index < 0:
            raise ValueError("def_index must be non-negative")
        log_prob = float(self.log_prob)
        if not np.isfinite(log_prob):
            raise ValueError("log_prob must be finite")
        object.__setattr__(self, "log_prob", log_prob)

    @staticmethod
    def from_policy_attempt(attempt: Stage1Attempt) -> "Stage1AttemptTrace":
        return Stage1AttemptTrace(
            def_index=attempt.def_index,
            log_prob=float(attempt.log_prob),
            accepted=attempt.accepted,
        )


@dataclass(frozen=True)
class StepTrace:
    step_index: int
    state_snapshot: dict[str, Any]
    stopped: bool
    def_attempts: tuple[Stage1AttemptTrace, ...]
    def_index: int | None
    action_space_snapshot: dict[str, Any] | None
    decision: dict[str, Any] | None
    decision_tokens: tuple[Token, ...]
    token_events: tuple[TokenChoiceEvent, ...]
    sample_log_prob: float

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        token_events = tuple(self.token_events)
        if not token_events:
            raise ValueError("step trace must contain token events")
        for event in token_events:
            if event.step_index != self.step_index:
                raise ValueError("token event step_index must match step trace")
        sample_log_prob = float(self.sample_log_prob)
        if not np.isfinite(sample_log_prob):
            raise ValueError("sample_log_prob must be finite")
        if self.stopped:
            if self.def_index is not None or self.action_space_snapshot is not None:
                raise ValueError("stopped step must not contain rewrite data")
            if self.decision is not None:
                raise ValueError("stopped step must not contain decision")
        else:
            if self.def_index is None:
                raise ValueError("rewrite step requires def_index")
            if self.action_space_snapshot is None:
                raise ValueError("rewrite step requires action_space_snapshot")
            if self.decision is None:
                raise ValueError("rewrite step requires decision")
        object.__setattr__(self, "state_snapshot", deepcopy(self.state_snapshot))
        object.__setattr__(self, "def_attempts", tuple(self.def_attempts))
        object.__setattr__(
            self,
            "action_space_snapshot",
            deepcopy(self.action_space_snapshot),
        )
        object.__setattr__(self, "decision", deepcopy(self.decision))
        object.__setattr__(self, "decision_tokens", tuple(self.decision_tokens))
        object.__setattr__(self, "token_events", token_events)
        object.__setattr__(self, "sample_log_prob", sample_log_prob)


@dataclass(frozen=True)
class EpisodeTrace:
    episode_index: int
    episode_seed: int
    steps: tuple[StepTrace, ...]
    final_snapshot: dict[str, Any]
    final_log_flops: float
    reward: float
    terminal_reason: TerminalReason

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self.terminal_reason not in {"stop", "max_steps"}:
            raise ValueError("terminal_reason must be 'stop' or 'max_steps'")
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("episode trace must contain at least one step")
        final_log_flops = float(self.final_log_flops)
        if not np.isfinite(final_log_flops):
            raise ValueError("final_log_flops must be finite")
        reward = float(self.reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if self.terminal_reason == "stop" and not steps[-1].stopped:
            raise ValueError("stop episode requires a final stopped step")
        if self.terminal_reason == "max_steps" and steps[-1].stopped:
            raise ValueError("max_steps episode must not end with stop")
        object.__setattr__(self, "steps", deepcopy(steps))
        object.__setattr__(self, "final_snapshot", deepcopy(self.final_snapshot))
        object.__setattr__(self, "final_log_flops", final_log_flops)
        object.__setattr__(self, "reward", reward)


def step_trace_from_traced_sample(
    *,
    step_index: int,
    state_snapshot: dict[str, Any],
    traced,
) -> StepTrace:
    sample = traced.sample
    action_space_snapshot = None
    if sample.action_space is not None:
        action_space_snapshot = sample.action_space.snapshot()
    return StepTrace(
        step_index=step_index,
        state_snapshot=state_snapshot,
        stopped=sample.stopped,
        def_attempts=tuple(
            Stage1AttemptTrace.from_policy_attempt(attempt)
            for attempt in sample.def_attempts
        ),
        def_index=sample.def_index,
        action_space_snapshot=action_space_snapshot,
        decision=sample.decision,
        decision_tokens=sample.decision_tokens,
        token_events=traced.events,
        sample_log_prob=float(sample.log_prob),
    )
