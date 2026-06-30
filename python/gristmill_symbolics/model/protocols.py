from __future__ import annotations

from typing import Protocol, TypeVar

BatchedStateType = TypeVar("BatchedStateType")
BatchedTransitionType = TypeVar("BatchedTransitionType")


class StepwiseModel(Protocol[BatchedStateType, BatchedTransitionType]):
    def init_params(self, rng) -> object:
        ...

    def sample_step(
        self,
        params,
        rng,
        states: BatchedStateType,
    ) -> tuple[BatchedStateType, object, object]:
        ...

    def score_step(
        self,
        params,
        transitions: BatchedTransitionType,
    ) -> tuple[object, object]:
        ...
