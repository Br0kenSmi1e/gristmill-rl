from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gristmill_rl.actions import SampledAction


@dataclass(frozen=True)
class RootTraceRecord:
    state_snapshot: dict[str, Any]
    action_space_snapshot: dict[str, Any]
    sampled_actions: list[SampledAction]
    visit_distribution: np.ndarray
    state_log_flops: float
    start_from: int


@dataclass(frozen=True)
class ReplayItem:
    state_snapshot: dict[str, Any]
    action_space_snapshot: dict[str, Any]
    sampled_actions: list[SampledAction]
    policy_target: np.ndarray
    value_target: float
    state_log_flops: float
    start_from: int


@dataclass
class EpisodeTrace:
    records: list[RootTraceRecord] = field(default_factory=list)

    def append(self, record: RootTraceRecord) -> None:
        _validated_visit_distribution(record)
        self.records.append(record)

    def complete(self, *, final_log_flops: float) -> list[ReplayItem]:
        completed = []
        for record in self.records:
            visit_distribution = _validated_visit_distribution(record)
            total = float(np.sum(visit_distribution))
            policy_target = visit_distribution.astype(np.float32, copy=True) / total
            completed.append(
                ReplayItem(
                    state_snapshot=copy.deepcopy(record.state_snapshot),
                    action_space_snapshot=copy.deepcopy(record.action_space_snapshot),
                    sampled_actions=copy.deepcopy(record.sampled_actions),
                    policy_target=policy_target,
                    value_target=record.state_log_flops - final_log_flops,
                    state_log_flops=record.state_log_flops,
                    start_from=record.start_from,
                )
            )
        return completed


def _validated_visit_distribution(record: RootTraceRecord) -> np.ndarray:
    visit_distribution = np.asarray(record.visit_distribution, dtype=np.float64)
    if len(visit_distribution) == 0:
        raise ValueError("visit_distribution must not be empty")
    if len(visit_distribution) != len(record.sampled_actions):
        raise ValueError("visit_distribution length must match sampled_actions")
    if not np.all(np.isfinite(visit_distribution)):
        raise ValueError("visit_distribution must contain only finite values")
    if np.any(visit_distribution < 0.0):
        raise ValueError("visit_distribution must not contain negative values")
    total = float(np.sum(visit_distribution))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("visit_distribution must have positive finite mass")
    return visit_distribution


class ReplayBuffer:
    def __init__(self, *, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: list[ReplayItem] = []
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def extend(self, items: Iterable[ReplayItem]) -> None:
        self._items.extend(items)
        overflow = len(self._items) - self.capacity
        if overflow > 0:
            del self._items[:overflow]

    def sample(self, *, batch_size: int) -> list[ReplayItem]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")
        count = min(batch_size, len(self._items))
        indices = self._rng.choice(len(self._items), size=count, replace=False)
        return [self._items[int(index)] for index in indices]
