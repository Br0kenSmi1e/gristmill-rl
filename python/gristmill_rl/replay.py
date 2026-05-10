from __future__ import annotations

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
        total = float(np.sum(record.visit_distribution))
        if total <= 0.0:
            raise ValueError("visit_distribution must have positive mass")
        self.records.append(record)

    def complete(self, *, final_log_flops: float) -> list[ReplayItem]:
        completed = []
        for record in self.records:
            policy_target = np.asarray(record.visit_distribution, dtype=np.float32)
            policy_target = policy_target / np.sum(policy_target)
            completed.append(
                ReplayItem(
                    state_snapshot=record.state_snapshot,
                    action_space_snapshot=record.action_space_snapshot,
                    sampled_actions=record.sampled_actions,
                    policy_target=policy_target,
                    value_target=record.state_log_flops - final_log_flops,
                    state_log_flops=record.state_log_flops,
                    start_from=record.start_from,
                )
            )
        return completed


class ReplayBuffer:
    def __init__(self, *, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: list[ReplayItem] = []
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def extend(self, items: list[ReplayItem]) -> None:
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
