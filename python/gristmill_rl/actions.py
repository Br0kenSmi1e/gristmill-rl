from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class SampledAction:
    decision: dict[str, Any]
    prior: float


def decision_key(decision: dict[str, Any]) -> tuple[int, tuple[bool, ...], tuple[bool, ...]]:
    return (
        int(decision["candidate_index"]),
        tuple(bool(value) for value in decision["left_mask"]),
        tuple(bool(value) for value in decision["right_mask"]),
    )


def _nonempty_mask(length: int, rng: np.random.Generator) -> list[bool]:
    if length <= 0:
        return []
    mask = [bool(value) for value in rng.integers(0, 2, size=length)]
    if not any(mask):
        mask[int(rng.integers(0, length))] = True
    return mask


def first_full_mask_action(
    action_space_snapshot: dict[str, Any], candidate_index: int = 0, prior: float = 1.0
) -> SampledAction:
    candidate = action_space_snapshot["candidate_templates"][candidate_index]
    return SampledAction(
        decision={
            "candidate_index": candidate_index,
            "left_mask": [True] * len(candidate["left_definition"]["terms"]),
            "right_mask": [True] * len(candidate["right_definition"]["terms"]),
        },
        prior=prior,
    )


def uniform_random_action(
    action_space_snapshot: dict[str, Any],
    rng: np.random.Generator,
    prior: float,
) -> SampledAction:
    candidates = action_space_snapshot["candidate_templates"]
    candidate_index = int(rng.integers(0, len(candidates)))
    candidate = candidates[candidate_index]
    return SampledAction(
        decision={
            "candidate_index": candidate_index,
            "left_mask": _nonempty_mask(len(candidate["left_definition"]["terms"]), rng),
            "right_mask": _nonempty_mask(len(candidate["right_definition"]["terms"]), rng),
        },
        prior=prior,
    )


def is_valid_action(comp: Any, space: Any, action: SampledAction) -> bool:
    child = comp.clone()
    try:
        child.apply_decision_with_space(space, action.decision)
    except Exception:
        return False
    return True


def sample_valid_actions(
    comp: Any,
    space: Any,
    proposal_fn: Callable[[], SampledAction],
    actions_per_node: int,
    sample_attempts: int,
) -> list[SampledAction]:
    if actions_per_node <= 0:
        raise ValueError("actions_per_node must be positive")
    if sample_attempts <= 0:
        raise ValueError("sample_attempts must be positive")

    unique: dict[tuple[int, tuple[bool, ...], tuple[bool, ...]], SampledAction] = {}
    for _ in range(sample_attempts):
        action = proposal_fn()
        key = decision_key(action.decision)
        if key in unique:
            continue
        if is_valid_action(comp, space, action):
            unique[key] = action
        if len(unique) >= actions_per_node:
            break
    return list(unique.values())
