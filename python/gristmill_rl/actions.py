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


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + float(np.exp(-value)))
    exp_value = float(np.exp(value))
    return exp_value / (1.0 + exp_value)


def _sample_mask_from_logits(
    logits: np.ndarray,
    valid_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[bool], float]:
    choices = []
    prior = 1.0
    valid_indices = np.flatnonzero(valid_mask)
    for index, logit in enumerate(logits[: valid_mask.shape[0]]):
        if not bool(valid_mask[index]):
            choices.append(False)
            continue
        probability = _sigmoid(float(logit))
        keep = bool(rng.random() < probability)
        choices.append(keep)
        prior *= probability if keep else (1.0 - probability)
    if valid_indices.size and not any(choices):
        forced = int(rng.choice(valid_indices))
        choices[forced] = True
        prior = max(prior, 1.0e-8)
    return choices, max(prior, 1.0e-8)


def _fit_mask_to_term_count(mask: list[bool], term_count: int) -> list[bool]:
    if len(mask) >= term_count:
        return mask[:term_count]
    return [*mask, *([False] * (term_count - len(mask)))]


def make_model_proposal_fn(
    *,
    model: Any,
    features: Any,
    action_space_snapshot: dict[str, Any],
    rng: np.random.Generator,
) -> Callable[[], SampledAction]:
    candidate_templates = action_space_snapshot["candidate_templates"]
    if not candidate_templates:
        raise ValueError("action space has no candidate templates")

    def propose() -> SampledAction:
        outputs = model(features)
        candidate_logits = np.asarray(outputs.candidate_logits, dtype=np.float64)
        candidate_mask = np.asarray(features.candidate_mask, dtype=bool).copy()
        represented_count = min(candidate_logits.shape[0], len(candidate_templates))
        candidate_mask[represented_count:] = False
        if not candidate_mask[:represented_count].any():
            raise ValueError("no valid represented action candidates")

        safe_logits = np.where(np.isfinite(candidate_logits), candidate_logits, -1.0e9)
        masked = np.where(candidate_mask, safe_logits, -1.0e9)
        shifted = masked - np.max(masked[candidate_mask])
        probabilities = np.where(candidate_mask, np.exp(shifted), 0.0)
        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0.0:
            probabilities = candidate_mask.astype(np.float64)
            total = probabilities.sum()
        probabilities = probabilities / total

        candidate_index = int(rng.choice(len(probabilities), p=probabilities))
        candidate = candidate_templates[candidate_index]
        left_mask, left_prior = _sample_mask_from_logits(
            np.asarray(outputs.left_logits[candidate_index], dtype=np.float64),
            np.asarray(features.left_term_mask[candidate_index], dtype=bool),
            rng,
        )
        right_mask, right_prior = _sample_mask_from_logits(
            np.asarray(outputs.right_logits[candidate_index], dtype=np.float64),
            np.asarray(features.right_term_mask[candidate_index], dtype=bool),
            rng,
        )
        return SampledAction(
            decision={
                "candidate_index": candidate_index,
                "left_mask": _fit_mask_to_term_count(
                    left_mask, len(candidate["left_definition"]["terms"])
                ),
                "right_mask": _fit_mask_to_term_count(
                    right_mask, len(candidate["right_definition"]["terms"])
                ),
            },
            prior=float(probabilities[candidate_index] * left_prior * right_prior),
        )

    return propose
