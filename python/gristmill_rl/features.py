from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Any

import numpy as np


STATE_DIM = 8
CANDIDATE_DIM = 6
TERM_DIM = 4


@dataclass(frozen=True)
class FeatureConfig:
    max_candidates: int = 16
    max_left_terms: int = 8
    max_right_terms: int = 8


@dataclass(frozen=True)
class TruncationCounts:
    candidates: int = 0
    left_terms: int = 0
    right_terms: int = 0


@dataclass(frozen=True)
class FeatureBatch:
    state: np.ndarray
    candidates: np.ndarray
    left_terms: np.ndarray
    right_terms: np.ndarray
    candidate_mask: np.ndarray
    left_term_mask: np.ndarray
    right_term_mask: np.ndarray
    truncation: TruncationCounts


def _coeff_pair(coeff: Any) -> tuple[float, float]:
    if isinstance(coeff, dict):
        return float(coeff["numer"]), float(coeff["denom"])
    if isinstance(coeff, list | tuple) and len(coeff) == 2:
        return float(coeff[0]), float(coeff[1])
    raise TypeError(f"unsupported coefficient shape: {coeff!r}")


def _term_features(term: dict[str, Any]) -> np.ndarray:
    numer, denom = _coeff_pair(term["coeff"])
    coeff = numer / denom
    sign = 1.0 if coeff > 0.0 else -1.0 if coeff < 0.0 else 0.0
    return np.asarray(
        [
            float(len(term["factors"])),
            float(len(term["sum_indices"])),
            sign,
            log1p(abs(coeff)),
        ],
        dtype=np.float32,
    )


def _definition_summary(definition: dict[str, Any]) -> tuple[float, float, float]:
    terms = definition["terms"]
    if not terms:
        return 0.0, 0.0, 0.0
    factor_counts = [len(term["factors"]) for term in terms]
    sum_counts = [len(term["sum_indices"]) for term in terms]
    return (
        float(len(terms)),
        float(np.mean(factor_counts)),
        float(np.mean(sum_counts)),
    )


def _candidate_features(candidate: dict[str, Any]) -> np.ndarray:
    left_count, left_factors, _ = _definition_summary(candidate["left_definition"])
    right_count, right_factors, _ = _definition_summary(candidate["right_definition"])
    rewritten_count, rewritten_factors, _ = _definition_summary(
        candidate["rewritten_definition"]
    )
    return np.asarray(
        [
            left_count,
            right_count,
            rewritten_count,
            left_factors,
            right_factors,
            rewritten_factors,
        ],
        dtype=np.float32,
    )


def _fill_terms(
    target: np.ndarray,
    mask: np.ndarray,
    candidate_index: int,
    terms: list[dict[str, Any]],
    max_terms: int,
) -> int:
    kept = min(len(terms), max_terms)
    for term_index, term in enumerate(terms[:kept]):
        target[candidate_index, term_index] = _term_features(term)
        mask[candidate_index, term_index] = True
    return max(0, len(terms) - max_terms)


def extract_features(
    *,
    comp_snapshot: dict[str, Any],
    action_space_snapshot: dict[str, Any],
    start_from: int,
    log_total_flops: float,
    config: FeatureConfig,
) -> FeatureBatch:
    definitions = comp_snapshot["definitions"]
    active = definitions[action_space_snapshot["def_index"]]
    active_terms, active_factors, active_sums = _definition_summary(active)

    state = np.asarray(
        [
            float(len(definitions)),
            float(len(comp_snapshot["tensors"])),
            float(len(comp_snapshot["ranges"])),
            float(start_from),
            float(log_total_flops),
            active_terms,
            active_factors,
            active_sums,
        ],
        dtype=np.float32,
    )

    candidates = np.zeros((config.max_candidates, CANDIDATE_DIM), dtype=np.float32)
    left_terms = np.zeros(
        (config.max_candidates, config.max_left_terms, TERM_DIM), dtype=np.float32
    )
    right_terms = np.zeros(
        (config.max_candidates, config.max_right_terms, TERM_DIM), dtype=np.float32
    )
    candidate_mask = np.zeros((config.max_candidates,), dtype=np.bool_)
    left_term_mask = np.zeros(
        (config.max_candidates, config.max_left_terms), dtype=np.bool_
    )
    right_term_mask = np.zeros(
        (config.max_candidates, config.max_right_terms), dtype=np.bool_
    )

    raw_candidates = action_space_snapshot["candidate_templates"]
    kept_candidates = min(len(raw_candidates), config.max_candidates)
    left_truncation = 0
    right_truncation = 0

    for candidate_index, candidate in enumerate(raw_candidates[:kept_candidates]):
        candidate_mask[candidate_index] = True
        candidates[candidate_index] = _candidate_features(candidate)
        left_truncation += _fill_terms(
            left_terms,
            left_term_mask,
            candidate_index,
            candidate["left_definition"]["terms"],
            config.max_left_terms,
        )
        right_truncation += _fill_terms(
            right_terms,
            right_term_mask,
            candidate_index,
            candidate["right_definition"]["terms"],
            config.max_right_terms,
        )

    return FeatureBatch(
        state=state,
        candidates=candidates,
        left_terms=left_terms,
        right_terms=right_terms,
        candidate_mask=candidate_mask,
        left_term_mask=left_term_mask,
        right_term_mask=right_term_mask,
        truncation=TruncationCounts(
            candidates=max(0, len(raw_candidates) - config.max_candidates),
            left_terms=left_truncation,
            right_terms=right_truncation,
        ),
    )
