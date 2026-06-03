from __future__ import annotations

from typing import Protocol

import numpy as np

from transformer_policy.tokenize import (
    build_action_space_context,
    build_state_context,
)
from transformer_policy.types import PolicySample, Stage1Attempt, T, Token


class NextTokenScorer(Protocol):
    def score_next(
        self,
        context_tokens: tuple[Token, ...],
        decision_prefix: tuple[Token, ...],
        legal_next_tokens: tuple[Token, ...],
    ):
        raise NotImplementedError


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return shifted - np.log(exp.sum())


def _validated_logits(
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: tuple[Token, ...],
    legal: tuple[Token, ...],
) -> np.ndarray:
    logits = np.asarray(scorer.score_next(context, prefix, legal), dtype=np.float64)
    if logits.ndim != 1 or len(logits) != len(legal):
        raise ValueError("scorer logits must be a 1-D vector matching legal tokens")
    if not np.all(np.isfinite(logits)):
        raise ValueError("scorer logits must be finite")
    return logits


def _sample_token(
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: tuple[Token, ...],
    legal: tuple[Token, ...],
    rng: np.random.Generator,
) -> tuple[Token, float]:
    if not legal:
        raise ValueError("legal token set must not be empty")
    logits = _validated_logits(scorer, context, prefix, legal)
    log_probs = _log_softmax(logits)
    probs = np.exp(log_probs)
    index = int(rng.choice(len(legal), p=probs))
    return legal[index], float(log_probs[index])


def _score_token(
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: tuple[Token, ...],
    legal: tuple[Token, ...],
    chosen: Token,
) -> float:
    if chosen not in legal:
        raise ValueError(f"illegal token {chosen.kind}")
    logits = _validated_logits(scorer, context, prefix, legal)
    log_probs = _log_softmax(logits)
    return float(log_probs[legal.index(chosen)])


def _stage1_legal(state) -> tuple[Token, ...]:
    tokens = [T("STOP")]
    for def_index, allowed in enumerate(state.definition_mask()):
        if allowed:
            tokens.append(T("DEF", def_index=def_index))
    return tuple(tokens)


def _candidate_legal(space_snapshot: dict) -> tuple[Token, ...]:
    return tuple(
        T("CAND", candidate_index=index)
        for index, _candidate in enumerate(space_snapshot["candidate_templates"])
    )


def _bit_legal(kind_prefix: str, is_final: bool, kept_any: bool) -> tuple[Token, ...]:
    keep = T(f"{kind_prefix}_KEEP")
    drop = T(f"{kind_prefix}_DROP")
    if is_final and not kept_any:
        return (keep,)
    return (keep, drop)


def _sample_bits(
    *,
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: list[Token],
    kind_prefix: str,
    term_count: int,
    rng: np.random.Generator,
) -> tuple[list[bool], float]:
    bits: list[bool] = []
    log_prob = 0.0
    kept_any = False
    for term_index in range(term_count):
        legal = _bit_legal(kind_prefix, term_index == term_count - 1, kept_any)
        token, token_log_prob = _sample_token(scorer, context, tuple(prefix), legal, rng)
        prefix.append(token)
        keep = token.kind.endswith("KEEP")
        bits.append(keep)
        kept_any = kept_any or keep
        log_prob += token_log_prob
    return bits, log_prob


def _score_bits(
    *,
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: list[Token],
    kind_prefix: str,
    bits: list[bool],
) -> float:
    log_prob = 0.0
    kept_any = False
    for index, keep in enumerate(bits):
        legal = _bit_legal(kind_prefix, index == len(bits) - 1, kept_any)
        chosen = T(f"{kind_prefix}_{'KEEP' if keep else 'DROP'}")
        log_prob += _score_token(scorer, context, tuple(prefix), legal, chosen)
        prefix.append(chosen)
        kept_any = kept_any or keep
    return log_prob


def _accepted_attempt_indices(sample: PolicySample) -> list[int]:
    return [
        index
        for index, attempt in enumerate(sample.def_attempts)
        if attempt.accepted
    ]


def _validate_stage1_trace(sample: PolicySample) -> None:
    accepted_indices = _accepted_attempt_indices(sample)
    if sample.stopped:
        if accepted_indices:
            raise ValueError("stopped sample must not contain accepted stage-1 attempts")
        return
    if len(accepted_indices) != 1:
        raise ValueError("rewrite sample requires exactly one accepted stage-1 attempt")
    if accepted_indices[0] != len(sample.def_attempts) - 1:
        raise ValueError("accepted stage-1 attempt must be final")


def _decision_candidate_index(decision: dict) -> int:
    candidate_index = decision["candidate_index"]
    if type(candidate_index) is not int:
        raise ValueError("candidate_index must be an int")
    return candidate_index


def _validate_mask(decision: dict, key: str, expected_length: int) -> list[bool]:
    mask = decision[key]
    if not isinstance(mask, list):
        raise ValueError(f"{key} must be a list")
    if len(mask) != expected_length:
        raise ValueError(f"invalid {key} length")
    if any(type(value) is not bool for value in mask):
        raise ValueError(f"{key} entries must be bool")
    return mask


def sample_step(state, scorer: NextTokenScorer, rng: np.random.Generator) -> PolicySample:
    attempts: list[Stage1Attempt] = []
    total_log_prob = 0.0
    while True:
        state_context = build_state_context(state.snapshot())
        stage1_token, stage1_log_prob = _sample_token(
            scorer, state_context, (), _stage1_legal(state), rng
        )
        total_log_prob += stage1_log_prob
        if stage1_token.kind == "STOP":
            return PolicySample(
                stopped=True,
                log_prob=total_log_prob,
                def_attempts=tuple(attempts),
                decision_tokens=(T("STOP"),),
            )
        def_index = int(stage1_token.payload_dict()["def_index"])
        space = state.action_space_for_def(def_index)
        accepted = space is not None
        attempts.append(
            Stage1Attempt(
                def_index=def_index,
                log_prob=stage1_log_prob,
                accepted=accepted,
            )
        )
        if accepted:
            break

    assert space is not None
    space_snapshot = space.snapshot()
    context = (
        *build_state_context(state.snapshot()),
        *build_action_space_context(space_snapshot),
    )
    prefix: list[Token] = []
    candidate_token, candidate_log_prob = _sample_token(
        scorer, context, tuple(prefix), _candidate_legal(space_snapshot), rng
    )
    prefix.append(candidate_token)
    total_log_prob += candidate_log_prob
    candidate_index = int(candidate_token.payload_dict()["candidate_index"])
    candidate = space_snapshot["candidate_templates"][candidate_index]

    left_bits, left_log_prob = _sample_bits(
        scorer=scorer,
        context=context,
        prefix=prefix,
        kind_prefix="LEFT",
        term_count=len(candidate["left_definition"]["terms"]),
        rng=rng,
    )
    right_bits, right_log_prob = _sample_bits(
        scorer=scorer,
        context=context,
        prefix=prefix,
        kind_prefix="RIGHT",
        term_count=len(candidate["right_definition"]["terms"]),
        rng=rng,
    )
    # END is a deterministic trace marker after fixed mask lengths, not a sampled decision.
    prefix.append(T("END"))
    total_log_prob += left_log_prob + right_log_prob
    return PolicySample(
        stopped=False,
        def_index=def_index,
        action_space=space,
        decision={
            "candidate_index": candidate_index,
            "left_mask": left_bits,
            "right_mask": right_bits,
        },
        log_prob=total_log_prob,
        def_attempts=tuple(attempts),
        decision_tokens=tuple(prefix),
    )


def score_step(state, scorer: NextTokenScorer, sample: PolicySample) -> float:
    _validate_stage1_trace(sample)
    total_log_prob = 0.0
    accepted_space = None
    for attempt in sample.def_attempts:
        context = build_state_context(state.snapshot())
        chosen = T("DEF", def_index=attempt.def_index)
        total_log_prob += _score_token(scorer, context, (), _stage1_legal(state), chosen)
        space = state.action_space_for_def(attempt.def_index)
        if attempt.accepted:
            if space is None:
                raise ValueError("invalid def_index accepted by sample trace")
            accepted_space = space
        else:
            if space is not None:
                raise ValueError("sample trace rejects an available def_index")
    if sample.stopped:
        context = build_state_context(state.snapshot())
        total_log_prob += _score_token(
            scorer, context, (), _stage1_legal(state), T("STOP")
        )
        return total_log_prob
    if accepted_space is None:
        raise ValueError("rewrite sample requires an accepted def_index")
    if sample.decision is None:
        raise ValueError("rewrite sample requires decision")
    space_snapshot = accepted_space.snapshot()
    context = (
        *build_state_context(state.snapshot()),
        *build_action_space_context(space_snapshot),
    )
    prefix: list[Token] = []
    candidate_index = _decision_candidate_index(sample.decision)
    if (
        candidate_index < 0
        or candidate_index >= len(space_snapshot["candidate_templates"])
    ):
        raise ValueError("invalid candidate_index")
    candidate_token = T("CAND", candidate_index=candidate_index)
    total_log_prob += _score_token(
        scorer, context, tuple(prefix), _candidate_legal(space_snapshot), candidate_token
    )
    prefix.append(candidate_token)
    candidate = space_snapshot["candidate_templates"][candidate_index]
    left_mask = _validate_mask(
        sample.decision,
        "left_mask",
        len(candidate["left_definition"]["terms"]),
    )
    right_mask = _validate_mask(
        sample.decision,
        "right_mask",
        len(candidate["right_definition"]["terms"]),
    )
    total_log_prob += _score_bits(
        scorer=scorer,
        context=context,
        prefix=prefix,
        kind_prefix="LEFT",
        bits=left_mask,
    )
    total_log_prob += _score_bits(
        scorer=scorer,
        context=context,
        prefix=prefix,
        kind_prefix="RIGHT",
        bits=right_mask,
    )
    return total_log_prob
