from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .constants import SIDE, TOKEN_KIND
from .model import embed_tokens, encode_tokens, masked_mean, pool_by_index

_ACTION_CHOICE_KEYS = {
    "candidate_index",
    "left_mask",
    "left_valid_mask",
    "right_mask",
    "right_valid_mask",
}


def _concrete_int(value):
    try:
        return int(np.asarray(value))
    except Exception:
        return None


def _concrete_array(value):
    try:
        return np.asarray(value)
    except Exception:
        return None


def _mask_illegal_logits(logits, mask):
    return jnp.where(mask, logits, jnp.asarray(-jnp.inf, dtype=logits.dtype))


def _masked_log_softmax(logits, mask):
    return jax.nn.log_softmax(_mask_illegal_logits(logits, mask))


def _target_logits(params, state_tokens, state_token_mask, def_mask):
    encoded = encode_tokens(params, embed_tokens(params, state_tokens), state_token_mask)
    global_state = masked_mean(encoded, state_token_mask)
    def_indices = jnp.arange(def_mask.shape[0], dtype=jnp.int32)
    def_embeddings = pool_by_index(
        encoded, state_tokens["def_index"], state_token_mask, def_indices
    )
    stop_logit = (
        jnp.dot(global_state, params["target"]["stop_w"])
        + params["target"]["stop_bias"]
    )
    def_logits = jax.vmap(
        lambda x: jnp.dot(x, params["target"]["def_w"]) + params["target"]["def_bias"]
    )(def_embeddings)
    return jnp.concatenate([stop_logit[None], def_logits], axis=0)


def _validate_target_choice(def_mask, target_choice):
    choice = _concrete_int(target_choice)
    if choice is None:
        return
    if choice < -1 or choice >= def_mask.shape[0]:
        raise ValueError(
            f"target choice {choice} is outside STOP or definition range "
            f"0..{def_mask.shape[0] - 1}"
        )
    try:
        mask = np.asarray(def_mask)
    except Exception:
        return
    if choice >= 0 and not bool(mask[choice]):
        raise ValueError(f"target choice {choice} selects a masked definition")


def score_target(params, state_tokens, state_token_mask, def_mask, target_choice):
    _validate_target_choice(def_mask, target_choice)
    logits = _target_logits(params, state_tokens, state_token_mask, def_mask)
    legal = jnp.concatenate([jnp.asarray([True]), def_mask.astype(jnp.bool_)], axis=0)
    log_probs = _masked_log_softmax(logits, legal)
    choice = jnp.asarray(target_choice, dtype=jnp.int32)
    valid = (choice == -1) | ((choice >= 0) & (choice < def_mask.shape[0]))
    logit_index = jnp.where(choice == -1, 0, choice + 1)
    safe_logit_index = jnp.clip(logit_index, 0, def_mask.shape[0])
    gathered_logp = log_probs[safe_logit_index]
    return jnp.where(
        valid, gathered_logp, jnp.asarray(-jnp.inf, dtype=gathered_logp.dtype)
    )


def sample_target(params, state_tokens, state_token_mask, def_mask, rng):
    logits = _target_logits(params, state_tokens, state_token_mask, def_mask)
    legal = jnp.concatenate([jnp.asarray([True]), def_mask.astype(jnp.bool_)], axis=0)
    masked_logits = _mask_illegal_logits(logits, legal)
    sampled_index = jax.random.categorical(rng, masked_logits)
    return jnp.where(sampled_index == 0, -1, sampled_index - 1).astype(jnp.int32)


def _action_width(action_space_tokens):
    return action_space_tokens["token_kind"].shape[0]


def _candidate_indices(action_space_tokens):
    return jnp.arange(_action_width(action_space_tokens), dtype=jnp.int32)


def _state_definition_start_mask(state_tokens, state_token_mask):
    return (
        state_token_mask
        & (state_tokens["token_kind"] == int(TOKEN_KIND.DEF_START))
        & (state_tokens["def_index"] >= 0)
    )


def _state_definition_token_mask(state_tokens, state_token_mask):
    return state_token_mask & (state_tokens["def_index"] >= 0)


def _action_space_start_mask(action_space_tokens, action_space_token_mask):
    return (
        action_space_token_mask
        & (action_space_tokens["token_kind"] == int(TOKEN_KIND.ACTION_SPACE_START))
    )


def _action_space_def_index(action_space_tokens, action_space_token_mask):
    starts = _action_space_start_mask(action_space_tokens, action_space_token_mask)
    start_defs = jnp.where(starts, action_space_tokens["def_index"], -1)
    return jnp.max(start_defs), jnp.any(starts)


def _selected_def_valid(
    state_tokens,
    state_token_mask,
    selected_def_index,
    action_space_tokens,
    action_space_token_mask,
):
    selected = jnp.asarray(selected_def_index, dtype=jnp.int32)
    state_defs = _state_definition_start_mask(state_tokens, state_token_mask)
    state_has_selected = jnp.any(state_defs & (state_tokens["def_index"] == selected))
    action_space_def, has_action_space_start = _action_space_def_index(
        action_space_tokens, action_space_token_mask
    )
    return (
        (selected >= 0)
        & state_has_selected
        & has_action_space_start
        & (action_space_def == selected)
    )


def _encoded_action_context(
    params,
    state_tokens,
    state_token_mask,
    selected_def_index,
    action_space_tokens,
    action_space_token_mask,
):
    state_encoded = encode_tokens(params, embed_tokens(params, state_tokens), state_token_mask)
    action_encoded = encode_tokens(
        params, embed_tokens(params, action_space_tokens), action_space_token_mask
    )
    selected = jnp.asarray(selected_def_index, dtype=jnp.int32)
    selected_def_mask = _state_definition_token_mask(state_tokens, state_token_mask)
    selected_def = pool_by_index(
        state_encoded,
        state_tokens["def_index"],
        selected_def_mask,
        selected.reshape((1,)),
    )[0]
    valid = _selected_def_valid(
        state_tokens,
        state_token_mask,
        selected,
        action_space_tokens,
        action_space_token_mask,
    )
    return (
        action_encoded,
        masked_mean(state_encoded, state_token_mask) + selected_def,
        valid,
    )


def _candidate_valid_mask(action_space_tokens, action_space_token_mask, indices):
    candidate_starts = (
        action_space_token_mask
        & (action_space_tokens["token_kind"] == int(TOKEN_KIND.CANDIDATE_START))
        & (action_space_tokens["candidate_index"] >= 0)
    )
    return jax.vmap(
        lambda index: jnp.any(
            candidate_starts & (action_space_tokens["candidate_index"] == index)
        )
    )(indices)


def _candidate_logits(
    params,
    state_tokens,
    state_token_mask,
    selected_def_index,
    action_space_tokens,
    action_space_token_mask,
):
    action_encoded, context, selected_def_valid = _encoded_action_context(
        params,
        state_tokens,
        state_token_mask,
        selected_def_index,
        action_space_tokens,
        action_space_token_mask,
    )
    indices = _candidate_indices(action_space_tokens)
    candidate_embeddings = pool_by_index(
        action_encoded,
        action_space_tokens["candidate_index"],
        action_space_token_mask & (action_space_tokens["candidate_index"] >= 0),
        indices,
    )
    logits = jax.vmap(
        lambda embedding: jnp.dot(
            embedding + context, params["action"]["candidate_w"]
        )
        + params["action"]["candidate_bias"]
    )(candidate_embeddings)
    return (
        logits,
        _candidate_valid_mask(action_space_tokens, action_space_token_mask, indices),
        candidate_embeddings,
        action_encoded,
        context,
        selected_def_valid,
    )


def _side_terms(
    params,
    action_encoded,
    action_space_tokens,
    action_space_token_mask,
    candidate_index,
    side,
):
    width = _action_width(action_space_tokens)
    term_indices = jnp.arange(width, dtype=jnp.int32)
    side_term_tokens = (
        action_space_token_mask
        & (action_space_tokens["candidate_index"] == candidate_index)
        & (action_space_tokens["side"] == int(side))
        & (action_space_tokens["term_index"] >= 0)
    )
    term_embeddings = pool_by_index(
        action_encoded,
        action_space_tokens["term_index"],
        side_term_tokens,
        term_indices,
    )
    term_starts = (
        action_space_token_mask
        & (action_space_tokens["token_kind"] == int(TOKEN_KIND.TERM_START))
        & (action_space_tokens["candidate_index"] == candidate_index)
        & (action_space_tokens["side"] == int(side))
        & (action_space_tokens["term_index"] >= 0)
    )
    valid = jax.vmap(
        lambda term_index: jnp.any(
            term_starts & (action_space_tokens["term_index"] == term_index)
        )
    )(term_indices)
    return term_embeddings, valid


def _left_logits(params, context, candidate_embedding, left_embeddings):
    return jax.vmap(
        lambda embedding: jnp.dot(
            embedding + candidate_embedding + context, params["action"]["left_w"]
        )
        + params["action"]["left_bias"]
    )(left_embeddings)


def _right_logits(params, context, candidate_embedding, right_embeddings, left_summary):
    context_bias = jnp.dot(left_summary, params["action"]["left_context_w"])
    return jax.vmap(
        lambda embedding: jnp.dot(
            embedding + candidate_embedding + context, params["action"]["right_w"]
        )
        + params["action"]["right_bias"]
        + context_bias
    )(right_embeddings)


def _last_valid_positions(valid_mask):
    valid_int = valid_mask.astype(jnp.int32)
    remaining_from_here = jnp.cumsum(valid_int[::-1])[::-1]
    remaining_after = remaining_from_here - valid_int
    return valid_mask & (remaining_after == 0)


def _bit_logp(logit, keep):
    return jnp.where(keep, jax.nn.log_sigmoid(logit), jax.nn.log_sigmoid(-logit))


def _sample_side(logits, valid_mask, rng):
    sampled = jax.random.bernoulli(rng, p=jax.nn.sigmoid(logits))
    final_valid = _last_valid_positions(valid_mask)

    def step(seen_keep, inputs):
        valid, final, raw_keep, logit = inputs
        forced_keep = valid & (~seen_keep) & final
        keep = jnp.where(valid, forced_keep | raw_keep, False)
        logp = jnp.where(valid & (~forced_keep), _bit_logp(logit, keep), 0.0)
        return seen_keep | (valid & keep), (keep, logp)

    _, (mask, logps) = jax.lax.scan(
        step, jnp.asarray(False), (valid_mask, final_valid, sampled, logits)
    )
    return mask, jnp.sum(logps)


def _score_side(logits, valid_mask, mask):
    mask = mask.astype(jnp.bool_)
    final_valid = _last_valid_positions(valid_mask)

    def step(seen_keep, inputs):
        valid, final, keep, logit = inputs
        forced_keep = valid & (~seen_keep) & final
        selected_valid = valid & keep
        legal = jnp.where(valid, jnp.where(forced_keep, keep, True), ~keep)
        logp = jnp.where(valid & (~forced_keep), _bit_logp(logit, keep), 0.0)
        return seen_keep | selected_valid, (logp, legal)

    _, (logps, legal_bits) = jax.lax.scan(
        step, jnp.asarray(False), (valid_mask, final_valid, mask, logits)
    )
    valid = (
        jnp.all(legal_bits)
        & jnp.any(valid_mask)
        & jnp.any(mask & valid_mask)
        & (~jnp.any(mask & (~valid_mask)))
    )
    return jnp.sum(logps), valid


def _concrete_candidate_valid_mask(action_space_tokens, action_space_token_mask, count):
    token_candidates = _concrete_array(action_space_tokens["candidate_index"])
    token_kinds = _concrete_array(action_space_tokens["token_kind"])
    token_mask = _concrete_array(action_space_token_mask)
    if token_candidates is None or token_kinds is None or token_mask is None:
        return None
    return np.asarray(
        [
            bool(
                np.any(
                    token_mask
                    & (token_kinds == int(TOKEN_KIND.CANDIDATE_START))
                    & (token_candidates == candidate)
                )
            )
            for candidate in range(count)
        ],
        dtype=bool,
    )


def _concrete_side_valid_mask(
    action_space_tokens,
    action_space_token_mask,
    candidate_index,
    side,
):
    token_candidates = _concrete_array(action_space_tokens["candidate_index"])
    token_sides = _concrete_array(action_space_tokens["side"])
    token_terms = _concrete_array(action_space_tokens["term_index"])
    token_kinds = _concrete_array(action_space_tokens["token_kind"])
    token_mask = _concrete_array(action_space_token_mask)
    if (
        token_candidates is None
        or token_sides is None
        or token_terms is None
        or token_kinds is None
        or token_mask is None
    ):
        return None
    width = _action_width(action_space_tokens)
    return np.asarray(
        [
            bool(
                np.any(
                    token_mask
                    & (token_kinds == int(TOKEN_KIND.TERM_START))
                    & (token_candidates == candidate_index)
                    & (token_sides == int(side))
                    & (token_terms == term_index)
                )
            )
            for term_index in range(width)
        ],
        dtype=bool,
    )


def _concrete_state_def_indices(state_tokens, state_token_mask):
    token_defs = _concrete_array(state_tokens["def_index"])
    token_kinds = _concrete_array(state_tokens["token_kind"])
    token_mask = _concrete_array(state_token_mask)
    if token_defs is None or token_kinds is None or token_mask is None:
        return None
    real_defs = (
        token_mask.astype(bool)
        & (token_kinds == int(TOKEN_KIND.DEF_START))
        & (token_defs >= 0)
    )
    return {int(value) for value in token_defs[real_defs].tolist()}


def _concrete_action_space_def_index(action_space_tokens, action_space_token_mask):
    token_defs = _concrete_array(action_space_tokens["def_index"])
    token_kinds = _concrete_array(action_space_tokens["token_kind"])
    token_mask = _concrete_array(action_space_token_mask)
    if token_defs is None or token_kinds is None or token_mask is None:
        return None
    starts = token_mask.astype(bool) & (
        token_kinds == int(TOKEN_KIND.ACTION_SPACE_START)
    )
    if not bool(np.any(starts)):
        raise ValueError("action space tokens contain no ACTION_SPACE_START")
    start_defs = token_defs[starts]
    return int(start_defs[0])


def _validate_selected_def_index(
    state_tokens,
    state_token_mask,
    selected_def_index,
    action_space_tokens,
    action_space_token_mask,
):
    selected = _concrete_int(selected_def_index)
    if selected is None:
        return
    if selected < 0:
        raise ValueError(f"selected_def_index {selected} must select a definition")

    state_defs = _concrete_state_def_indices(state_tokens, state_token_mask)
    if state_defs is not None and selected not in state_defs:
        raise ValueError(f"selected_def_index {selected} is not present in state tokens")

    action_space_def = _concrete_action_space_def_index(
        action_space_tokens, action_space_token_mask
    )
    if action_space_def is not None and action_space_def != selected:
        raise ValueError(
            f"selected_def_index {selected} does not match action space def_index "
            f"{action_space_def}"
        )


def _validate_side_choice(
    params,
    action_space_tokens,
    action_space_token_mask,
    action_choice,
    *,
    side_name,
    side,
    candidate_index,
):
    mask = _concrete_array(action_choice[f"{side_name}_mask"])
    provided_valid = _concrete_array(action_choice[f"{side_name}_valid_mask"])
    if mask is not None and provided_valid is not None:
        selected = mask.astype(bool) & provided_valid.astype(bool)
        if not bool(np.any(provided_valid)):
            raise ValueError(f"{side_name}_valid_mask selects no valid side terms")
        if bool(np.any(mask.astype(bool) & ~provided_valid.astype(bool))):
            raise ValueError(f"{side_name}_mask selects an invalid side term")
        if not bool(np.any(selected)):
            raise ValueError(f"empty {side_name}_mask")
    if candidate_index is None:
        return
    computed_valid = _concrete_side_valid_mask(
        action_space_tokens, action_space_token_mask, candidate_index, side
    )
    if computed_valid is None:
        return
    if not bool(np.any(computed_valid)):
        raise ValueError(f"candidate_index {candidate_index} has no valid {side_name} terms")
    if provided_valid is not None and not bool(
        np.array_equal(provided_valid.astype(bool), computed_valid)
    ):
        raise ValueError(f"{side_name}_valid_mask does not match action space")
    if mask is not None and not bool(np.any(mask.astype(bool) & computed_valid)):
        raise ValueError(f"empty {side_name}_mask")


def _validate_action_choice(
    params, action_space_tokens, action_space_token_mask, action_choice
):
    if set(action_choice) != _ACTION_CHOICE_KEYS:
        raise ValueError(
            "action_choice keys must be exactly "
            "candidate_index, left_mask, left_valid_mask, right_mask, right_valid_mask"
        )
    candidate = jnp.asarray(action_choice["candidate_index"], dtype=jnp.int32)
    if candidate.shape != ():
        raise ValueError(f"candidate_index must be scalar, got shape {candidate.shape}")
    width = _action_width(action_space_tokens)
    for name in ("left_mask", "left_valid_mask", "right_mask", "right_valid_mask"):
        values = jnp.asarray(action_choice[name], dtype=jnp.bool_)
        if values.shape != (width,):
            raise ValueError(f"{name} must have shape ({width},), got {values.shape}")

    candidate_index = _concrete_int(action_choice["candidate_index"])
    candidate_count = _action_width(action_space_tokens)
    if candidate_index is not None:
        if candidate_index < 0 or candidate_index >= candidate_count:
            raise ValueError(f"candidate_index {candidate_index} is illegal")
        candidate_valid = _concrete_candidate_valid_mask(
            action_space_tokens, action_space_token_mask, candidate_count
        )
        if candidate_valid is not None and not bool(candidate_valid[candidate_index]):
            raise ValueError(f"candidate_index {candidate_index} is illegal")

    _validate_side_choice(
        params,
        action_space_tokens,
        action_space_token_mask,
        action_choice,
        side_name="left",
        side=SIDE.LEFT,
        candidate_index=candidate_index,
    )
    _validate_side_choice(
        params,
        action_space_tokens,
        action_space_token_mask,
        action_choice,
        side_name="right",
        side=SIDE.RIGHT,
        candidate_index=candidate_index,
    )


def score_action(
    params,
    state_tokens,
    state_token_mask,
    selected_def_index,
    action_space_tokens,
    action_space_token_mask,
    action_choice,
):
    _validate_selected_def_index(
        state_tokens,
        state_token_mask,
        selected_def_index,
        action_space_tokens,
        action_space_token_mask,
    )
    _validate_action_choice(params, action_space_tokens, action_space_token_mask, action_choice)
    (
        candidate_logits,
        candidate_valid_mask,
        candidate_embeddings,
        action_encoded,
        context,
        selected_def_is_valid,
    ) = _candidate_logits(
        params,
        state_tokens,
        state_token_mask,
        selected_def_index,
        action_space_tokens,
        action_space_token_mask,
    )
    candidate_log_probs = _masked_log_softmax(candidate_logits, candidate_valid_mask)
    candidate = jnp.asarray(action_choice["candidate_index"], dtype=jnp.int32)
    candidate_in_range = (candidate >= 0) & (candidate < candidate_valid_mask.shape[0])
    safe_candidate = jnp.clip(candidate, 0, candidate_valid_mask.shape[0] - 1)
    candidate_is_valid = candidate_in_range & candidate_valid_mask[safe_candidate]

    left_embeddings, left_valid = _side_terms(
        params,
        action_encoded,
        action_space_tokens,
        action_space_token_mask,
        safe_candidate,
        SIDE.LEFT,
    )
    candidate_embedding = candidate_embeddings[safe_candidate]
    left_logits = _left_logits(params, context, candidate_embedding, left_embeddings)
    left_mask = jnp.asarray(action_choice["left_mask"], dtype=jnp.bool_)
    provided_left_valid = jnp.asarray(action_choice["left_valid_mask"], dtype=jnp.bool_)
    left_valid_matches = jnp.all(provided_left_valid == left_valid)
    left_logp, left_is_valid = _score_side(left_logits, left_valid, left_mask)
    left_summary = masked_mean(left_embeddings, left_valid & left_mask)

    right_embeddings, right_valid = _side_terms(
        params,
        action_encoded,
        action_space_tokens,
        action_space_token_mask,
        safe_candidate,
        SIDE.RIGHT,
    )
    right_logits = _right_logits(
        params, context, candidate_embedding, right_embeddings, left_summary
    )
    provided_right_valid = jnp.asarray(
        action_choice["right_valid_mask"], dtype=jnp.bool_
    )
    right_valid_matches = jnp.all(provided_right_valid == right_valid)
    right_logp, right_is_valid = _score_side(
        right_logits,
        right_valid,
        jnp.asarray(action_choice["right_mask"], dtype=jnp.bool_),
    )
    logp = candidate_log_probs[safe_candidate] + left_logp + right_logp
    valid = (
        selected_def_is_valid
        & candidate_is_valid
        & left_is_valid
        & right_is_valid
        & left_valid_matches
        & right_valid_matches
    )
    return jnp.where(valid, logp, jnp.asarray(-jnp.inf, dtype=logp.dtype))


def sample_action(
    params,
    state_tokens,
    state_token_mask,
    selected_def_index,
    action_space_tokens,
    action_space_token_mask,
    rng,
):
    _validate_selected_def_index(
        state_tokens,
        state_token_mask,
        selected_def_index,
        action_space_tokens,
        action_space_token_mask,
    )
    candidate_rng, left_rng, right_rng = jax.random.split(rng, 3)
    (
        candidate_logits,
        candidate_valid_mask,
        candidate_embeddings,
        action_encoded,
        context,
        _selected_def_is_valid,
    ) = _candidate_logits(
        params,
        state_tokens,
        state_token_mask,
        selected_def_index,
        action_space_tokens,
        action_space_token_mask,
    )
    candidate = jax.random.categorical(
        candidate_rng, _mask_illegal_logits(candidate_logits, candidate_valid_mask)
    ).astype(jnp.int32)
    candidate_embedding = candidate_embeddings[candidate]

    left_embeddings, left_valid = _side_terms(
        params,
        action_encoded,
        action_space_tokens,
        action_space_token_mask,
        candidate,
        SIDE.LEFT,
    )
    left_logits = _left_logits(params, context, candidate_embedding, left_embeddings)
    left_mask, _ = _sample_side(left_logits, left_valid, left_rng)
    left_summary = masked_mean(left_embeddings, left_valid & left_mask)

    right_embeddings, right_valid = _side_terms(
        params,
        action_encoded,
        action_space_tokens,
        action_space_token_mask,
        candidate,
        SIDE.RIGHT,
    )
    right_logits = _right_logits(
        params, context, candidate_embedding, right_embeddings, left_summary
    )
    right_mask, _ = _sample_side(right_logits, right_valid, right_rng)

    return {
        "candidate_index": candidate,
        "left_mask": left_mask,
        "left_valid_mask": left_valid,
        "right_mask": right_mask,
        "right_valid_mask": right_valid,
    }
