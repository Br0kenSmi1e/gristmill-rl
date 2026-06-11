from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .model import embed_tokens, encode_tokens, masked_mean, pool_by_index


def _concrete_int(value):
    try:
        return int(np.asarray(value))
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
    target_choice = jnp.where(sampled_index == 0, -1, sampled_index - 1).astype(
        jnp.int32
    )
    return target_choice, score_target(
        params, state_tokens, state_token_mask, def_mask, target_choice
    )
