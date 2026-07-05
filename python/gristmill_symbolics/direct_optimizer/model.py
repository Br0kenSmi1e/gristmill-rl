from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from .tokens import (
    KIND,
    KEYWORD,
    SCALAR_TYPE,
    SENTINEL,
    TOKEN_FIELDS,
    validate_scalar_bounds,
)


def make_decoder_inputs(
    target_tokens: Mapping[str, Any],
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array]:
    target = _token_arrays(target_tokens)
    target_len = target["kind"].shape[0]
    real_length = int(jnp.sum(target["mask"]))
    if real_length >= target_len:
        raise ValueError("target has no room for EOS label")

    decoder_input = _pad_like(target_len)
    labels = _pad_like(target_len)

    decoder_input["kind"] = decoder_input["kind"].at[0].set(KIND["BOS"])
    decoder_input["mask"] = decoder_input["mask"].at[0].set(True)
    for field in TOKEN_FIELDS:
        decoder_input[field] = decoder_input[field].at[1:].set(target[field][:-1])
        labels[field] = labels[field].at[:real_length].set(target[field][:real_length])

    labels["kind"] = labels["kind"].at[real_length].set(KIND["EOS"])
    labels["keyword"] = labels["keyword"].at[real_length].set(SENTINEL)
    labels["scalar_type"] = labels["scalar_type"].at[real_length].set(SENTINEL)
    labels["scalar_value"] = labels["scalar_value"].at[real_length].set(SENTINEL)
    labels["mask"] = labels["mask"].at[real_length].set(True)

    return decoder_input, labels, labels["mask"]


def token_log_probs(
    logits: Mapping[str, Any],
    target_tokens: Mapping[str, Any],
) -> jax.Array:
    target = _token_arrays(target_tokens)
    scalar_value_logits = jnp.asarray(logits["scalar_value"])
    scalar_value_min = int(logits["scalar_value_min"])
    validate_scalar_bounds(
        target_tokens,
        scalar_value_min=scalar_value_min,
        scalar_value_max=scalar_value_min + scalar_value_logits.shape[-1] - 1,
    )

    kind = target["kind"]
    token_scores = _take_log_probs(jnp.asarray(logits["kind"]), kind)

    keyword_mask = kind == KIND["KEYWORD"]
    keyword_scores = _take_log_probs(
        jnp.asarray(logits["keyword"]),
        _safe_index(target["keyword"], keyword_mask),
    )
    token_scores = token_scores + jnp.where(keyword_mask, keyword_scores, 0.0)

    scalar_mask = kind == KIND["SCALAR"]
    scalar_type_scores = _take_log_probs(
        jnp.asarray(logits["scalar_type"]),
        _safe_index(target["scalar_type"], scalar_mask),
    )
    scalar_value_index = target["scalar_value"] - scalar_value_min
    scalar_value_scores = _take_log_probs(
        scalar_value_logits,
        _safe_index(scalar_value_index, scalar_mask),
    )
    token_scores = token_scores + jnp.where(
        scalar_mask,
        scalar_type_scores + scalar_value_scores,
        0.0,
    )

    return token_scores


def sequence_log_prob(
    logits: Mapping[str, Any],
    target_tokens: Mapping[str, Any],
    target_mask: jax.Array,
) -> jax.Array:
    return jnp.sum(token_log_probs(logits, target_tokens) * target_mask, axis=1)


def _token_arrays(tokens: Mapping[str, Any]) -> dict[str, jax.Array]:
    missing = [field for field in TOKEN_FIELDS if field not in tokens]
    if missing:
        raise ValueError(f"missing token fields: {missing}")
    arrays = {field: jnp.asarray(tokens[field]) for field in TOKEN_FIELDS}
    shape = arrays["kind"].shape
    for field, array in arrays.items():
        if array.shape != shape:
            raise ValueError(
                f"expected {field} shape {array.shape} to match kind shape {shape}"
            )
    return arrays


def _pad_like(length: int) -> dict[str, jax.Array]:
    return {
        "kind": jnp.full((length,), KIND["PAD"], dtype=jnp.int32),
        "keyword": jnp.full((length,), SENTINEL, dtype=jnp.int32),
        "scalar_type": jnp.full((length,), SENTINEL, dtype=jnp.int32),
        "scalar_value": jnp.full((length,), SENTINEL, dtype=jnp.int32),
        "mask": jnp.zeros((length,), dtype=bool),
    }


def _take_log_probs(logits: jax.Array, indices: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, indices[..., None], axis=-1)[..., 0]


def _safe_index(indices: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.where(mask, indices, 0)


_NNX_MODULE = nnx.Module
_TOKEN_HEAD_SIZES = (len(KIND), len(KEYWORD), len(SCALAR_TYPE))
