from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .constants import SENTINEL, TOKEN_KIND
from .types import PolicyConfig, TokenTree

_ID_FIELDS = (
    "def_index",
    "term_index",
    "factor_index",
    "tensor_id",
    "range_id",
    "index_id",
    "candidate_index",
)
_FIELD_EMBEDDING_KEY_COUNT = 3 + len(_ID_FIELDS)
_FIXED_PARAM_KEY_COUNT = _FIELD_EMBEDDING_KEY_COUNT + 1 + 2 + 4
_ATTENTION_KEYS_PER_LAYER = 8


def _normal(key, shape, scale):
    return jax.random.normal(key, shape, dtype=jnp.float32) * scale


def _split(key, count):
    return iter(jax.random.split(key, count))


def init_policy_params(config: PolicyConfig, rng) -> dict[str, object]:
    key_count = _FIXED_PARAM_KEY_COUNT + (
        _ATTENTION_KEYS_PER_LAYER * config.num_attention_layers
    )
    keys = _split(rng, key_count)
    d = config.d_model
    params: dict[str, object] = {
        "field_embeddings": {
            "token_kind": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (int(max(TOKEN_KIND)), d), config.init_scale),
                ]
            ),
            "segment": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (4, d), config.init_scale),
                ]
            ),
            "side": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (3, d), config.init_scale),
                ]
            ),
            "def_index": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (config.id_vocab_size, d), config.init_scale),
                ]
            ),
            "term_index": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (config.id_vocab_size, d), config.init_scale),
                ]
            ),
            "factor_index": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (config.id_vocab_size, d), config.init_scale),
                ]
            ),
            "tensor_id": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (config.id_vocab_size, d), config.init_scale),
                ]
            ),
            "range_id": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (config.id_vocab_size, d), config.init_scale),
                ]
            ),
            "index_id": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (config.id_vocab_size, d), config.init_scale),
                ]
            ),
            "candidate_index": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (config.id_vocab_size, d), config.init_scale),
                ]
            ),
        },
        "numeric_projection": _normal(next(keys), (3, d), config.init_scale),
        "attention": [
            {
                "wq": _normal(next(keys), (d, d), config.init_scale),
                "wk": _normal(next(keys), (d, d), config.init_scale),
                "wv": _normal(next(keys), (d, d), config.init_scale),
                "wo": _normal(next(keys), (d, d), config.init_scale),
                "w1": _normal(next(keys), (d, 2 * d), config.init_scale),
                "b1": jnp.zeros((2 * d,), dtype=jnp.float32),
                "w2": _normal(next(keys), (2 * d, d), config.init_scale),
                "b2": jnp.zeros((d,), dtype=jnp.float32),
            }
            for _ in range(config.num_attention_layers)
        ],
        "target": {
            "stop_w": _normal(next(keys), (d,), config.init_scale),
            "def_w": _normal(next(keys), (d,), config.init_scale),
            "stop_bias": jnp.asarray(config.stop_bias_init, dtype=jnp.float32),
            "def_bias": jnp.asarray(0.0, dtype=jnp.float32),
        },
        "action": {
            "candidate_w": _normal(next(keys), (d,), config.init_scale),
            "candidate_slot_bias": jnp.zeros(
                (config.max_candidates,), dtype=jnp.float32
            ),
            "left_w": _normal(next(keys), (d,), config.init_scale),
            "right_w": _normal(next(keys), (d,), config.init_scale),
            "left_position_bias": jnp.zeros(
                (config.max_side_terms,), dtype=jnp.float32
            ),
            "right_position_bias": jnp.zeros(
                (config.max_side_terms,), dtype=jnp.float32
            ),
            "left_context_w": _normal(next(keys), (d,), config.init_scale),
        },
    }
    return params


def _token_kind_index(values, table):
    return jnp.where(values <= 0, 0, jnp.minimum(values, table.shape[0] - 1))


def _embedding_index(values, table):
    size = table.shape[0] - 1
    return jnp.where(values < 0, 0, (values % size) + 1)


def _field(tokens: TokenTree, name: str, length: int):
    return tokens.get(name, jnp.full((length,), SENTINEL, dtype=jnp.int32))


def embed_tokens(params, tokens: TokenTree):
    tables = params["field_embeddings"]
    d = tables["token_kind"].shape[1]
    length = tokens["token_kind"].shape[0]
    out = jnp.zeros((length, d), dtype=jnp.float32)
    if "token_kind" in tokens:
        out = out + tables["token_kind"][
            _token_kind_index(tokens["token_kind"], tables["token_kind"])
        ]
    for field in ("segment", "side"):
        if field in tokens:
            out = out + tables[field][_embedding_index(tokens[field], tables[field])]
    for field in _ID_FIELDS:
        if field in tokens:
            out = out + tables[field][_embedding_index(tokens[field], tables[field])]
    coeff_num = _field(tokens, "coeff_num", length)
    coeff_den = _field(tokens, "coeff_den", length)
    position = _field(tokens, "position", length)
    numeric = jnp.stack(
        [
            jnp.where(coeff_num < 0, 0.0, coeff_num.astype(jnp.float32)),
            jnp.where(coeff_den < 0, 0.0, coeff_den.astype(jnp.float32)),
            jnp.where(position < 0, 0.0, position.astype(jnp.float32)),
        ],
        axis=-1,
    )
    return out + numeric @ params["numeric_projection"]


def _layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _attention_block(layer, x, mask):
    q = x @ layer["wq"]
    k = x @ layer["wk"]
    v = x @ layer["wv"]
    scale = 1.0 / math.sqrt(x.shape[-1])
    scores = (q @ k.T) * scale
    scores = jnp.where(mask[None, :], scores, -1.0e30)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = weights @ v
    x = _layer_norm(x + attended @ layer["wo"])
    hidden = jax.nn.gelu(x @ layer["w1"] + layer["b1"])
    return _layer_norm(x + hidden @ layer["w2"] + layer["b2"])


def encode_tokens(params, embedded, mask):
    x = jnp.where(mask[:, None], embedded, 0.0)
    for layer in params["attention"]:
        x = _attention_block(layer, x, mask)
        x = jnp.where(mask[:, None], x, 0.0)
    return x


def masked_mean(values, mask):
    weights = mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.sum(values * weights[:, None], axis=0) / denom


def pool_by_index(values, item_index, mask, requested_indices):
    def one(index):
        item_mask = mask & (item_index == index)
        return masked_mean(values, item_mask)

    return jax.vmap(one)(requested_indices)
