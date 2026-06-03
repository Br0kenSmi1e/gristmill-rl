from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from transformer_policy.types import PayloadValue, Token


TOKEN_KINDS = (
    "ACTION_SPACE_END",
    "ACTION_SPACE_START",
    "BASE",
    "CAND",
    "CAND_END",
    "CAND_START",
    "COEFF_DEN",
    "COEFF_NUM",
    "DEF",
    "DEF_END",
    "DEF_START",
    "END",
    "EXT_INDEX",
    "FACTOR",
    "INDEX",
    "LEFT_DEF_END",
    "LEFT_DEF_START",
    "LEFT_DROP",
    "LEFT_KEEP",
    "REWRITTEN_DEF_END",
    "REWRITTEN_DEF_START",
    "RIGHT_DEF_END",
    "RIGHT_DEF_START",
    "RIGHT_DROP",
    "RIGHT_KEEP",
    "STATE_DEF",
    "STATE_END",
    "STATE_START",
    "STOP",
    "SUM_INDEX",
    "TERM_END",
    "TERM_START",
)
TOKEN_KIND_TO_ID = {kind: index for index, kind in enumerate(TOKEN_KINDS)}

PAYLOAD_KEYS = (
    "accepted",
    "arity",
    "candidate_index",
    "def_index",
    "id",
    "position",
    "range",
    "tensor",
    "value",
)
PAYLOAD_KEY_TO_COLUMN = {key: index for index, key in enumerate(PAYLOAD_KEYS)}
TOKEN_FEATURE_DIM = 2 + len(PAYLOAD_KEYS)


def _payload_value(value: PayloadValue) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return float(int.from_bytes(digest, byteorder="big")) / float(2**32)


def token_features(tokens: tuple[Token, ...]) -> np.ndarray:
    features = np.zeros((len(tokens), TOKEN_FEATURE_DIM), dtype=np.float32)
    denominator = max(len(TOKEN_KINDS) - 1, 1)
    for row, token in enumerate(tokens):
        if token.kind not in TOKEN_KIND_TO_ID:
            raise ValueError(f"unknown token kind '{token.kind}'")
        features[row, 0] = float(TOKEN_KIND_TO_ID[token.kind]) / float(denominator)
        features[row, 1] = float(row)
        for key, value in token.payload:
            if key not in PAYLOAD_KEY_TO_COLUMN:
                continue
            features[row, 2 + PAYLOAD_KEY_TO_COLUMN[key]] = _payload_value(value)
    return features


class TokenEmbedder(nnx.Module):
    def __init__(self, *, hidden_dim: int, rngs: nnx.Rngs):
        self.proj = nnx.Linear(TOKEN_FEATURE_DIM, hidden_dim, rngs=rngs)

    def __call__(self, tokens: tuple[Token, ...]) -> jax.Array:
        features = jnp.asarray(token_features(tokens), dtype=jnp.float32)
        return nnx.relu(self.proj(features))
