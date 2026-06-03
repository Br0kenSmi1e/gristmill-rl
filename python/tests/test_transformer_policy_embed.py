import json
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
from flax import nnx

from transformer_policy.embed import (
    PAYLOAD_KEYS,
    TOKEN_FEATURE_DIM,
    TokenEmbedder,
    token_features,
)
from transformer_policy.types import T


def test_token_features_are_deterministic_float32_matrix():
    tokens = (
        T("DEF_START"),
        T("FACTOR", tensor=3, position=1, arity=2),
    )

    features = token_features(tokens)

    assert features.shape == (2, TOKEN_FEATURE_DIM)
    assert features.dtype == np.float32
    assert TOKEN_FEATURE_DIM == 2 + len(PAYLOAD_KEYS)
    np.testing.assert_array_equal(features, token_features(tokens))


def test_token_features_encode_strings_deterministically_across_processes():
    script = """
import json

from transformer_policy.embed import token_features
from transformer_policy.types import T

print(json.dumps(token_features((T("DEF", id="abc"),)).tolist()))
"""

    def run_with_hash_seed(seed: str) -> list[list[float]]:
        output = subprocess.check_output(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        return json.loads(output)

    assert run_with_hash_seed("1") == run_with_hash_seed("2")


def test_token_features_place_payloads_in_fixed_columns():
    tokens = (T("FACTOR", arity=2, id="abc", position=1, tensor=3),)

    features = token_features(tokens)
    repeated_features = token_features(tokens)

    def payload_column(key: str) -> int:
        return 2 + PAYLOAD_KEYS.index(key)

    assert features[0, payload_column("arity")] == np.float32(2.0)
    assert features[0, payload_column("position")] == np.float32(1.0)
    assert features[0, payload_column("tensor")] == np.float32(3.0)
    assert features[0, payload_column("id")] == repeated_features[0, payload_column("id")]


def test_token_features_reject_unknown_kind():
    tokens = (T("UNKNOWN_KIND"),)

    try:
        token_features(tokens)
    except ValueError as error:
        assert "unknown token kind" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_token_embedder_projects_tokens_to_hidden_vectors():
    tokens = (
        T("STATE_START"),
        T("STATE_END"),
    )
    embedder = TokenEmbedder(hidden_dim=8, rngs=nnx.Rngs(0))

    values = embedder(tokens)

    assert values.shape == (2, 8)
    assert np.isfinite(np.asarray(values)).all()


def test_token_embedder_returns_linear_projection():
    tokens = (
        T("STATE_START"),
        T("STATE_END"),
    )
    embedder = TokenEmbedder(hidden_dim=8, rngs=nnx.Rngs(0))

    features = jnp.asarray(token_features(tokens), dtype=jnp.float32)
    values = embedder(tokens)

    np.testing.assert_array_equal(
        np.asarray(values),
        np.asarray(embedder.proj(features)),
    )
