import json
import os
import subprocess
import sys

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
