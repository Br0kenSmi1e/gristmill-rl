import json

import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics import TensorComputation
from gristmill_symbolics.grammar import FlatDefinitionGrammar
from gristmill_symbolics.sampling import (
    generated_ids_to_tensor_computation,
    sample_tensor_computations,
)
from gristmill_symbolics.tokenizer import FlatDefinitionTokenizer


def _tokenizer() -> FlatDefinitionTokenizer:
    return FlatDefinitionTokenizer(
        max_range_id=0,
        max_tensor_id=2,
        max_index_id=0,
        coeff_nums=(1,),
        coeff_dens=(1,),
    )


def _source_computation() -> TensorComputation:
    return TensorComputation.from_json_string(
        json.dumps(
            {
                "ranges": [{"id": 0, "size": 3}],
                "tensors": [{"id": 0, "symmetry": []}],
                "definitions": [
                    {
                        "base": 0,
                        "ext_indices": [{"id": 0, "range": 0}],
                        "terms": [
                            {
                                "coeff": [1, 1],
                                "sum_indices": [],
                                "factors": [{"tensor": 0, "indices": [0]}],
                            }
                        ],
                    }
                ],
            }
        )
    )


def _id(tokenizer: FlatDefinitionTokenizer, kind: str, offset: int = 0) -> int:
    return tokenizer.token_ids_for_kind(kind)[offset]


def _generated_row(tokenizer: FlatDefinitionTokenizer, factor_tensor_offset: int = 0):
    content_ids = [
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid", 1),
        _id(tokenizer, "indexid"),
        _id(tokenizer, "rangeid"),
        _id(tokenizer, "coeff_num"),
        _id(tokenizer, "coeff_den"),
        _id(tokenizer, "tensorid", factor_tensor_offset),
        _id(tokenizer, "indexid"),
        _id(tokenizer, "def_end"),
    ]
    return [
        tokenizer.bos_token_id,
        *content_ids,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ]


def _scripted_model(tokenizer: FlatDefinitionTokenizer, choices: list[int]):
    def model(source_ids, decoder_input_ids, *, deterministic=True):
        assert deterministic is True
        batch_size = source_ids.shape[0]
        target_len = decoder_input_ids.shape[1]
        logits = jnp.full(
            (batch_size, target_len, tokenizer.vocab_size),
            -1000.0,
            dtype=jnp.float32,
        )
        for position, token_id in enumerate(choices):
            logits = logits.at[:, position, token_id].set(1000.0)
        return logits

    return model


def test_generated_ids_to_tensor_computation_reuses_source_envelope():
    tokenizer = _tokenizer()
    source = _source_computation()

    candidate = generated_ids_to_tensor_computation(
        source,
        tokenizer,
        _generated_row(tokenizer),
    )

    assert candidate.snapshot() == {
        "ranges": [{"id": 0, "size": 3}],
        "tensors": [{"id": 0, "symmetry": []}, {"id": 1, "symmetry": []}],
        "definitions": [
            {
                "base": 1,
                "ext_indices": [{"id": 0, "range": 0}],
                "terms": [
                    {
                        "coeff": {"numer": 1, "denom": 1},
                        "sum_indices": [],
                        "factors": [{"tensor": 0, "indices": [0]}],
                    }
                ],
            }
        ],
    }


def test_generated_ids_to_tensor_computation_rejects_unknown_factor_tensor():
    tokenizer = _tokenizer()

    with pytest.raises(ValueError, match="unknown tensor_id:2"):
        generated_ids_to_tensor_computation(
            _source_computation(),
            tokenizer,
            _generated_row(tokenizer, factor_tensor_offset=2),
        )


def test_sample_tensor_computations_wraps_token_sampler():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    generated = _generated_row(tokenizer)
    choices = generated[1:-1]
    source_ids = jnp.asarray([[1, 0, 0]], dtype=jnp.int32)

    candidates, metrics = sample_tensor_computations(
        _scripted_model(tokenizer, choices),
        jax.random.key(0),
        _source_computation(),
        source_ids,
        tokenizer,
        grammar,
        target_len=len(generated),
    )

    assert len(candidates) == 1
    assert metrics == {
        "total_samples": 1,
        "decode_failures": 0,
        "reconstruction_failures": 0,
        "verifier_failures": 0,
        "valid_samples": 1,
    }
    assert candidates[0].snapshot()["definitions"][0]["base"] == 1


def test_sample_tensor_computations_counts_decode_failures_separately():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    choices = [
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid", 1),
        _id(tokenizer, "def_end"),
    ]
    source_ids = jnp.asarray([[1, 0, 0]], dtype=jnp.int32)

    candidates, metrics = sample_tensor_computations(
        _scripted_model(tokenizer, choices),
        jax.random.key(1),
        _source_computation(),
        source_ids,
        tokenizer,
        grammar,
        target_len=4,
    )

    assert candidates == []
    assert metrics == {
        "total_samples": 1,
        "decode_failures": 1,
        "reconstruction_failures": 0,
        "verifier_failures": 0,
        "valid_samples": 0,
    }
