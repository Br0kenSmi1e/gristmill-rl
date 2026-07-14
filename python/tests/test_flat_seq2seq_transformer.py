import importlib
import sys

import jax.numpy as jnp
import pytest
from flax import nnx

from gristmill_symbolics.nn import FlatDefinitionSeq2SeqTransformer


def _model(
    *,
    source_len: int = 5,
    target_len: int = 4,
    vocab_size: int = 13,
    pad_token_id: int = 0,
    dropout: float = 0.0,
    rng_seed: int = 0,
) -> FlatDefinitionSeq2SeqTransformer:
    return FlatDefinitionSeq2SeqTransformer(
        source_len=source_len,
        target_len=target_len,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=dropout,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(rng_seed),
    )


def test_flat_seq2seq_returns_vocab_logits():
    model = _model()
    source_ids = jnp.asarray(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    decoder_input_ids = jnp.asarray(
        [
            [1, 6, 7, 0],
            [1, 8, 0, 0],
        ],
        dtype=jnp.int32,
    )

    logits = model(source_ids, decoder_input_ids, deterministic=True)

    assert logits.shape == (2, 4, 13)
    assert logits.dtype == jnp.float32


def test_pad_token_positions_are_zeroed_after_embedding():
    model = _model()
    ids = jnp.asarray([[1, 0, 2, 0]], dtype=jnp.int32)

    vectors = model._embed(
        ids,
        model.target_position_embed,
        deterministic=True,
    )

    assert vectors.shape == (1, 4, 8)
    assert jnp.any(jnp.abs(vectors[0, 0]) > 0.0)
    assert jnp.allclose(vectors[0, 1], 0.0)
    assert jnp.any(jnp.abs(vectors[0, 2]) > 0.0)
    assert jnp.allclose(vectors[0, 3], 0.0)


def test_padding_masks_are_derived_from_ids_and_passed_to_core(monkeypatch):
    import gristmill_symbolics.nn.flat_seq2seq as flat_seq2seq

    seen = {}

    class FakeEncoder:
        def __init__(self, **kwargs):
            seen["encoder_kwargs"] = kwargs

        def __call__(self, x, source_mask=None, *, deterministic=True):
            seen["source_mask"] = source_mask
            seen["encoder_deterministic"] = deterministic
            return x

    class FakeDecoder:
        def __init__(self, **kwargs):
            seen["decoder_kwargs"] = kwargs

        def __call__(
            self,
            x,
            memory,
            *,
            target_mask=None,
            source_mask=None,
            deterministic=True,
        ):
            seen["target_mask"] = target_mask
            seen["decoder_source_mask"] = source_mask
            seen["decoder_deterministic"] = deterministic
            return x

    monkeypatch.setattr(flat_seq2seq, "TransformerEncoder", FakeEncoder)
    monkeypatch.setattr(flat_seq2seq, "TransformerDecoder", FakeDecoder)
    model = _model(source_len=4, target_len=3, rng_seed=1)
    source_ids = jnp.asarray([[1, 0, 2, 0]], dtype=jnp.int32)
    decoder_input_ids = jnp.asarray([[1, 3, 0]], dtype=jnp.int32)

    logits = model(source_ids, decoder_input_ids, deterministic=False)

    assert logits.shape == (1, 3, 13)
    assert seen["source_mask"].tolist() == [[True, False, True, False]]
    assert seen["target_mask"].tolist() == [[True, True, False]]
    assert seen["decoder_source_mask"].tolist() == [[True, False, True, False]]
    assert seen["encoder_deterministic"] is False
    assert seen["decoder_deterministic"] is False


def test_deterministic_calls_are_repeatable_with_dropout_disabled():
    model = _model(rng_seed=2)
    source_ids = jnp.asarray([[1, 2, 3, 0, 0]], dtype=jnp.int32)
    decoder_input_ids = jnp.asarray([[1, 4, 5, 0]], dtype=jnp.int32)

    first = model(source_ids, decoder_input_ids, deterministic=True)
    second = model(source_ids, decoder_input_ids, deterministic=True)

    assert jnp.allclose(first, second)


def test_flat_seq2seq_exposes_one_token_decode_step():
    model = _model(source_len=5, target_len=4, rng_seed=4)
    source_ids = jnp.asarray([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=jnp.int32)
    token_ids = jnp.asarray([1, 1], dtype=jnp.int32)

    memory, source_mask = model.encode(source_ids, deterministic=True)
    model.init_decode_cache(batch_size=2, target_len=4)
    logits = model.decode_step(
        token_ids,
        memory,
        source_mask=source_mask,
        step=0,
        deterministic=True,
    )

    assert memory.shape == (2, 5, 8)
    assert source_mask.tolist() == [
        [True, True, True, False, False],
        [True, True, False, False, False],
    ]
    assert logits.shape == (2, 13)
    assert logits.dtype == jnp.float32


def test_decode_step_uses_encoder_memory_without_full_model_call(monkeypatch):
    model = _model(source_len=3, target_len=3, rng_seed=5)
    source_ids = jnp.asarray([[1, 2, 0]], dtype=jnp.int32)
    memory, source_mask = model.encode(source_ids, deterministic=True)
    model.init_decode_cache(batch_size=1, target_len=3)

    def fail_call(*args, **kwargs):
        raise AssertionError("decode_step must not call full model")

    monkeypatch.setattr(model, "__call__", fail_call)

    logits = model.decode_step(
        jnp.asarray([1], dtype=jnp.int32),
        memory,
        source_mask=source_mask,
        step=0,
        deterministic=True,
    )

    assert logits.shape == (1, 13)


def test_flat_seq2seq_requires_integer_token_id_arrays():
    model = _model(source_len=2, target_len=2, rng_seed=3)
    source_ids = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)
    decoder_input_ids = jnp.asarray([[1, 0]], dtype=jnp.int32)

    with pytest.raises((TypeError, ValueError)):
        model(source_ids, decoder_input_ids, deterministic=True)


def test_flat_seq2seq_module_does_not_import_tokenizer_or_grammar():
    module_name = "gristmill_symbolics.nn.flat_seq2seq"
    tokenizer_name = "gristmill_symbolics.tokenizer"
    grammar_name = "gristmill_symbolics.grammar"
    saved_modules = {
        module_name: sys.modules.pop(module_name, None),
        tokenizer_name: sys.modules.pop(tokenizer_name, None),
        grammar_name: sys.modules.pop(grammar_name, None),
    }

    nn_package = sys.modules.get("gristmill_symbolics.nn")
    had_attr = nn_package is not None and hasattr(nn_package, "flat_seq2seq")
    saved_attr = getattr(nn_package, "flat_seq2seq", None) if nn_package else None

    try:
        importlib.import_module(module_name)

        assert tokenizer_name not in sys.modules
        assert grammar_name not in sys.modules
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.pop(tokenizer_name, None)
        sys.modules.pop(grammar_name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        if nn_package is not None:
            if had_attr:
                nn_package.flat_seq2seq = saved_attr
            elif hasattr(nn_package, "flat_seq2seq"):
                del nn_package.flat_seq2seq
