import importlib
import sys

import jax
import jax.numpy as jnp
from flax import nnx

from gristmill_symbolics.nn import (
    DecoderBlock,
    EncoderBlock,
    TransformerDecoder,
    TransformerEncoder,
)


def _source_vectors(batch: int = 2, length: int = 5, d_model: int = 8):
    return jnp.arange(batch * length * d_model, dtype=jnp.float32).reshape(
        batch,
        length,
        d_model,
    ) / 100.0


def _target_vectors(batch: int = 2, length: int = 4, d_model: int = 8):
    return jnp.arange(batch * length * d_model, dtype=jnp.float32).reshape(
        batch,
        length,
        d_model,
    ) / 50.0


def test_encoder_and_decoder_return_vector_shapes():
    source = _source_vectors()
    target = _target_vectors()
    source_mask = jnp.array(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]
    )
    target_mask = jnp.array(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(0),
    )
    decoder = TransformerDecoder(
        d_model=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(1),
    )

    memory = encoder(source, source_mask, deterministic=True)
    decoded = decoder(
        target,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert memory.shape == source.shape
    assert decoded.shape == target.shape


def test_encoder_and_decoder_accept_dtype_fields():
    source = _source_vectors()
    target = _target_vectors()
    source_mask = jnp.ones(source.shape[:2], dtype=bool)
    target_mask = jnp.ones(target.shape[:2], dtype=bool)
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(12),
    )
    decoder = TransformerDecoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(13),
    )

    memory = encoder(source, source_mask, deterministic=True)
    decoded = decoder(
        target,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert memory.shape == source.shape
    assert decoded.shape == target.shape


def test_blocks_return_vector_shapes():
    source = _source_vectors()
    target = _target_vectors()
    source_mask = jnp.ones(source.shape[:2], dtype=bool)
    target_mask = jnp.ones(target.shape[:2], dtype=bool)
    encoder_block = EncoderBlock(
        d_model=8,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(2),
    )
    decoder_block = DecoderBlock(
        d_model=8,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(3),
    )

    memory = encoder_block(source, source_mask, deterministic=True)
    decoded = decoder_block(
        target,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert memory.shape == source.shape
    assert decoded.shape == target.shape


def test_decoder_causal_mask_hides_later_target_positions():
    source = _source_vectors(batch=1, length=3, d_model=8)
    target = _target_vectors(batch=1, length=5, d_model=8)
    changed_future = target.at[:, 4, :].add(1000.0)
    source_mask = jnp.ones(source.shape[:2], dtype=bool)
    target_mask = jnp.ones(target.shape[:2], dtype=bool)
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(4),
    )
    decoder = TransformerDecoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(5),
    )
    memory = encoder(source, source_mask, deterministic=True)

    decoded = decoder(
        target,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )
    changed_decoded = decoder(
        changed_future,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert jnp.allclose(decoded[:, :4, :], changed_decoded[:, :4, :], atol=1e-5)


def test_masked_source_positions_do_not_affect_decoder_outputs():
    memory = _source_vectors(batch=1, length=4, d_model=8)
    changed_memory = memory.at[:, 3, :].add(5000.0)
    target = _target_vectors(batch=1, length=3, d_model=8)
    source_mask = jnp.array([[True, True, True, False]])
    target_mask = jnp.ones(target.shape[:2], dtype=bool)
    decoder = TransformerDecoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(7),
    )

    decoded = decoder(
        target,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )
    changed_decoded = decoder(
        target,
        changed_memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert jnp.allclose(decoded, changed_decoded, atol=1e-5)


def test_deterministic_calls_are_repeatable_with_dropout_configured():
    source = _source_vectors(batch=1, length=3, d_model=8)
    source_mask = jnp.ones(source.shape[:2], dtype=bool)
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.25,
        rngs=nnx.Rngs(8),
    )

    first = encoder(source, source_mask, deterministic=True)
    second = encoder(source, source_mask, deterministic=True)

    assert jnp.allclose(first, second)


def test_nondeterministic_calls_forward_false_to_dropout(monkeypatch):
    source = _source_vectors(batch=1, length=3, d_model=8)
    source_mask = jnp.ones(source.shape[:2], dtype=bool)
    seen: list[bool | None] = []
    original_dropout_call = nnx.Dropout.__call__

    def spy_dropout(self, inputs, *, deterministic=None, rngs=None):
        seen.append(deterministic)
        return original_dropout_call(
            self,
            inputs,
            deterministic=deterministic,
            rngs=rngs,
        )

    monkeypatch.setattr(nnx.Dropout, "__call__", spy_dropout)
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.25,
        rngs=nnx.Rngs(10),
    )

    encoder(source, source_mask, deterministic=False)

    assert seen
    assert all(deterministic is False for deterministic in seen)


def test_attention_implementation_is_passed_to_jax_attention(monkeypatch):
    source = _source_vectors(batch=1, length=3, d_model=8)
    target = _target_vectors(batch=1, length=2, d_model=8)
    source_mask = jnp.ones(source.shape[:2], dtype=bool)
    target_mask = jnp.ones(target.shape[:2], dtype=bool)
    seen: list[str | None] = []
    original_attention = jax.nn.dot_product_attention

    def spy_attention(*args, **kwargs):
        seen.append(kwargs.get("implementation"))
        return original_attention(*args, **kwargs)

    monkeypatch.setattr(jax.nn, "dot_product_attention", spy_attention)
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        attention_implementation="xla",
        rngs=nnx.Rngs(9),
    )

    encoder(source, source_mask, deterministic=True)

    assert seen == ["xla"]
    seen.clear()

    decoder = TransformerDecoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        attention_implementation="xla",
        rngs=nnx.Rngs(11),
    )

    decoder(
        target,
        source,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert seen == ["xla", "xla"]


def test_transformer_core_does_not_import_tokenizer():
    transformer_name = "gristmill_symbolics.nn.transformer"
    tokenizer_name = "gristmill_symbolics.tokenizer"
    saved_modules = {
        transformer_name: sys.modules.pop(transformer_name, None),
        tokenizer_name: sys.modules.pop(tokenizer_name, None),
    }

    nn_package = sys.modules.get("gristmill_symbolics.nn")
    had_transformer_attr = nn_package is not None and hasattr(nn_package, "transformer")
    saved_transformer_attr = (
        getattr(nn_package, "transformer", None) if nn_package is not None else None
    )

    try:
        importlib.import_module(transformer_name)

        assert tokenizer_name not in sys.modules
    finally:
        sys.modules.pop(transformer_name, None)
        sys.modules.pop(tokenizer_name, None)
        for module_name, module in saved_modules.items():
            if module is not None:
                sys.modules[module_name] = module
        if nn_package is not None:
            if had_transformer_attr:
                nn_package.transformer = saved_transformer_attr
            elif hasattr(nn_package, "transformer"):
                del nn_package.transformer
