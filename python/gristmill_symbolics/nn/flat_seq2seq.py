from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from .transformer import (
    AttentionImplementation,
    TransformerDecoder,
    TransformerEncoder,
)

__all__ = ("FlatDefinitionSeq2SeqTransformer",)


class FlatDefinitionSeq2SeqTransformer(nnx.Module):
    def __init__(
        self,
        *,
        source_len: int,
        target_len: int,
        vocab_size: int,
        pad_token_id: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attention_implementation: AttentionImplementation = None,
        dtype: Any = jnp.bfloat16,
        param_dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.source_len = source_len
        self.target_len = target_len
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout
        self.attention_implementation = attention_implementation
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.token_embed = nnx.Embed(
            vocab_size,
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.source_position_embed = nnx.Embed(
            source_len,
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.target_position_embed = nnx.Embed(
            target_len,
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.embedding_dropout = nnx.Dropout(dropout, rngs=rngs)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            attention_implementation=attention_implementation,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            attention_implementation=attention_implementation,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.output_head = nnx.Linear(
            d_model,
            vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        source_ids: jax.Array,
        decoder_input_ids: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        source_ids = jnp.asarray(source_ids, dtype=jnp.int32)
        decoder_input_ids = jnp.asarray(decoder_input_ids, dtype=jnp.int32)
        source_mask = source_ids != self.pad_token_id
        target_mask = decoder_input_ids != self.pad_token_id

        source_vectors = self._embed(
            source_ids,
            self.source_position_embed,
            deterministic=deterministic,
        )
        target_vectors = self._embed(
            decoder_input_ids,
            self.target_position_embed,
            deterministic=deterministic,
        )

        memory = self.encoder(
            source_vectors,
            source_mask,
            deterministic=deterministic,
        )
        decoded = self.decoder(
            target_vectors,
            memory,
            target_mask=target_mask,
            source_mask=source_mask,
            deterministic=deterministic,
        )
        decoded = jnp.where(target_mask[..., None], decoded, 0.0)
        return self.output_head(decoded)

    def _embed(
        self,
        ids: jax.Array,
        position_embed: nnx.Embed,
        *,
        deterministic: bool,
    ) -> jax.Array:
        ids = jnp.asarray(ids, dtype=jnp.int32)
        length = ids.shape[-1]
        positions = jnp.arange(length, dtype=jnp.int32)
        x = self.token_embed(ids) + position_embed(positions)[None, :, :]
        x = jnp.where(ids[..., None] == self.pad_token_id, 0.0, x)
        return self.embedding_dropout(x, deterministic=deterministic)
