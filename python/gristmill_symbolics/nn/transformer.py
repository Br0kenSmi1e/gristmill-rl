from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx

AttentionImplementation = Literal["xla", "cudnn"] | None

__all__ = (
    "DecoderBlock",
    "EncoderBlock",
    "TransformerDecoder",
    "TransformerEncoder",
)


class TransformerEncoder(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attention_implementation: AttentionImplementation = None,
        dtype: Any = jnp.bfloat16,
        param_dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim or 4 * d_model
        self.dropout = dropout
        self.attention_implementation = attention_implementation
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layers = nnx.List(
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                dropout=dropout,
                attention_implementation=attention_implementation,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nnx.LayerNorm(
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        source_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        for layer in self.layers:
            x = layer(x, source_mask, deterministic=deterministic)
        x = self.final_norm(x)
        return _zero_masked_queries(x, source_mask)


class TransformerDecoder(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attention_implementation: AttentionImplementation = None,
        dtype: Any = jnp.bfloat16,
        param_dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim or 4 * d_model
        self.dropout = dropout
        self.attention_implementation = attention_implementation
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layers = nnx.List(
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                dropout=dropout,
                attention_implementation=attention_implementation,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nnx.LayerNorm(
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        *,
        target_mask: jax.Array | None = None,
        source_mask: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                target_mask=target_mask,
                source_mask=source_mask,
                deterministic=deterministic,
            )
        x = self.final_norm(x)
        return _zero_masked_queries(x, target_mask)


class EncoderBlock(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attention_implementation: AttentionImplementation = None,
        dtype: Any = jnp.bfloat16,
        param_dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.self_attention_norm = nnx.LayerNorm(
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.self_attention = _MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_implementation=attention_implementation,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.feed_forward_norm = nnx.LayerNorm(
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.feed_forward = _FeedForward(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim or 4 * d_model,
            dropout=dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        source_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        normalized = self.self_attention_norm(x)
        x = x + self.self_attention(
            normalized,
            normalized,
            mask=source_mask,
            deterministic=deterministic,
        )
        x = x + self.feed_forward(
            self.feed_forward_norm(x),
            deterministic=deterministic,
        )
        return _zero_masked_queries(x, source_mask)


class DecoderBlock(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attention_implementation: AttentionImplementation = None,
        dtype: Any = jnp.bfloat16,
        param_dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.self_attention_norm = nnx.LayerNorm(
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.self_attention = _MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_implementation=attention_implementation,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.cross_attention_norm = nnx.LayerNorm(
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.cross_attention = _MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_implementation=attention_implementation,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.feed_forward_norm = nnx.LayerNorm(
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.feed_forward = _FeedForward(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim or 4 * d_model,
            dropout=dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        *,
        target_mask: jax.Array | None = None,
        source_mask: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        normalized = self.self_attention_norm(x)
        x = x + self.self_attention(
            normalized,
            normalized,
            mask=target_mask,
            is_causal=True,
            deterministic=deterministic,
        )
        x = _zero_masked_queries(x, target_mask)
        x = x + self.cross_attention(
            self.cross_attention_norm(x),
            memory,
            mask=source_mask,
            deterministic=deterministic,
        )
        x = x + self.feed_forward(
            self.feed_forward_norm(x),
            deterministic=deterministic,
        )
        return _zero_masked_queries(x, target_mask)


class _MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dropout: float,
        attention_implementation: AttentionImplementation,
        dtype: Any,
        param_dtype: Any,
        rngs: nnx.Rngs,
    ):
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_implementation = attention_implementation
        self.query = nnx.Linear(
            d_model,
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.key = nnx.Linear(
            d_model,
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.value = nnx.Linear(
            d_model,
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            d_model,
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        *,
        mask: jax.Array | None = None,
        is_causal: bool = False,
        deterministic: bool = True,
    ) -> jax.Array:
        query = self._split_heads(self.query(x))
        key = self._split_heads(self.key(memory))
        value = self._split_heads(self.value(memory))
        attention_mask = (
            jnp.broadcast_to(
                mask[:, None, None, :],
                (query.shape[0], query.shape[2], query.shape[1], key.shape[1]),
            )
            if mask is not None
            else None
        )
        attended = jax.nn.dot_product_attention(
            query,
            key,
            value,
            mask=attention_mask,
            is_causal=is_causal,
            implementation=self.attention_implementation,
        )
        output = self.output(self._merge_heads(attended))
        return self.dropout(output, deterministic=deterministic)

    def _split_heads(self, x: jax.Array) -> jax.Array:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_dim)

    def _merge_heads(self, x: jax.Array) -> jax.Array:
        batch, length, _, _ = x.shape
        return x.reshape(batch, length, self.d_model)


class _FeedForward(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        dtype: Any,
        param_dtype: Any,
        rngs: nnx.Rngs,
    ):
        self.input = nnx.Linear(
            d_model,
            hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            hidden_dim,
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        x = self.input(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.output(x)
        return self.dropout(x, deterministic=deterministic)


def _zero_masked_queries(x: jax.Array, mask: jax.Array | None) -> jax.Array:
    if mask is None:
        return x
    return jnp.where(mask[..., None], x, 0.0)
