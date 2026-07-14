from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.attention import dot_product_attention as _flax_dot_product_attention

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

    def init_decode_cache(self, *, batch_size: int, target_len: int) -> None:
        for layer in self.layers:
            layer.init_decode_cache(batch_size=batch_size, target_len=target_len)

    def decode_step(
        self,
        x_t: jax.Array,
        memory: jax.Array,
        *,
        source_mask: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        for layer in self.layers:
            x_t = layer.decode_step(
                x_t,
                memory,
                source_mask=source_mask,
                deterministic=deterministic,
            )
        return self.final_norm(x_t)


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

    def init_decode_cache(self, *, batch_size: int, target_len: int) -> None:
        self.self_attention.init_decode_cache(
            batch_size=batch_size,
            target_len=target_len,
        )

    def decode_step(
        self,
        x_t: jax.Array,
        memory: jax.Array,
        *,
        source_mask: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        x_t = x_t + self.self_attention.decode_step(
            self.self_attention_norm(x_t),
            deterministic=deterministic,
        )
        x_t = x_t + self.cross_attention(
            self.cross_attention_norm(x_t),
            memory,
            mask=source_mask,
            deterministic=deterministic,
        )
        return x_t + self.feed_forward(
            self.feed_forward_norm(x_t),
            deterministic=deterministic,
        )


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
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            qkv_features=d_model,
            out_features=d_model,
            dropout_rate=dropout,
            attention_fn=_attention_fn(attention_implementation),
            decode=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        *,
        mask: jax.Array | None = None,
        is_causal: bool = False,
        deterministic: bool = True,
    ) -> jax.Array:
        attention_mask = _attention_mask(
            mask,
            batch_size=x.shape[0],
            query_len=x.shape[1],
            key_len=memory.shape[1],
            is_causal=is_causal,
        )
        return self.attention(
            x,
            memory,
            mask=attention_mask,
            deterministic=deterministic,
            decode=False,
        )

    def init_decode_cache(self, *, batch_size: int, target_len: int) -> None:
        self.attention.init_cache((batch_size, target_len, self.d_model))

    def decode_step(
        self,
        x_t: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        return self.attention(
            x_t,
            deterministic=deterministic,
            decode=True,
        )


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


def _attention_mask(
    mask: jax.Array | None,
    *,
    batch_size: int,
    query_len: int,
    key_len: int,
    is_causal: bool,
) -> jax.Array | None:
    attention_mask = None
    if mask is not None:
        attention_mask = jnp.broadcast_to(
            mask[:, None, None, :],
            (batch_size, 1, query_len, key_len),
        )
    if is_causal:
        causal_mask = jnp.tril(jnp.ones((query_len, key_len), dtype=bool))[
            None,
            None,
            :,
            :,
        ]
        attention_mask = (
            causal_mask if attention_mask is None else attention_mask & causal_mask
        )
    return attention_mask


def _attention_fn(implementation: AttentionImplementation):
    def attention(
        query,
        key,
        value,
        *,
        bias=None,
        mask=None,
        dropout_rng=None,
        dropout_rate=0.0,
        broadcast_dropout=True,
        deterministic=True,
        dtype=None,
        precision=None,
        module=None,
        **_kwargs,
    ):
        if mask is not None:
            mask = mask.astype(bool)
        if dropout_rate == 0.0 and module is None:
            return jax.nn.dot_product_attention(
                query,
                key,
                value,
                bias=bias,
                mask=mask,
                implementation=implementation,
            )
        return _flax_dot_product_attention(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=dropout_rate,
            broadcast_dropout=broadcast_dropout,
            deterministic=deterministic,
            dtype=dtype,
            precision=precision,
            module=module,
        )

    return attention
