from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from transformer_policy.embed import TokenEmbedder, token_features
from transformer_policy.types import Token


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


class TransformerBlock(nnx.Module):
    def __init__(self, *, hidden_dim: int, num_heads: int, mlp_dim: int, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            rngs=rngs,
        )
        self.ln_2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.mlp_1 = nnx.Linear(hidden_dim, mlp_dim, rngs=rngs)
        self.mlp_2 = nnx.Linear(mlp_dim, hidden_dim, rngs=rngs)

    def __call__(self, values: jax.Array, mask: jax.Array) -> jax.Array:
        attended = self.attn(
            self.ln_1(values), mask=mask, deterministic=True, decode=False
        )
        values = values + attended
        mlp = self.mlp_2(nnx.gelu(self.mlp_1(self.ln_2(values))))
        return values + mlp


class CausalTransformerScorer(nnx.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        mlp_dim: int = 64,
        rngs: nnx.Rngs,
    ):
        _validate_positive("hidden_dim", hidden_dim)
        _validate_positive("num_heads", num_heads)
        _validate_positive("num_layers", num_layers)
        _validate_positive("mlp_dim", mlp_dim)
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.embedder = TokenEmbedder(hidden_dim=hidden_dim, rngs=rngs)
        self.blocks = nnx.List(
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                rngs=rngs,
            )
            for _ in range(num_layers)
        )
        self.final_ln = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.query = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

    def _encode(self, tokens: tuple[Token, ...]) -> jax.Array:
        if not tokens:
            raise ValueError("context plus prefix must not be empty")
        values = self.embedder(tokens)[None, :, :]
        token_mask = jnp.ones((1, len(tokens)), dtype=bool)
        causal_mask = nnx.make_causal_mask(token_mask)
        for block in self.blocks:
            values = block(values, causal_mask)
        return self.final_ln(values[0])

    def _encode_features(
        self,
        sequence_features: jax.Array,
        sequence_mask: jax.Array,
    ) -> jax.Array:
        if sequence_features.ndim != 2:
            raise ValueError("sequence_features must be a 2-D matrix")
        if sequence_mask.ndim != 1 or sequence_mask.shape[0] != sequence_features.shape[0]:
            raise ValueError("sequence_mask must match sequence_features length")
        values = self.embedder.proj(sequence_features)[None, :, :]
        token_mask = sequence_mask[None, :]
        causal_mask = nnx.make_causal_mask(token_mask)
        for block in self.blocks:
            values = block(values, causal_mask)
        encoded = self.final_ln(values[0])
        final_index = jnp.maximum(jnp.sum(sequence_mask.astype(jnp.int32)) - 1, 0)
        return encoded[final_index]

    def _embed_legal_tokens(
        self, tokens: tuple[Token, ...], next_position: int
    ) -> jax.Array:
        features = jnp.asarray(token_features(tokens), dtype=jnp.float32)
        features = features.at[:, 1].set(float(next_position))
        return self.embedder.proj(features)

    def score_next(
        self,
        context_tokens: tuple[Token, ...],
        decision_prefix: tuple[Token, ...],
        legal_next_tokens: tuple[Token, ...],
    ) -> jax.Array:
        if not legal_next_tokens:
            raise ValueError("legal_next_tokens must not be empty")
        sequence_tokens = (*context_tokens, *decision_prefix)
        hidden = self._encode(sequence_tokens)[-1]
        query = self.query(hidden)
        legal_embeddings = self._embed_legal_tokens(
            legal_next_tokens, next_position=len(sequence_tokens)
        )
        return jnp.matmul(legal_embeddings, query)

    def score_next_features(
        self,
        sequence_features: jax.Array,
        sequence_mask: jax.Array,
        legal_features: jax.Array,
        legal_mask: jax.Array,
    ) -> jax.Array:
        hidden = self._encode_features(sequence_features, sequence_mask)
        query = self.query(hidden)
        legal_embeddings = self.embedder.proj(legal_features)
        logits = jnp.matmul(legal_embeddings, query)
        return jnp.where(legal_mask, logits, -jnp.inf)
