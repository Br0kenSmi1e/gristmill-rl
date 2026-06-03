from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from transformer_policy.embed import TokenEmbedder
from transformer_policy.types import Token


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

    def score_next(
        self,
        context_tokens: tuple[Token, ...],
        decision_prefix: tuple[Token, ...],
        legal_next_tokens: tuple[Token, ...],
    ) -> jax.Array:
        if not legal_next_tokens:
            raise ValueError("legal_next_tokens must not be empty")
        hidden = self._encode((*context_tokens, *decision_prefix))[-1]
        query = self.query(hidden)
        legal_embeddings = self.embedder(legal_next_tokens)
        return jnp.matmul(legal_embeddings, query)
