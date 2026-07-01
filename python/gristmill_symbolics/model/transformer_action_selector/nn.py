from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from gristmill_symbolics.model.tokenizer import SEGMENT, SIDE, SYM_ACTION
from gristmill_symbolics.model.tokenizer import SENTINEL, TOKEN_KIND
from gristmill_symbolics.model.tokenizer import TokenArrays

ID_FIELDS = (
    "def_index",
    "term_index",
    "factor_index",
    "tensor_id",
    "range_id",
    "index_id",
    "candidate_index",
    "symmetry_index",
    "perm_index",
    "perm_value",
)

ENUM_FIELDS = (
    "segment",
    "side",
    "symmetry_action",
)

_ENUM_TABLE_SIZES = {
    "token_kind": int(max(TOKEN_KIND)) + 1,
    "segment": int(max(SEGMENT)) + 1,
    "side": int(max(SIDE)) + 1,
    "symmetry_action": int(max(SYM_ACTION)) + 1,
}


class TokenEmbedder(nn.Module):
    d_model: int
    id_vocab_size: int
    init_scale: float = 0.02
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, tokens: TokenArrays):
        shape = tokens["token_kind"].shape
        out = jnp.zeros((*shape, self.d_model), dtype=self.dtype)
        out = out + self._embed_field("token_kind", tokens, shape)
        for field in ENUM_FIELDS:
            out = out + self._embed_field(field, tokens, shape)
        for field in ID_FIELDS:
            out = out + self._embed_field(field, tokens, shape)
        return out + self._embed_numeric(tokens, shape)

    def _embed_field(self, field: str, tokens: TokenArrays, shape):
        rows = self._embedding_rows(field)
        table = self.param(
            f"{field}_embedding",
            self._zero_head_normal(),
            (rows, self.d_model),
            jnp.float32,
        )
        values = self._field(tokens, field, shape)
        if field == "token_kind":
            index = self._token_kind_index(values, rows)
        else:
            index = self._shifted_index(values, rows)
        gathered = jnp.take(table, index, axis=0)
        keep = (index != 0).astype(gathered.dtype)
        return jnp.asarray(gathered * keep[..., None], self.dtype)

    def _embed_numeric(self, tokens: TokenArrays, shape):
        return nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(self.init_scale),
            name="numeric_projection",
        )(self._numeric_features(tokens, shape))

    def _embedding_rows(self, field: str) -> int:
        size = _ENUM_TABLE_SIZES.get(field, self.id_vocab_size)
        if field == "token_kind":
            return size
        return size + 1

    def _field(self, tokens: TokenArrays, name: str, shape):
        if name == "token_kind":
            return tokens[name]
        return tokens.get(name, jnp.full(shape, SENTINEL, dtype=jnp.int32))

    def _numeric_features(self, tokens: TokenArrays, shape):
        coeff_num = self._field(tokens, "coeff_num", shape)
        coeff_den = self._field(tokens, "coeff_den", shape)
        position = self._field(tokens, "position", shape)
        coeff_present = coeff_den >= 0
        return jnp.stack(
            [
                jnp.where(coeff_present, coeff_num.astype(jnp.float32), 0.0),
                jnp.where(coeff_present, coeff_den.astype(jnp.float32), 0.0),
                jnp.where(position < 0, 0.0, position.astype(jnp.float32)),
            ],
            axis=-1,
        )

    def _token_kind_index(self, values, rows: int):
        return jnp.where(values <= 0, 0, jnp.minimum(values, rows - 1))

    def _shifted_index(self, values, rows: int):
        size = rows - 1
        return jnp.where(values < 0, 0, (values % size) + 1)

    def _zero_head_normal(self):
        normal = nn.initializers.normal(self.init_scale)

        def init(key, shape, dtype=jnp.float32):
            values = normal(key, shape, dtype)
            return values.at[0].set(0)

        return init


class TransformerEncoder(nn.Module):
    d_model: int
    num_layers: int
    num_heads: int
    init_scale: float = 0.02
    dtype: Any = jnp.bfloat16
    prefer_cudnn: bool = True

    @nn.compact
    def __call__(self, vectors, mask):
        out = self._apply_mask(jnp.asarray(vectors, self.dtype), mask)
        for index in range(self.num_layers):
            attended = self._self_attention(
                self._norm(f"attention_norm_{index}", out),
                mask,
                index,
            )
            out = self._apply_mask(out + attended, mask)
            hidden = self._mlp(self._norm(f"mlp_norm_{index}", out), index)
            out = self._apply_mask(out + hidden, mask)
        return out

    def _self_attention(self, x, mask, index: int):
        single = x.ndim == 2
        if single:
            x = x[None]
            mask = mask[None]
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            use_bias=False,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(self.init_scale),
            out_kernel_init=nn.initializers.normal(self.init_scale),
            attention_fn=self._dot_product_attention,
            name=f"attention_{index}",
        )(
            x,
            mask=self._attention_mask(mask),
            deterministic=True,
        )
        if single:
            return attended[0]
        return attended

    def _mlp(self, x, index: int):
        hidden = nn.Dense(
            2 * self.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(self.init_scale),
            name=f"mlp_in_{index}",
        )(x)
        hidden = nn.gelu(hidden)
        return nn.Dense(
            self.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(self.init_scale),
            name=f"mlp_out_{index}",
        )(hidden)

    def _apply_mask(self, values, mask):
        return jnp.where(mask[..., None], values, 0.0)

    def _dot_product_attention(
        self,
        query,
        key,
        value,
        bias=None,
        mask=None,
        **_,
    ):
        return jax.nn.dot_product_attention(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            implementation=self._attention_implementation(),
        )

    def _attention_mask(self, mask):
        return nn.make_attention_mask(mask, mask, dtype=jnp.bool_)

    def _attention_implementation(self):
        if self.prefer_cudnn and jax.default_backend() == "gpu":
            return "cudnn"
        return None

    def _norm(self, name: str, x):
        return nn.LayerNorm(
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name=name,
        )(x)


class LogitDecoder(nn.Module):
    d_model: int
    output_size: int
    init_scale: float = 0.02
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, encoded_tokens, token_mask):
        context = self._masked_mean(encoded_tokens, token_mask)
        hidden = nn.Dense(
            2 * self.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(self.init_scale),
            name="hidden",
        )(context)
        hidden = nn.gelu(hidden)
        return nn.Dense(
            self.output_size,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(self.init_scale),
            name="logits",
        )(hidden)

    def _masked_mean(self, encoded_tokens, token_mask):
        weights = token_mask.astype(encoded_tokens.dtype)
        total = jnp.sum(encoded_tokens * weights[..., None], axis=-2)
        count = jnp.sum(weights, axis=-1, keepdims=True)
        return total / jnp.maximum(count, 1.0)
