from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from .tokens import (
    KIND,
    KEYWORD,
    SCALAR_TYPE,
    SENTINEL,
    TOKEN_FIELDS,
    validate_scalar_bounds,
)


def make_decoder_inputs(
    target_tokens: Mapping[str, Any],
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array]:
    target = _token_row(target_tokens)
    target_len = target["kind"].shape[0]
    real_length = int(jnp.sum(target["mask"]))
    if real_length < 2:
        raise ValueError("encoded target row must contain at least BOS and EOS")
    if int(target["kind"][0]) != KIND["BOS"]:
        raise ValueError("encoded target row must start with BOS")
    if int(target["kind"][real_length - 1]) != KIND["EOS"]:
        raise ValueError("encoded target row must end with EOS")

    decoder_input = _pad_like(target_len)
    labels = _pad_like(target_len)
    label_length = real_length - 1

    for field in ("kind", "keyword", "scalar_type", "scalar_value"):
        decoder_input[field] = decoder_input[field].at[:label_length].set(
            target[field][:label_length]
        )
        labels[field] = labels[field].at[:label_length].set(
            target[field][1:real_length]
        )
    decoder_input["mask"] = decoder_input["mask"].at[:label_length].set(True)
    labels["mask"] = labels["mask"].at[:label_length].set(True)

    return decoder_input, labels, labels["mask"]


def token_log_probs(
    logits: Mapping[str, Any],
    target_tokens: Mapping[str, Any],
) -> jax.Array:
    target = _token_arrays(target_tokens)
    scalar_value_logits = jnp.asarray(logits["scalar_value"])
    scalar_value_min = jnp.asarray(logits["scalar_value_min"])
    if not _contains_tracer((target, scalar_value_logits, scalar_value_min)):
        concrete_scalar_value_min = int(scalar_value_min)
        validate_scalar_bounds(
            target_tokens,
            scalar_value_min=concrete_scalar_value_min,
            scalar_value_max=(
                concrete_scalar_value_min + scalar_value_logits.shape[-1] - 1
            ),
        )

    active_mask = target["mask"].astype(bool)
    kind = target["kind"]
    token_scores = jnp.where(
        active_mask,
        _take_log_probs(
            jnp.asarray(logits["kind"]),
            _safe_index(kind, active_mask),
        ),
        0.0,
    )

    keyword_mask = active_mask & (kind == KIND["KEYWORD"])
    keyword_scores = _take_log_probs(
        jnp.asarray(logits["keyword"]),
        _safe_index(target["keyword"], keyword_mask),
    )
    token_scores = token_scores + jnp.where(keyword_mask, keyword_scores, 0.0)

    scalar_mask = active_mask & (kind == KIND["SCALAR"])
    scalar_type_scores = _take_log_probs(
        jnp.asarray(logits["scalar_type"]),
        _safe_index(target["scalar_type"], scalar_mask),
    )
    scalar_value_index = target["scalar_value"] - scalar_value_min
    scalar_value_scores = _take_log_probs(
        scalar_value_logits,
        _safe_index(scalar_value_index, scalar_mask),
    )
    token_scores = token_scores + jnp.where(
        scalar_mask,
        scalar_type_scores + scalar_value_scores,
        0.0,
    )

    return token_scores


def sequence_log_prob(
    logits: Mapping[str, Any],
    target_tokens: Mapping[str, Any],
    target_mask: jax.Array,
) -> jax.Array:
    return jnp.sum(token_log_probs(logits, target_tokens) * target_mask, axis=1)


def sample_tokens(
    model: "DirectOptimizerTransformer",
    rng,
    source_tokens,
    *,
    max_length: int,
    temperature: float,
    mask_provider=None,
) -> tuple[dict[str, jax.Array], jax.Array]:
    if max_length <= 0 or max_length > model.target_len:
        raise ValueError("max_length must satisfy 0 < max_length <= model.target_len")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    source = _token_batch(source_tokens)
    _validate_token_length(source, model.source_len, "source_tokens")
    batch_size = source["kind"].shape[0]

    prefix = _sample_pad_batch(batch_size, model.target_len)
    prefix["kind"] = prefix["kind"].at[:, 0].set(KIND["BOS"])
    prefix["mask"] = prefix["mask"].at[:, 0].set(True)
    generated = _sample_pad_batch(batch_size, model.target_len)
    alive = jnp.ones((batch_size,), dtype=bool)

    def scan_step(carry, step):
        rng_key, decoder_prefix, output_tokens, rows_alive = carry
        rng_key, kind_key, keyword_key, scalar_type_key, scalar_value_key = (
            jax.random.split(rng_key, 5)
        )
        logits = model(source, decoder_prefix, deterministic=True)
        step_logits = {
            field: jnp.asarray(logits[field])[:, step, :] / temperature
            for field in ("kind", "keyword", "scalar_type", "scalar_value")
        }
        if mask_provider is not None:
            step_logits = _apply_sampling_masks(
                step_logits,
                mask_provider(decoder_prefix, step, source),
            )

        sampled_kind = jax.random.categorical(kind_key, step_logits["kind"]).astype(
            jnp.int32
        )
        sampled_keyword = jax.random.categorical(
            keyword_key,
            step_logits["keyword"],
        ).astype(jnp.int32)
        sampled_scalar_type = jax.random.categorical(
            scalar_type_key,
            step_logits["scalar_type"],
        ).astype(jnp.int32)
        sampled_scalar_value = jax.random.categorical(
            scalar_value_key,
            step_logits["scalar_value"],
        ).astype(jnp.int32)
        sampled_scalar_value = sampled_scalar_value + model.scalar_value_min

        active = rows_alive
        kind = jnp.where(active, sampled_kind, KIND["PAD"])
        keyword = jnp.where(
            active & (kind == KIND["KEYWORD"]),
            sampled_keyword,
            SENTINEL,
        )
        scalar_type = jnp.where(
            active & (kind == KIND["SCALAR"]),
            sampled_scalar_type,
            SENTINEL,
        )
        scalar_value = jnp.where(
            active & (kind == KIND["SCALAR"]),
            sampled_scalar_value,
            SENTINEL,
        )
        token_mask = active

        output_tokens = _set_sample_position(
            output_tokens,
            step,
            kind=kind,
            keyword=keyword,
            scalar_type=scalar_type,
            scalar_value=scalar_value,
            mask=token_mask,
        )

        next_position = step + 1
        next_alive = rows_alive & (kind != KIND["EOS"])
        prefix_mask = active & next_alive
        decoder_prefix = _set_sample_position(
            decoder_prefix,
            next_position,
            kind=kind,
            keyword=keyword,
            scalar_type=scalar_type,
            scalar_value=scalar_value,
            mask=prefix_mask,
        )
        return (rng_key, decoder_prefix, output_tokens, next_alive), None

    (_rng, _prefix, generated, _alive), _ = jax.lax.scan(
        scan_step,
        (jnp.asarray(rng), prefix, generated, alive),
        jnp.arange(max_length, dtype=jnp.int32),
    )
    return generated, generated["mask"]


def _token_arrays(tokens: Mapping[str, Any]) -> dict[str, jax.Array]:
    missing = [field for field in TOKEN_FIELDS if field not in tokens]
    if missing:
        raise ValueError(f"missing token fields: {missing}")
    arrays = {field: jnp.asarray(tokens[field]) for field in TOKEN_FIELDS}
    shape = arrays["kind"].shape
    for field, array in arrays.items():
        if array.shape != shape:
            raise ValueError(
                f"expected {field} shape {array.shape} to match kind shape {shape}"
            )
    return arrays


def _token_row(tokens: Mapping[str, Any]) -> dict[str, jax.Array]:
    arrays = _token_arrays(tokens)
    for field, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f"expected {field} to be a 1D encoded token row")
    return arrays


def _sample_pad_batch(batch_size: int, length: int) -> dict[str, jax.Array]:
    return {
        "kind": jnp.full((batch_size, length), KIND["PAD"], dtype=jnp.int32),
        "keyword": jnp.full((batch_size, length), SENTINEL, dtype=jnp.int32),
        "scalar_type": jnp.full((batch_size, length), SENTINEL, dtype=jnp.int32),
        "scalar_value": jnp.full((batch_size, length), SENTINEL, dtype=jnp.int32),
        "mask": jnp.zeros((batch_size, length), dtype=bool),
    }


def _set_sample_position(
    tokens: Mapping[str, jax.Array],
    position: jax.Array,
    *,
    kind: jax.Array,
    keyword: jax.Array,
    scalar_type: jax.Array,
    scalar_value: jax.Array,
    mask: jax.Array,
) -> dict[str, jax.Array]:
    updated = dict(tokens)
    updated["kind"] = updated["kind"].at[:, position].set(kind)
    updated["keyword"] = updated["keyword"].at[:, position].set(keyword)
    updated["scalar_type"] = updated["scalar_type"].at[:, position].set(scalar_type)
    updated["scalar_value"] = updated["scalar_value"].at[:, position].set(scalar_value)
    updated["mask"] = updated["mask"].at[:, position].set(mask)
    return updated


def _apply_sampling_masks(
    logits: Mapping[str, jax.Array],
    masks,
) -> dict[str, jax.Array]:
    if masks is None:
        return dict(logits)
    masked = dict(logits)
    for field in ("kind", "keyword", "scalar_type", "scalar_value"):
        additive_mask = _optional_mask_field(masks, field)
        if additive_mask is not None:
            masked[field] = masked[field] + jnp.asarray(additive_mask)
    return masked


def _optional_mask_field(masks, field: str):
    if isinstance(masks, Mapping):
        return masks.get(field)
    return getattr(masks, field, None)


def _pad_like(length: int) -> dict[str, jax.Array]:
    return {
        "kind": jnp.full((length,), KIND["PAD"], dtype=jnp.int32),
        "keyword": jnp.full((length,), SENTINEL, dtype=jnp.int32),
        "scalar_type": jnp.full((length,), SENTINEL, dtype=jnp.int32),
        "scalar_value": jnp.full((length,), SENTINEL, dtype=jnp.int32),
        "mask": jnp.zeros((length,), dtype=bool),
    }


def _take_log_probs(logits: jax.Array, indices: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, indices[..., None], axis=-1)[..., 0]


def _safe_index(indices: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.where(mask, indices, 0)


def _contains_tracer(value: Any) -> bool:
    return any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(value)
    )


_NNX_MODULE = nnx.Module
_TOKEN_HEAD_SIZES = (len(KIND), len(KEYWORD), len(SCALAR_TYPE))
_CUDNN_ATTENTION_DTYPE = jnp.bfloat16


def _attention_implementation() -> str | None:
    if jax.default_backend() == "gpu":
        return "cudnn"
    return None


def _sequence_lengths(mask: jax.Array) -> jax.Array:
    return jnp.atleast_1d(jnp.sum(mask, axis=-1, dtype=jnp.int32))


def _attention_mask(
    query_mask: jax.Array,
    key_value_mask: jax.Array,
) -> jax.Array | None:
    if _attention_implementation() == "cudnn":
        return None
    return nnx.make_attention_mask(query_mask, key_value_mask, dtype=bool)


def _multi_head_attention(
    attention,
    inputs_q: jax.Array,
    inputs_k: jax.Array | None = None,
    inputs_v: jax.Array | None = None,
    *,
    query_mask: jax.Array,
    key_value_mask: jax.Array,
    mask: jax.Array | None,
    is_causal: bool,
    deterministic: bool,
) -> jax.Array:
    if _attention_implementation() != "cudnn":
        return attention(
            inputs_q,
            inputs_k,
            inputs_v,
            mask=mask,
            deterministic=deterministic,
        )

    if inputs_k is None:
        if inputs_v is not None:
            raise ValueError("inputs_k cannot be None when inputs_v is provided")
        inputs_k = inputs_q
    if inputs_v is None:
        inputs_v = inputs_k
    if attention.dropout_rate > 0.0 and not deterministic:
        raise ValueError(
            "cuDNN attention does not support attention dropout; use dropout=0.0"
        )

    query = attention.query(inputs_q)
    key = attention.key(inputs_k)
    value = attention.value(inputs_v)
    if attention.normalize_qk:
        if attention.query_ln is None or attention.key_ln is None:
            raise ValueError("attention qk normalization is not initialized")
        query = attention.query_ln(query)
        key = attention.key_ln(key)

    attended = jax.nn.dot_product_attention(
        jnp.asarray(query, _CUDNN_ATTENTION_DTYPE),
        jnp.asarray(key, _CUDNN_ATTENTION_DTYPE),
        jnp.asarray(value, _CUDNN_ATTENTION_DTYPE),
        bias=None,
        mask=None,
        is_causal=is_causal,
        query_seq_lengths=_sequence_lengths(query_mask),
        key_value_seq_lengths=_sequence_lengths(key_value_mask),
        implementation="cudnn",
    )
    return attention.out(attended)


class FeedForward(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        kernel_init,
        rngs: nnx.Rngs,
    ):
        self.input = nnx.Linear(
            d_model,
            hidden_dim,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            hidden_dim,
            d_model,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        x = self.input(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.output(x)
        return self.dropout(x, deterministic=deterministic)


class EncoderLayer(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dropout: float,
        kernel_init,
        rngs: nnx.Rngs,
    ):
        self.self_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            dropout_rate=dropout,
            decode=False,
            kernel_init=kernel_init,
            out_kernel_init=kernel_init,
            rngs=rngs,
        )
        self.feed_forward = FeedForward(
            d_model=d_model,
            hidden_dim=4 * d_model,
            dropout=dropout,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.attention_dropout = nnx.Dropout(dropout, rngs=rngs)
        self.attention_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.feed_forward_norm = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        source_mask: jax.Array,
        deterministic: bool,
    ) -> jax.Array:
        attention = _multi_head_attention(
            self.self_attention,
            x,
            query_mask=source_mask,
            key_value_mask=source_mask,
            mask=_attention_mask(source_mask, source_mask),
            is_causal=False,
            deterministic=deterministic,
        )
        x = self.attention_norm(
            x + self.attention_dropout(attention, deterministic=deterministic)
        )
        x = _mask_sequence(x, source_mask)
        x = self.feed_forward_norm(
            x + self.feed_forward(x, deterministic=deterministic)
        )
        return _mask_sequence(x, source_mask)


class DecoderLayer(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dropout: float,
        kernel_init,
        rngs: nnx.Rngs,
    ):
        self.self_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            dropout_rate=dropout,
            decode=False,
            kernel_init=kernel_init,
            out_kernel_init=kernel_init,
            rngs=rngs,
        )
        self.cross_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            dropout_rate=dropout,
            decode=False,
            kernel_init=kernel_init,
            out_kernel_init=kernel_init,
            rngs=rngs,
        )
        self.feed_forward = FeedForward(
            d_model=d_model,
            hidden_dim=4 * d_model,
            dropout=dropout,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.self_attention_dropout = nnx.Dropout(dropout, rngs=rngs)
        self.cross_attention_dropout = nnx.Dropout(dropout, rngs=rngs)
        self.self_attention_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.cross_attention_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.feed_forward_norm = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        source_memory: jax.Array,
        *,
        target_mask: jax.Array,
        source_mask: jax.Array,
        deterministic: bool,
    ) -> jax.Array:
        self_attention_mask = None
        if _attention_implementation() != "cudnn":
            target_attention_mask = nnx.make_attention_mask(
                target_mask,
                target_mask,
                dtype=bool,
            )
            causal_mask = nnx.make_causal_mask(target_mask, dtype=bool)
            self_attention_mask = nnx.combine_masks(
                target_attention_mask,
                causal_mask,
                dtype=bool,
            )
        self_attention = _multi_head_attention(
            self.self_attention,
            x,
            query_mask=target_mask,
            key_value_mask=target_mask,
            mask=self_attention_mask,
            is_causal=True,
            deterministic=deterministic,
        )
        x = self.self_attention_norm(
            x
            + self.self_attention_dropout(
                self_attention,
                deterministic=deterministic,
            )
        )
        x = _mask_sequence(x, target_mask)

        cross_attention = _multi_head_attention(
            self.cross_attention,
            x,
            source_memory,
            query_mask=target_mask,
            key_value_mask=source_mask,
            mask=_attention_mask(target_mask, source_mask),
            is_causal=False,
            deterministic=deterministic,
        )
        x = self.cross_attention_norm(
            x
            + self.cross_attention_dropout(
                cross_attention,
                deterministic=deterministic,
            )
        )
        x = _mask_sequence(x, target_mask)
        x = self.feed_forward_norm(
            x + self.feed_forward(x, deterministic=deterministic)
        )
        return _mask_sequence(x, target_mask)


class DirectOptimizerTransformer(nnx.Module):
    def __init__(
        self,
        *,
        source_len: int,
        target_len: int,
        scalar_value_min: int,
        scalar_value_max: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        init_scale: float = 0.02,
        rngs: nnx.Rngs,
    ):
        _validate_positive("source_len", source_len)
        _validate_positive("target_len", target_len)
        _validate_positive("d_model", d_model)
        _validate_positive("num_layers", num_layers)
        _validate_positive("num_heads", num_heads)
        if scalar_value_min > scalar_value_max:
            raise ValueError("scalar_value_min must be <= scalar_value_max")

        self.source_len = int(source_len)
        self.target_len = int(target_len)
        self.scalar_value_min = int(scalar_value_min)
        self.scalar_value_max = int(scalar_value_max)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.init_scale = float(init_scale)
        self.scalar_value_count = self.scalar_value_max - self.scalar_value_min + 1

        kernel_init = jax.nn.initializers.normal(self.init_scale)
        self.kind_embed = nnx.Embed(
            len(KIND),
            self.d_model,
            embedding_init=kernel_init,
            rngs=rngs,
        )
        self.keyword_embed = nnx.Embed(
            len(KEYWORD),
            self.d_model,
            embedding_init=kernel_init,
            rngs=rngs,
        )
        self.scalar_type_embed = nnx.Embed(
            len(SCALAR_TYPE),
            self.d_model,
            embedding_init=kernel_init,
            rngs=rngs,
        )
        self.scalar_value_projection = nnx.Linear(
            1,
            self.d_model,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.source_position_embed = nnx.Embed(
            self.source_len,
            self.d_model,
            embedding_init=kernel_init,
            rngs=rngs,
        )
        self.target_position_embed = nnx.Embed(
            self.target_len,
            self.d_model,
            embedding_init=kernel_init,
            rngs=rngs,
        )
        self.embedding_dropout = nnx.Dropout(self.dropout, rngs=rngs)
        self.encoder_layers = nnx.List(
            EncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout=self.dropout,
                kernel_init=kernel_init,
                rngs=rngs,
            )
            for _ in range(self.num_layers)
        )
        self.decoder_layers = nnx.List(
            DecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout=self.dropout,
                kernel_init=kernel_init,
                rngs=rngs,
            )
            for _ in range(self.num_layers)
        )
        self.decoder_norm = nnx.LayerNorm(self.d_model, rngs=rngs)
        self.kind_head = nnx.Linear(
            self.d_model,
            len(KIND),
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.keyword_head = nnx.Linear(
            self.d_model,
            len(KEYWORD),
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.scalar_type_head = nnx.Linear(
            self.d_model,
            len(SCALAR_TYPE),
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.scalar_value_head = nnx.Linear(
            self.d_model,
            self.scalar_value_count,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def model_kwargs(self) -> dict[str, object]:
        return {
            "source_len": self.source_len,
            "target_len": self.target_len,
            "scalar_value_min": self.scalar_value_min,
            "scalar_value_max": self.scalar_value_max,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "init_scale": self.init_scale,
        }

    def embed_tokens(self, tokens, *, length: int) -> jax.Array:
        return self._embed_tokens(
            tokens,
            length=length,
            position_embed=self.source_position_embed,
            deterministic=True,
        )

    def __call__(
        self,
        source_tokens,
        decoder_input_tokens,
        *,
        deterministic: bool = True,
    ) -> dict[str, jax.Array | int]:
        source = _token_batch(source_tokens)
        decoder_input = _token_batch(decoder_input_tokens)
        _validate_token_length(source, self.source_len, "source_tokens")
        _validate_token_length(
            decoder_input,
            self.target_len,
            "decoder_input_tokens",
        )

        source_mask = source["mask"]
        target_mask = decoder_input["mask"]
        source_memory = self._embed_tokens(
            source,
            length=self.source_len,
            position_embed=self.source_position_embed,
            deterministic=deterministic,
        )
        for layer in self.encoder_layers:
            source_memory = layer(
                source_memory,
                source_mask=source_mask,
                deterministic=deterministic,
            )

        target = self._embed_tokens(
            decoder_input,
            length=self.target_len,
            position_embed=self.target_position_embed,
            deterministic=deterministic,
        )
        for layer in self.decoder_layers:
            target = layer(
                target,
                source_memory,
                target_mask=target_mask,
                source_mask=source_mask,
                deterministic=deterministic,
            )
        target = _mask_sequence(self.decoder_norm(target), target_mask)

        return {
            "kind": self.kind_head(target),
            "keyword": self.keyword_head(target),
            "scalar_type": self.scalar_type_head(target),
            "scalar_value": self.scalar_value_head(target),
            "scalar_value_min": self.scalar_value_min,
        }

    def _embed_tokens(
        self,
        tokens,
        *,
        length: int,
        position_embed: nnx.Embed,
        deterministic: bool,
    ) -> jax.Array:
        arrays = _token_batch(tokens)
        _validate_token_length(arrays, length, "tokens")

        kind = arrays["kind"]
        keyword_mask = kind == KIND["KEYWORD"]
        scalar_mask = kind == KIND["SCALAR"]
        keyword = _safe_embed_index(arrays["keyword"], keyword_mask)
        scalar_type = _safe_embed_index(arrays["scalar_type"], scalar_mask)
        scalar_value = jnp.where(
            scalar_mask,
            arrays["scalar_value"],
            0,
        )
        scalar_value = _normalize_scalar_values(
            scalar_value,
            scalar_value_min=self.scalar_value_min,
            scalar_value_max=self.scalar_value_max,
        )
        positions = jnp.arange(length, dtype=jnp.int32)

        embedded = (
            self.kind_embed(kind)
            + _where_component(keyword_mask, self.keyword_embed(keyword))
            + _where_component(scalar_mask, self.scalar_type_embed(scalar_type))
            + _where_component(
                scalar_mask,
                self.scalar_value_projection(scalar_value[..., None]),
            )
            + position_embed(positions)[None, :, :]
        )
        embedded = _mask_sequence(embedded, arrays["mask"])
        return self.embedding_dropout(embedded, deterministic=deterministic)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _token_batch(tokens: Mapping[str, Any]) -> dict[str, jax.Array]:
    arrays = _token_arrays(tokens)
    for field, array in arrays.items():
        if array.ndim != 2:
            raise ValueError(f"expected {field} to be a 2D token batch")
    return arrays


def _validate_token_length(
    tokens: Mapping[str, jax.Array],
    expected_length: int,
    name: str,
) -> None:
    actual_length = tokens["kind"].shape[1]
    if actual_length != expected_length:
        raise ValueError(
            f"expected {name} length {expected_length}, got {actual_length}"
        )


def _safe_embed_index(indices: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.where(mask, indices, 0)


def _normalize_scalar_values(
    values: jax.Array,
    *,
    scalar_value_min: int,
    scalar_value_max: int,
) -> jax.Array:
    midpoint = (scalar_value_min + scalar_value_max) / 2.0
    half_range = max((scalar_value_max - scalar_value_min) / 2.0, 1.0)
    return (values.astype(jnp.float32) - midpoint) / half_range


def _mask_sequence(x: jax.Array, mask: jax.Array) -> jax.Array:
    return x * mask[..., None].astype(x.dtype)


def _where_component(mask: jax.Array, component: jax.Array) -> jax.Array:
    return jnp.where(mask[..., None], component, 0)
