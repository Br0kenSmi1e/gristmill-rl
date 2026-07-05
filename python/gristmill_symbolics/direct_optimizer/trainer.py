from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .model import DirectOptimizerTransformer, make_decoder_inputs, sequence_log_prob
from .tokens import encode_text, pad_tokens, validate_scalar_bounds


_TOKEN_FIELDS = ("kind", "keyword", "scalar_type", "scalar_value", "mask")
_REQUIRED_FIELDS = (
    "source_text",
    "target_text",
    "weight",
    "input_key",
    "candidate_key",
    "candidate_log_flops",
)


class DirectOptimizerTrainer:
    def __init__(
        self,
        *,
        batch_size: int,
        learning_rate: float = 1.0e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1.0e-8,
    ):
        self.batch_size = _validate_positive_int("batch_size", batch_size)
        self.learning_rate = _validate_positive_finite(
            "learning_rate",
            learning_rate,
        )
        self.b1 = _validate_adam_decay("b1", b1)
        self.b2 = _validate_adam_decay("b2", b2)
        self.eps = _validate_positive_finite("eps", eps)

    def constructor_kwargs(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "b1": self.b1,
            "b2": self.b2,
            "eps": self.eps,
        }

    def init_optimizer(self, model: DirectOptimizerTransformer) -> nnx.Optimizer:
        tx = optax.adam(
            self.learning_rate,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
        )
        return nnx.Optimizer(model, tx, wrt=nnx.Param)

    def train_step(
        self,
        model: DirectOptimizerTransformer,
        optimizer: nnx.Optimizer,
        batch: Mapping[str, Any],
    ) -> dict[str, jax.Array]:
        loss = _train_step(model, optimizer, batch)
        return {"train_loss": loss}

    def eval_step(
        self,
        model: DirectOptimizerTransformer,
        batch: Mapping[str, Any],
        *,
        metric_name: str = "valid_loss",
    ) -> dict[str, jax.Array]:
        loss = _eval_loss(model, batch)
        return {metric_name: loss}


def collate_processed_rows(
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    source_len: int,
    target_len: int,
    scalar_value_min: int,
    scalar_value_max: int,
) -> list[dict[str, Any]]:
    _validate_positive("batch_size", batch_size)
    _validate_positive("source_len", source_len)
    _validate_positive("target_len", target_len)
    if scalar_value_min > scalar_value_max:
        raise ValueError("scalar_value_min must be <= scalar_value_max")

    compatible_rows: list[dict[str, Any]] = []
    for row in rows:
        collated = _collate_row(
            row,
            source_len=source_len,
            target_len=target_len,
            scalar_value_min=scalar_value_min,
            scalar_value_max=scalar_value_max,
        )
        if collated is not None:
            compatible_rows.append(collated)

    batches: list[dict[str, Any]] = []
    full_count = len(compatible_rows) - (len(compatible_rows) % batch_size)
    for start in range(0, full_count, batch_size):
        batches.append(_batch_rows(compatible_rows[start : start + batch_size]))
    if not batches:
        raise ValueError("fewer compatible rows than batch_size")
    return batches


def weighted_sequence_loss(
    sequence_logp,
    example_weight,
    *,
    epsilon: float = 1.0e-8,
):
    weights = jnp.asarray(example_weight, dtype=jnp.float32)
    nll = -jnp.asarray(sequence_logp, dtype=jnp.float32)
    return jnp.sum(weights * nll) / jnp.maximum(jnp.sum(weights), epsilon)


@nnx.jit
def _train_step(
    model: DirectOptimizerTransformer,
    optimizer: nnx.Optimizer,
    batch: Mapping[str, Any],
) -> jax.Array:
    def loss_fn(model: DirectOptimizerTransformer) -> jax.Array:
        logits = model(
            batch["source_tokens"],
            batch["decoder_input_tokens"],
            deterministic=False,
        )
        sequence_logp = sequence_log_prob(
            logits,
            batch["target_tokens"],
            batch["target_mask"],
        )
        return weighted_sequence_loss(sequence_logp, batch["example_weight"])

    loss, grads = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, nnx.Param),
    )(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def _eval_loss(
    model: DirectOptimizerTransformer,
    batch: Mapping[str, Any],
) -> jax.Array:
    logits = model(
        batch["source_tokens"],
        batch["decoder_input_tokens"],
        deterministic=True,
    )
    sequence_logp = sequence_log_prob(
        logits,
        batch["target_tokens"],
        batch["target_mask"],
    )
    return weighted_sequence_loss(sequence_logp, batch["example_weight"])


def _collate_row(
    row: dict[str, Any],
    *,
    source_len: int,
    target_len: int,
    scalar_value_min: int,
    scalar_value_max: int,
) -> dict[str, Any] | None:
    try:
        if any(field not in row for field in _REQUIRED_FIELDS):
            return None

        weight = float(row["weight"])
        if not math.isfinite(weight) or weight < 0.0:
            return None
        if not math.isfinite(float(row["candidate_log_flops"])):
            return None

        source_tokens = pad_tokens(encode_text(row["source_text"]), length=source_len)
        target_row = pad_tokens(encode_text(row["target_text"]), length=target_len)
        decoder_input_tokens, target_tokens, target_mask = make_decoder_inputs(
            target_row
        )

        source_tokens = _numpy_token_row(source_tokens)
        decoder_input_tokens = _numpy_token_row(decoder_input_tokens)
        target_tokens = _numpy_token_row(target_tokens)
        target_mask = np.asarray(target_mask, dtype=bool)

        for tokens in (source_tokens, decoder_input_tokens, target_tokens):
            validate_scalar_bounds(
                tokens,
                scalar_value_min=scalar_value_min,
                scalar_value_max=scalar_value_max,
            )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return None

    return {
        "source_tokens": source_tokens,
        "decoder_input_tokens": decoder_input_tokens,
        "target_tokens": target_tokens,
        "target_mask": target_mask,
        "example_weight": np.float32(weight),
    }


def _batch_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_tokens": _stack_token_rows(
            [row["source_tokens"] for row in rows]
        ),
        "decoder_input_tokens": _stack_token_rows(
            [row["decoder_input_tokens"] for row in rows]
        ),
        "target_tokens": _stack_token_rows(
            [row["target_tokens"] for row in rows]
        ),
        "target_mask": np.stack([row["target_mask"] for row in rows]).astype(bool),
        "example_weight": np.asarray(
            [row["example_weight"] for row in rows],
            dtype=np.float32,
        ),
    }


def _stack_token_rows(rows: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        field: np.stack([row[field] for row in rows]).astype(_dtype_for_field(field))
        for field in _TOKEN_FIELDS
    }


def _numpy_token_row(tokens: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(tokens[field], dtype=_dtype_for_field(field))
        for field in _TOKEN_FIELDS
    }


def _dtype_for_field(field: str):
    if field == "mask":
        return bool
    return np.int32


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validate_positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_adam_decay(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be finite and satisfy 0 <= {name} < 1")
    return value


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
