from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from .model import make_decoder_inputs
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


@dataclass(frozen=True)
class DirectOptimizerTrainer:
    batch_size: int
    learning_rate: float


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


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
