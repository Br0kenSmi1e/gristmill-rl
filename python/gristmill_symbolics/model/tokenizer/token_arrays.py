from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from .vocabulary import SENTINEL, TOKEN_FIELDS, TOKEN_KIND

TokenArrays: TypeAlias = dict[str, jax.Array | np.ndarray]


def _pad_value(field: str) -> int:
    if field == "token_kind":
        return int(TOKEN_KIND.PAD)
    return SENTINEL


def validate_token_arrays(
    tokens: TokenArrays,
    mask: jax.Array | np.ndarray,
) -> int:
    fields = set(tokens)
    expected = set(TOKEN_FIELDS)
    if fields != expected:
        raise ValueError(
            f"token arrays field set mismatch: {fields} != {expected}"
        )
    if not _has_dtype(mask, np.bool_):
        raise ValueError(f"mask must have dtype bool, got {mask.dtype}")
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape {mask.shape}")
    length = int(mask.shape[0])
    for field, values in tokens.items():
        if not _has_dtype(values, np.int32):
            raise ValueError(
                f"{field} leaf must have dtype int32, got {values.dtype}"
            )
        if values.ndim != 1:
            raise ValueError(
                f"{field} leaf must be 1D, got shape {values.shape}"
            )
        if int(values.shape[0]) != length:
            raise ValueError(
                f"{field} leaf length {values.shape[0]} "
                f"does not match mask length {length}"
            )
    return length


def make_token_arrays(
    rows: Sequence[dict[str, int]],
) -> tuple[TokenArrays, np.ndarray]:
    columns: dict[str, list[int]] = {field: [] for field in TOKEN_FIELDS}
    for position, row in enumerate(rows):
        for field in TOKEN_FIELDS:
            value = row.get(field, _default_value(field, position))
            columns[field].append(int(value))
    tokens = {
        field: np.asarray(values, dtype=np.int32)
        for field, values in columns.items()
    }
    mask = np.ones((len(rows),), dtype=np.bool_)
    return tokens, mask


def pad_token_arrays(
    tokens: TokenArrays,
    mask: jax.Array | np.ndarray,
    length: int,
) -> tuple[TokenArrays, np.ndarray]:
    current = validate_token_arrays(tokens, mask)
    _check_pad_length(current, length)
    padded: TokenArrays = {}
    pad_count = length - current
    for field, values in tokens.items():
        values_array = np.asarray(values, dtype=np.int32)
        pad = np.full((pad_count,), _pad_value(field), dtype=np.int32)
        padded[field] = np.concatenate([values_array, pad], axis=0)
    mask_array = np.asarray(mask, dtype=np.bool_)
    padded_mask = np.concatenate(
        [mask_array, np.zeros((pad_count,), dtype=np.bool_)],
        axis=0,
    )
    return padded, padded_mask


def stack_token_arrays(
    items: Iterable[tuple[TokenArrays, jax.Array | np.ndarray]],
    pad_to: int | None = None,
) -> tuple[TokenArrays, jax.Array]:
    materialized = list(items)
    if not materialized:
        raise ValueError(
            "stack_token_arrays requires at least one token array set"
        )
    lengths = [
        validate_token_arrays(tokens, mask)
        for tokens, mask in materialized
    ]
    length = _stack_width(lengths, pad_to)
    batch_size = len(materialized)
    stacked = _empty_stacked_tokens(batch_size, length)
    stacked_mask = np.zeros((batch_size, length), dtype=np.bool_)
    for row, ((tokens, mask), current) in enumerate(
        zip(materialized, lengths)
    ):
        _check_pad_length(current, length)
        _write_stacked_row(stacked, row, tokens, current)
        stacked_mask[row, :current] = np.asarray(mask, dtype=np.bool_)
    return _to_jax_tokens(stacked), jnp.asarray(stacked_mask)


def _has_dtype(values: jax.Array | np.ndarray, dtype) -> bool:
    return np.dtype(values.dtype) == np.dtype(dtype)


def _check_pad_length(current: int, length: int) -> None:
    if current <= length:
        return
    raise ValueError(
        f"cannot pad token arrays of length {current} "
        f"to shorter length {length}"
    )


def _empty_stacked_tokens(
    batch_size: int,
    length: int,
) -> dict[str, np.ndarray]:
    return {
        field: np.full(
            (batch_size, length),
            _pad_value(field),
            dtype=np.int32,
        )
        for field in TOKEN_FIELDS
    }


def _write_stacked_row(
    stacked: dict[str, np.ndarray],
    row: int,
    tokens: TokenArrays,
    length: int,
) -> None:
    for field in TOKEN_FIELDS:
        stacked[field][row, :length] = np.asarray(
            tokens[field],
            dtype=np.int32,
        )


def _to_jax_tokens(tokens: dict[str, np.ndarray]) -> TokenArrays:
    return {
        field: jnp.asarray(values)
        for field, values in tokens.items()
    }


def _default_value(field: str, position: int) -> int:
    if field == "position":
        return position
    return _pad_value(field)


def _stack_width(
    lengths: Sequence[int],
    pad_to: int | None,
) -> int:
    if pad_to is not None:
        return pad_to
    return max(lengths)
