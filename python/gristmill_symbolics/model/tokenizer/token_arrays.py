from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp

from .vocabulary import SENTINEL, TOKEN_FIELDS, TOKEN_KIND

TokenArrays: TypeAlias = dict[str, jax.Array]


def _pad_value(field: str) -> int:
    if field == "token_kind":
        return int(TOKEN_KIND.PAD)
    return SENTINEL


def validate_token_arrays(tokens: TokenArrays, mask: jax.Array) -> int:
    fields = set(tokens)
    expected = set(TOKEN_FIELDS)
    if fields != expected:
        raise ValueError(
            f"token arrays field set mismatch: {fields} != {expected}"
        )
    if mask.dtype != jnp.bool_:
        raise ValueError(f"mask must have dtype bool, got {mask.dtype}")
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape {mask.shape}")
    length = int(mask.shape[0])
    for field, values in tokens.items():
        if values.dtype != jnp.int32:
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
) -> tuple[TokenArrays, jax.Array]:
    columns: dict[str, list[int]] = {field: [] for field in TOKEN_FIELDS}
    for position, row in enumerate(rows):
        for field in TOKEN_FIELDS:
            value = row.get(field, _default_value(field, position))
            columns[field].append(int(value))
    tokens = {
        field: jnp.asarray(values, dtype=jnp.int32)
        for field, values in columns.items()
    }
    mask = jnp.ones((len(rows),), dtype=jnp.bool_)
    return tokens, mask


def pad_token_arrays(
    tokens: TokenArrays,
    mask: jax.Array,
    length: int,
) -> tuple[TokenArrays, jax.Array]:
    current = validate_token_arrays(tokens, mask)
    if current > length:
        raise ValueError(
            f"cannot pad token arrays of length {current} "
            f"to shorter length {length}"
        )
    padded: TokenArrays = {}
    pad_count = length - current
    for field, values in tokens.items():
        pad = jnp.full((pad_count,), _pad_value(field), dtype=values.dtype)
        padded[field] = jnp.concatenate([values, pad], axis=0)
    padded_mask = jnp.concatenate(
        [mask, jnp.zeros((pad_count,), dtype=jnp.bool_)],
        axis=0,
    )
    return padded, padded_mask


def stack_token_arrays(
    items: Iterable[tuple[TokenArrays, jax.Array]],
    pad_to: int | None = None,
) -> tuple[TokenArrays, jax.Array]:
    materialized = list(items)
    if not materialized:
        raise ValueError(
            "stack_token_arrays requires at least one token array set"
        )
    fields = set(materialized[0][0])
    for tokens, _ in materialized[1:]:
        if set(tokens) != fields:
            raise ValueError(
                f"token arrays field set mismatch: {set(tokens)} != {fields}"
            )
    for tokens, mask in materialized:
        validate_token_arrays(tokens, mask)
    length = _stack_width(materialized, pad_to)
    padded = [
        pad_token_arrays(tokens, mask, length)
        for tokens, mask in materialized
    ]
    field_names = tuple(padded[0][0])
    stacked = {
        field: jnp.stack([tokens[field] for tokens, _ in padded], axis=0)
        for field in field_names
    }
    return stacked, jnp.stack([mask for _, mask in padded], axis=0)


def _default_value(field: str, position: int) -> int:
    if field == "position":
        return position
    return _pad_value(field)


def _stack_width(
    items: Sequence[tuple[TokenArrays, jax.Array]],
    pad_to: int | None,
) -> int:
    if pad_to is not None:
        return pad_to
    return max(int(mask.shape[0]) for _, mask in items)
