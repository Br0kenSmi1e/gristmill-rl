from __future__ import annotations

from collections.abc import Iterable, Sequence

import jax.numpy as jnp

from .constants import SENTINEL, TOKEN_KIND
from .types import TokenTree


def _pad_value(field: str) -> int:
    if field == "token_kind":
        return int(TOKEN_KIND.PAD)
    return SENTINEL


def _validate_token_tree(tokens: TokenTree, mask: jnp.ndarray) -> None:
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape {mask.shape}")
    length = int(mask.shape[0])
    for field, values in tokens.items():
        if values.ndim != 1:
            raise ValueError(f"{field} leaf must be 1D, got shape {values.shape}")
        if int(values.shape[0]) != length:
            raise ValueError(
                f"{field} leaf length {values.shape[0]} does not match mask length {length}"
            )


def make_token_tree(
    rows: Sequence[dict[str, int]], fields: Sequence[str]
) -> tuple[TokenTree, jnp.ndarray]:
    columns: dict[str, list[int]] = {field: [] for field in fields}
    for row in rows:
        for field in fields:
            columns[field].append(int(row.get(field, _pad_value(field))))
    tokens = {
        field: jnp.asarray(values, dtype=jnp.int32) for field, values in columns.items()
    }
    mask = jnp.ones((len(rows),), dtype=jnp.bool_)
    return tokens, mask


def pad_token_tree(
    tokens: TokenTree, mask: jnp.ndarray, length: int
) -> tuple[TokenTree, jnp.ndarray]:
    _validate_token_tree(tokens, mask)
    current = int(mask.shape[0])
    if current > length:
        raise ValueError(
            f"cannot pad token tree of length {current} to shorter length {length}"
        )
    padded: TokenTree = {}
    pad_count = length - current
    for field, values in tokens.items():
        pad = jnp.full((pad_count,), _pad_value(field), dtype=values.dtype)
        padded[field] = jnp.concatenate([values, pad], axis=0)
    padded_mask = jnp.concatenate(
        [mask, jnp.zeros((pad_count,), dtype=jnp.bool_)], axis=0
    )
    return padded, padded_mask


def stack_token_trees(
    items: Iterable[tuple[TokenTree, jnp.ndarray]], pad_to: int | None = None
) -> tuple[TokenTree, jnp.ndarray]:
    materialized = list(items)
    if not materialized:
        raise ValueError("stack_token_trees requires at least one token tree")
    fields = set(materialized[0][0])
    for tokens, _ in materialized[1:]:
        if set(tokens) != fields:
            raise ValueError(
                f"token tree field set mismatch: {set(tokens)} != {fields}"
            )
    for tokens, mask in materialized:
        _validate_token_tree(tokens, mask)
    length = (
        pad_to
        if pad_to is not None
        else max(int(mask.shape[0]) for _, mask in materialized)
    )
    padded_items = [pad_token_tree(tokens, mask, length) for tokens, mask in materialized]
    field_names = tuple(padded_items[0][0])
    stacked = {
        field: jnp.stack([tokens[field] for tokens, _ in padded_items], axis=0)
        for field in field_names
    }
    stacked_mask = jnp.stack([mask for _, mask in padded_items], axis=0)
    return stacked, stacked_mask
