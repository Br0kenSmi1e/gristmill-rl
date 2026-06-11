from __future__ import annotations

from collections.abc import Iterable, Sequence

import jax.numpy as jnp

from .constants import SENTINEL, TOKEN_KIND
from .types import TokenTree


def _pad_value(field: str) -> int:
    if field == "token_kind":
        return int(TOKEN_KIND.PAD)
    return SENTINEL


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
    length = (
        pad_to
        if pad_to is not None
        else max(int(mask.shape[0]) for _, mask in materialized)
    )
    padded_items = [pad_token_tree(tokens, mask, length) for tokens, mask in materialized]
    fields = tuple(padded_items[0][0])
    stacked = {
        field: jnp.stack([tokens[field] for tokens, _ in padded_items], axis=0)
        for field in fields
    }
    stacked_mask = jnp.stack([mask for _, mask in padded_items], axis=0)
    return stacked, stacked_mask
