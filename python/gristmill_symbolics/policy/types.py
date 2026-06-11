from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

TokenTree: TypeAlias = dict[str, jax.Array]
ActionChoiceTree: TypeAlias = dict[str, jax.Array]


@dataclass(frozen=True)
class PolicyConfig:
    d_model: int = 32
    num_attention_layers: int = 1
    id_vocab_size: int = 128
    max_candidates: int = 32
    max_side_terms: int = 32
    init_scale: float = 0.02
    stop_bias_init: float = -20.0


def make_action_choice(
    *,
    candidate_index: int | list[int] | tuple[int, ...] | np.ndarray,
    left_mask: list[bool] | tuple[bool, ...] | np.ndarray,
    left_valid_mask: list[bool] | tuple[bool, ...] | np.ndarray,
    right_mask: list[bool] | tuple[bool, ...] | np.ndarray,
    right_valid_mask: list[bool] | tuple[bool, ...] | np.ndarray,
) -> ActionChoiceTree:
    candidate = jnp.asarray(candidate_index, dtype=jnp.int32)
    left = jnp.asarray(left_mask, dtype=jnp.bool_)
    left_valid = jnp.asarray(left_valid_mask, dtype=jnp.bool_)
    right = jnp.asarray(right_mask, dtype=jnp.bool_)
    right_valid = jnp.asarray(right_valid_mask, dtype=jnp.bool_)
    if candidate.shape != ():
        raise ValueError(f"candidate_index must be scalar, got shape {candidate.shape}")
    for name, values in (
        ("left_mask", left),
        ("left_valid_mask", left_valid),
        ("right_mask", right),
        ("right_valid_mask", right_valid),
    ):
        if values.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {values.shape}")
    if left.shape != left_valid.shape:
        raise ValueError(
            f"left_mask and left_valid_mask shapes differ: {left.shape} != {left_valid.shape}"
        )
    if right.shape != right_valid.shape:
        raise ValueError(
            f"right_mask and right_valid_mask shapes differ: {right.shape} != {right_valid.shape}"
        )
    if left.shape != right.shape:
        raise ValueError(
            f"left and right mask shapes differ: {left.shape} != {right.shape}"
        )
    return {
        "candidate_index": candidate,
        "left_mask": left,
        "left_valid_mask": left_valid,
        "right_mask": right,
        "right_valid_mask": right_valid,
    }


def action_choice_to_python(choice: Mapping[str, jax.Array]) -> dict[str, object]:
    return {
        "candidate_index": int(np.asarray(choice["candidate_index"])),
        "left_mask": [bool(v) for v in np.asarray(choice["left_mask"]).tolist()],
        "left_valid_mask": [
            bool(v) for v in np.asarray(choice["left_valid_mask"]).tolist()
        ],
        "right_mask": [bool(v) for v in np.asarray(choice["right_mask"]).tolist()],
        "right_valid_mask": [
            bool(v) for v in np.asarray(choice["right_valid_mask"]).tolist()
        ],
    }
