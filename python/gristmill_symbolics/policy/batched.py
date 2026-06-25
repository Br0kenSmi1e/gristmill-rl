from __future__ import annotations

import jax

from .api import sample_action, sample_target, score_action, score_target

batched_sample_target = jax.jit(
    jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))
)

batched_score_target_grad = jax.jit(
    jax.vmap(
        jax.value_and_grad(score_target, argnums=0),
        in_axes=(None, 0, 0, 0, 0),
    )
)

batched_sample_action = jax.jit(
    jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))
)

batched_score_action_grad = jax.jit(
    jax.vmap(
        jax.value_and_grad(score_action, argnums=0),
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )
)

__all__ = (
    "batched_sample_target",
    "batched_score_target_grad",
    "batched_sample_action",
    "batched_score_action_grad",
)
