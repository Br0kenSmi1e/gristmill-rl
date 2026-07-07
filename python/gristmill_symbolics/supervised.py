from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .grammar import FlatDefinitionGrammar
from .scoring import constrained_sequence_log_prob

__all__ = ("weighted_nll",)


def weighted_nll(
    model: nnx.Module,
    batch: dict[str, jax.Array],
    grammar: FlatDefinitionGrammar,
    *,
    deterministic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    logits = model(
        batch["source_ids"],
        batch["decoder_input_ids"],
        deterministic=deterministic,
    )
    sequence_logp = constrained_sequence_log_prob(
        logits,
        batch["decoder_input_ids"],
        batch["target_ids"],
        batch["target_mask"],
        grammar,
    )
    example_nll = -sequence_logp
    weighted_nll_sum = jnp.sum(batch["example_weight"] * example_nll)
    weight_sum = jnp.sum(batch["example_weight"])
    return weighted_nll_sum, weight_sum
