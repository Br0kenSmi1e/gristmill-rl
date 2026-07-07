from __future__ import annotations

import jax
import jax.numpy as jnp

from .grammar import FlatDefinitionGrammar
from .scoring import constrained_sequence_log_prob

__all__ = ("weighted_supervised_nll_totals",)


def weighted_supervised_nll_totals(
    logits: jax.Array,
    decoder_input_ids: jax.Array,
    target_ids: jax.Array,
    target_mask: jax.Array,
    example_weight: jax.Array,
    grammar: FlatDefinitionGrammar,
) -> tuple[jax.Array, jax.Array]:
    sequence_logp = constrained_sequence_log_prob(
        logits,
        decoder_input_ids,
        target_ids,
        target_mask,
        grammar,
    )
    example_nll = -sequence_logp
    weighted_nll_sum = jnp.sum(example_weight * example_nll)
    weight_sum = jnp.sum(example_weight)
    return weighted_nll_sum, weight_sum
