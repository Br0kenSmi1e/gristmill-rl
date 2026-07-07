from __future__ import annotations

import jax
import jax.numpy as jnp

from .grammar import FlatDefinitionGrammar

__all__ = (
    "constrained_sequence_log_prob",
    "constrained_token_log_probs",
)


def constrained_token_log_probs(
    logits: jax.Array,
    decoder_input_ids: jax.Array,
    labels: jax.Array,
    label_mask: jax.Array,
    grammar: FlatDefinitionGrammar,
) -> jax.Array:
    valid_next = grammar.valid_next_masks_for_decoder_input(decoder_input_ids)
    valid_any = jnp.any(valid_next, axis=-1, keepdims=True)
    masked_logits = jnp.where(valid_next, logits, -jnp.inf)
    safe_logits = jnp.where(valid_any, masked_logits, jnp.zeros_like(masked_logits))
    log_probs = jax.nn.log_softmax(safe_logits, axis=-1)

    safe_labels = jnp.where(label_mask, labels, 0)
    selected = jnp.take_along_axis(
        log_probs,
        safe_labels[..., None],
        axis=-1,
    )[..., 0]
    label_is_valid = jnp.take_along_axis(
        valid_next,
        safe_labels[..., None],
        axis=-1,
    )[..., 0]
    active_logp = jnp.where(label_is_valid, selected, -jnp.inf)
    return jnp.where(label_mask, active_logp, 0.0)


def constrained_sequence_log_prob(
    logits: jax.Array,
    decoder_input_ids: jax.Array,
    labels: jax.Array,
    label_mask: jax.Array,
    grammar: FlatDefinitionGrammar,
) -> jax.Array:
    token_logp = constrained_token_log_probs(
        logits,
        decoder_input_ids,
        labels,
        label_mask,
        grammar,
    )
    return jnp.sum(token_logp, axis=-1)
