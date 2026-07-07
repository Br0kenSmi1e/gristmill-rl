from __future__ import annotations

import jax
import jax.numpy as jnp

from .grammar import FlatDefinitionGrammar

__all__ = (
    "constrained_next_token_step",
    "constrained_sequence_log_prob",
    "constrained_token_log_probs",
)


def constrained_next_token_step(
    grammar_state: jax.Array,
    input_token_ids: jax.Array,
    logits: jax.Array,
    grammar: FlatDefinitionGrammar,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    next_state = grammar.advance_state(grammar_state, input_token_ids)
    valid_next = jnp.take(grammar.allowed_by_state, next_state, axis=0)

    valid_any = jnp.any(valid_next, axis=-1, keepdims=True)
    masked_logits = jnp.where(valid_next, logits, -jnp.inf)
    safe_logits = jnp.where(valid_any, masked_logits, jnp.zeros_like(masked_logits))
    log_probs = jax.nn.log_softmax(safe_logits, axis=-1)

    return next_state, log_probs, valid_next


def constrained_token_log_probs(
    logits: jax.Array,
    decoder_input_ids: jax.Array,
    labels: jax.Array,
    label_mask: jax.Array,
    grammar: FlatDefinitionGrammar,
) -> jax.Array:
    batch_size = decoder_input_ids.shape[0]
    init_state = grammar.initial_state((batch_size,))

    def step(
        state: jax.Array,
        xs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        input_token_ids_t, logits_t, labels_t, label_mask_t = xs
        next_state, log_probs_t, valid_next_t = constrained_next_token_step(
            state,
            input_token_ids_t,
            logits_t,
            grammar,
        )

        safe_labels_t = jnp.where(label_mask_t, labels_t, 0)
        selected_t = jnp.take_along_axis(
            log_probs_t,
            safe_labels_t[:, None],
            axis=-1,
        )[:, 0]
        label_is_valid_t = jnp.take_along_axis(
            valid_next_t,
            safe_labels_t[:, None],
            axis=-1,
        )[:, 0]
        active_logp_t = jnp.where(label_is_valid_t, selected_t, -jnp.inf)
        token_logp_t = jnp.where(label_mask_t, active_logp_t, 0.0)
        return next_state, token_logp_t

    _final_state, token_logp_t_b = jax.lax.scan(
        step,
        init_state,
        (
            jnp.swapaxes(decoder_input_ids, 0, 1),
            jnp.swapaxes(logits, 0, 1),
            jnp.swapaxes(labels, 0, 1),
            jnp.swapaxes(label_mask, 0, 1),
        ),
    )
    return jnp.swapaxes(token_logp_t_b, 0, 1)


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
