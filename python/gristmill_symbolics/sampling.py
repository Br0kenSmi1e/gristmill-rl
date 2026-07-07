from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .grammar import FlatDefinitionGrammar

__all__ = ("FlatTokenSamplingResult", "sample_token_ids")


class FlatTokenSamplingResult(NamedTuple):
    generated_ids: jax.Array
    token_log_probs: jax.Array
    sequence_log_prob: jax.Array


def sample_token_ids(
    model,
    rng: jax.Array,
    source_ids: jax.Array,
    grammar: FlatDefinitionGrammar,
    *,
    target_len: int,
) -> FlatTokenSamplingResult:
    batch_size = source_ids.shape[0]
    generated_ids = jnp.full(
        (batch_size, target_len),
        grammar.pad_token_id,
        dtype=jnp.int32,
    )
    generated_ids = generated_ids.at[:, 0].set(grammar.bos_token_id)
    token_log_probs = jnp.zeros((batch_size, target_len), dtype=jnp.float32)

    init_state = grammar.advance_state(
        grammar.initial_state((batch_size,)),
        jnp.full((batch_size,), grammar.bos_token_id, dtype=jnp.int32),
    )
    init_finished = jnp.zeros((batch_size,), dtype=bool)

    def step(carry, t: jax.Array):
        prefix, logps, state, finished, step_rng = carry
        step_rng, sample_rng = jax.random.split(step_rng)

        logits = model(source_ids, prefix, deterministic=True)
        step_logits = logits[:, t, :]
        valid_next = jnp.take(grammar.allowed_by_state, state, axis=0)
        masked_logits = jnp.where(valid_next, step_logits, -jnp.inf)
        sampled_ids = jax.random.categorical(sample_rng, masked_logits, axis=-1)
        sampled_ids = sampled_ids.astype(jnp.int32)

        next_ids = jnp.where(finished, grammar.pad_token_id, sampled_ids)
        step_log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
        selected_logps = jnp.take_along_axis(
            step_log_probs,
            sampled_ids[:, None],
            axis=-1,
        )[:, 0]
        selected_logps = jnp.where(finished, 0.0, selected_logps)

        next_state = grammar.advance_state(state, next_ids)
        next_finished = finished | (next_ids == grammar.eos_token_id)
        next_pos = t + 1
        prefix = prefix.at[:, next_pos].set(next_ids)
        logps = logps.at[:, next_pos].set(selected_logps)

        return (prefix, logps, next_state, next_finished, step_rng), None

    (generated_ids, token_log_probs, _state, _finished, _rng), _ = jax.lax.scan(
        step,
        (generated_ids, token_log_probs, init_state, init_finished, rng),
        jnp.arange(target_len - 1),
    )
    sequence_log_prob = jnp.sum(token_log_probs, axis=-1)

    return FlatTokenSamplingResult(
        generated_ids=generated_ids,
        token_log_probs=token_log_probs,
        sequence_log_prob=sequence_log_prob,
    )
