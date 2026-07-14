from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from . import TensorComputation, equivalent_computations
from .grammar import FlatDefinitionGrammar
from .scoring import constrained_next_token_step
from .tokenizer import FlatDefinitionTokenizer, TokenizerError

__all__ = (
    "generated_ids_to_tensor_computation",
    "sample_tensor_computations",
    "sample_token_ids",
)


def sample_token_ids(
    model,
    rng: jax.Array,
    source_ids: jax.Array,
    grammar: FlatDefinitionGrammar,
    *,
    target_len: int,
    temperature: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    model.init_decode_cache(batch_size=source_ids.shape[0], target_len=target_len)
    return _sample_token_ids_jit(
        model,
        rng,
        source_ids,
        grammar,
        target_len=target_len,
        temperature=jnp.asarray(temperature, dtype=jnp.float32),
    )


@nnx.jit(static_argnames=("grammar", "target_len"))
def _sample_token_ids_jit(
    model,
    rng: jax.Array,
    source_ids: jax.Array,
    grammar: FlatDefinitionGrammar,
    *,
    target_len: int,
    temperature: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_size = source_ids.shape[0]
    generated_ids = jnp.full(
        (batch_size, target_len),
        grammar.pad_token_id,
        dtype=jnp.int32,
    )
    generated_ids = generated_ids.at[:, 0].set(grammar.bos_token_id)
    token_log_probs = jnp.zeros((batch_size, target_len), dtype=jnp.float32)
    memory, source_mask = model.encode(source_ids, deterministic=True)

    init_state = grammar.initial_state((batch_size,))
    init_finished = jnp.zeros((batch_size,), dtype=bool)

    def step(
        t: int,
        carry: tuple[
            nnx.Module,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ) -> tuple[nnx.Module, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        (
            model,
            generated_ids,
            token_log_probs,
            state,
            finished,
            step_rng,
        ) = carry
        next_step_rng, sample_rng = jax.random.split(step_rng)

        step_logits = model.decode_step(
            generated_ids[:, t],
            memory,
            source_mask=source_mask,
            step=t,
            deterministic=True,
        )
        step_logits = step_logits / temperature
        next_state, step_log_probs, _valid_next = constrained_next_token_step(
            state,
            generated_ids[:, t],
            step_logits,
            grammar,
        )
        sampled_ids = jax.random.categorical(sample_rng, step_log_probs, axis=-1)
        sampled_ids = sampled_ids.astype(jnp.int32)

        next_ids = jnp.where(finished, grammar.pad_token_id, sampled_ids)
        selected_logps = jnp.take_along_axis(
            step_log_probs,
            sampled_ids[:, None],
            axis=-1,
        )[:, 0]
        selected_logps = jnp.where(finished, 0.0, selected_logps)

        next_finished = finished | (next_ids == grammar.eos_token_id)
        next_pos = t + 1
        generated_ids = generated_ids.at[:, next_pos].set(next_ids)
        token_log_probs = token_log_probs.at[:, next_pos].set(selected_logps)
        return (
            model,
            generated_ids,
            token_log_probs,
            next_state,
            next_finished,
            next_step_rng,
        )

    (
        _model,
        generated_ids,
        token_log_probs,
        _state,
        _finished,
        _rng,
    ) = nnx.fori_loop(
        0,
        target_len - 1,
        step,
        (
            model,
            generated_ids,
            token_log_probs,
            init_state,
            init_finished,
            rng,
        ),
    )

    sequence_log_prob = jnp.sum(token_log_probs, axis=-1)

    return generated_ids, token_log_probs, sequence_log_prob


def generated_ids_to_tensor_computation(
    input_computation: TensorComputation,
    tokenizer: FlatDefinitionTokenizer,
    generated_ids: Sequence[int],
) -> TensorComputation:
    definitions = tokenizer.decode_definitions_generated(generated_ids)
    input_snapshot = input_computation.snapshot()
    tensors = list(input_snapshot["tensors"])
    known_tensors = {int(tensor["id"]) for tensor in tensors}

    generated_bases = {int(definition["base"]) for definition in definitions}
    for base in sorted(generated_bases - known_tensors):
        tensors.append({"id": base, "symmetry": []})
        known_tensors.add(base)

    for definition in definitions:
        for term in definition["terms"]:
            for factor in term["factors"]:
                tensor_id = int(factor["tensor"])
                if tensor_id not in known_tensors:
                    raise ValueError(f"factor references unknown tensor_id:{tensor_id}")

    snapshot = {
        "ranges": list(input_snapshot["ranges"]),
        "tensors": tensors,
        "definitions": _definitions_to_constructor_json(definitions),
    }
    try:
        return TensorComputation.from_json_string(json.dumps(snapshot))
    except Exception as exc:
        tensor_ids = [int(tensor["id"]) for tensor in tensors]
        raise ValueError(
            "sampled reconstruction failed TensorComputation validation "
            f"for tensor_ids:{tensor_ids}: {exc}"
        ) from exc


def sample_tensor_computations(
    model,
    rng: jax.Array,
    input_computation: TensorComputation,
    source_ids: jax.Array,
    tokenizer: FlatDefinitionTokenizer,
    grammar: FlatDefinitionGrammar,
    *,
    target_len: int,
    outputs: Sequence[int] | None = None,
    temperature: float = 1.0,
) -> tuple[list[TensorComputation], dict[str, int]]:
    generated_ids, _token_log_probs, _sequence_log_prob = sample_token_ids(
        model,
        rng,
        source_ids,
        grammar,
        target_len=target_len,
        temperature=temperature,
    )

    metrics = {
        "total_samples": int(generated_ids.shape[0]),
        "decode_failures": 0,
        "reconstruction_failures": 0,
        "verifier_failures": 0,
        "valid_samples": 0,
    }
    candidates: list[TensorComputation] = []

    for row in jax.device_get(generated_ids):
        token_ids = [int(token_id) for token_id in row]
        try:
            candidate = generated_ids_to_tensor_computation(
                input_computation,
                tokenizer,
                token_ids,
            )
        except TokenizerError:
            metrics["decode_failures"] += 1
            continue
        except ValueError:
            metrics["reconstruction_failures"] += 1
            continue

        if outputs is not None:
            try:
                if not equivalent_computations(input_computation, candidate, outputs):
                    metrics["verifier_failures"] += 1
                    continue
            except Exception:
                metrics["verifier_failures"] += 1
                continue

        candidates.append(candidate)
        metrics["valid_samples"] += 1

    return candidates, metrics


def _definitions_to_constructor_json(
    definitions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    constructor_definitions = []
    for definition in definitions:
        constructor_terms = []
        for term in definition["terms"]:
            coeff = term["coeff"]
            constructor_terms.append(
                {
                    "coeff": [coeff["numer"], coeff["denom"]],
                    "sum_indices": list(term["sum_indices"]),
                    "factors": list(term["factors"]),
                }
            )
        constructor_definitions.append(
            {
                "base": definition["base"],
                "ext_indices": list(definition["ext_indices"]),
                "terms": constructor_terms,
            }
        )
    return constructor_definitions
