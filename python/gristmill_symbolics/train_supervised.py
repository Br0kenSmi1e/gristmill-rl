from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from .grammar import FlatDefinitionGrammar
from .nn import FlatDefinitionSeq2SeqTransformer
from .supervised import SupervisedTrainer, weighted_nll
from .supervised_dataset import (
    iter_supervised_batches,
    load_preprocessed_supervised_dataset,
)
from .tokenizer import FlatDefinitionTokenizer


def _tokenizer_from_metadata(metadata: dict[str, Any]) -> FlatDefinitionTokenizer:
    tokenizer_metadata = metadata["tokenizer"]
    tokenizer = FlatDefinitionTokenizer(
        max_range_id=tokenizer_metadata["max_range_id"],
        max_tensor_id=tokenizer_metadata["max_tensor_id"],
        max_index_id=tokenizer_metadata["max_index_id"],
        coeff_nums=tuple(tokenizer_metadata["coeff_nums"]),
        coeff_dens=tuple(tokenizer_metadata["coeff_dens"]),
    )
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if getattr(tokenizer, name) != tokenizer_metadata[name]:
            raise ValueError(
                f"metadata tokenizer {name} does not match rebuilt tokenizer"
            )
    if metadata["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("metadata vocab_size does not match rebuilt tokenizer")
    return tokenizer


def _validate_compatible_metadata(
    train_metadata: dict[str, Any],
    valid_metadata: dict[str, Any],
) -> None:
    for key in ("source_len", "target_len", "vocab_size", "tokenizer"):
        if train_metadata[key] != valid_metadata[key]:
            raise ValueError(f"train and valid metadata mismatch for {key}")


def _dtype_from_name(name: str):
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float16":
        return jnp.float16
    raise ValueError(f"unsupported dtype {name!r}")


def _attention_from_name(name: str):
    if name == "default":
        return None
    if name in ("xla", "cudnn"):
        return name
    raise ValueError(f"unsupported attention implementation {name!r}")


def _as_jax_batch(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def _iter_update_groups(
    dataset: dict[str, Any],
    *,
    batch_size: int,
    accumulate_steps: int,
    rng: np.random.Generator,
):
    if accumulate_steps <= 0:
        raise ValueError("accumulate_steps must be positive")
    current = []
    for batch in iter_supervised_batches(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        rng=rng,
    ):
        current.append(_as_jax_batch(batch))
        if len(current) == accumulate_steps:
            yield tuple(current)
            current = []


def _evaluate_dataset(
    model: nnx.Module,
    dataset: dict[str, Any],
    grammar: FlatDefinitionGrammar,
    *,
    batch_size: int,
) -> dict[str, float | int]:
    def eval_step(model: nnx.Module, batch: dict[str, jax.Array]):
        return weighted_nll(model, batch, grammar, deterministic=True)

    jitted_eval_step = nnx.jit(eval_step)
    weighted_nll_sum = 0.0
    weight_sum = 0.0
    num_batches = 0
    for batch in iter_supervised_batches(dataset, batch_size=batch_size):
        batch_nll, batch_weight = jitted_eval_step(model, _as_jax_batch(batch))
        weighted_nll_sum += float(batch_nll)
        weight_sum += float(batch_weight)
        num_batches += 1
    mean_nll = weighted_nll_sum / weight_sum if weight_sum else 0.0
    return {
        "weighted_nll_sum": weighted_nll_sum,
        "weight_sum": weight_sum,
        "mean_nll": mean_nll,
        "num_batches": num_batches,
    }


def _build_model(args: argparse.Namespace, metadata: dict[str, Any], rng_seed: int):
    dtype = _dtype_from_name(args.dtype)
    return FlatDefinitionSeq2SeqTransformer(
        source_len=metadata["source_len"],
        target_len=metadata["target_len"],
        vocab_size=metadata["vocab_size"],
        pad_token_id=metadata["tokenizer"]["pad_token_id"],
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
        attention_implementation=_attention_from_name(args.attention_implementation),
        dtype=dtype,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(rng_seed),
    )
