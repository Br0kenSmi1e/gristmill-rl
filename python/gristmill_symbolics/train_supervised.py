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
    return name


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
