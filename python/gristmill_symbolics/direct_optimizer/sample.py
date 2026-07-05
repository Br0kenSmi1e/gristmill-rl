from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from gristmill_symbolics import TensorComputation, equivalent_computations

from .converter import (
    computation_to_source_text,
    target_text_to_computation,
    target_text_to_definitions,
)
from .model import DirectOptimizerTransformer, sample_tokens
from .tokens import decode_token_row_to_text, encode_text, pad_tokens, repeat_token_row


def optimize_with_model(
    model: DirectOptimizerTransformer,
    params,
    input_computation: TensorComputation,
    outputs: list[int],
    *,
    num_samples: int,
    sample_batch_size: int,
    source_len: int,
    target_len: int,
    temperature: float,
    seed: int,
) -> tuple[TensorComputation | None, dict[str, Any]]:
    del params
    _validate_positive_int(num_samples, "num_samples")
    _validate_positive_int(sample_batch_size, "sample_batch_size")
    _validate_positive_int(source_len, "source_len")
    _validate_positive_int(target_len, "target_len")
    _validate_temperature(temperature)
    _validate_outputs(outputs)

    source_text = computation_to_source_text(input_computation)
    source_row = pad_tokens(encode_text(source_text), length=source_len)
    source_batch = {
        field: jnp.asarray(values)
        for field, values in repeat_token_row(
            source_row,
            batch_size=sample_batch_size,
        ).items()
    }

    metrics: dict[str, Any] = {
        "total_samples": num_samples,
        "decode_failures": 0,
        "parse_failures": 0,
        "reconstruction_failures": 0,
        "verifier_failures": 0,
        "valid_samples": 0,
        "best_log_flops": None,
    }
    best_candidate: TensorComputation | None = None
    best_log_flops: float | None = None

    key = jax.random.PRNGKey(seed)
    batches = math.ceil(num_samples / sample_batch_size)
    samples_seen = 0
    for _ in range(batches):
        key, batch_key = jax.random.split(key)
        generated, _mask = sample_tokens(
            model,
            batch_key,
            source_batch,
            max_length=target_len,
            temperature=temperature,
        )
        remaining = num_samples - samples_seen
        rows_to_evaluate = min(sample_batch_size, remaining)
        for row_index in range(rows_to_evaluate):
            token_row = {field: values[row_index] for field, values in generated.items()}
            candidate, log_flops = _candidate_from_tokens(
                token_row,
                input_computation,
                outputs,
                metrics,
            )
            if candidate is None:
                continue
            metrics["valid_samples"] += 1
            if best_log_flops is None or log_flops < best_log_flops:
                best_candidate = candidate
                best_log_flops = log_flops
                metrics["best_log_flops"] = log_flops
        samples_seen += rows_to_evaluate

    return best_candidate, metrics


def _candidate_from_tokens(
    token_row: dict[str, Any],
    input_computation: TensorComputation,
    outputs: list[int],
    metrics: dict[str, Any],
) -> tuple[TensorComputation | None, float | None]:
    try:
        target_text = decode_token_row_to_text(token_row)
    except ValueError:
        metrics["decode_failures"] += 1
        return None, None

    try:
        target_text_to_definitions(target_text)
    except ValueError:
        metrics["parse_failures"] += 1
        return None, None

    try:
        candidate = target_text_to_computation(input_computation, target_text)
    except ValueError:
        metrics["reconstruction_failures"] += 1
        return None, None

    try:
        if not equivalent_computations(input_computation, candidate, outputs):
            metrics["verifier_failures"] += 1
            return None, None
        log_flops = float(candidate.log_total_flops())
    except Exception:
        metrics["verifier_failures"] += 1
        return None, None

    if not math.isfinite(log_flops):
        metrics["verifier_failures"] += 1
        return None, None
    return candidate, log_flops


def _validate_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_temperature(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("temperature must be finite and positive")


def _validate_outputs(outputs: Any) -> None:
    if not isinstance(outputs, list) or len(outputs) == 0:
        raise ValueError("outputs must be a non-empty list of integer ids")
    seen: set[int] = set()
    for output in outputs:
        if isinstance(output, bool) or not isinstance(output, int) or output < 0:
            raise ValueError("outputs must be non-negative integer ids")
        if output in seen:
            raise ValueError("outputs must not contain duplicates")
        seen.add(output)
