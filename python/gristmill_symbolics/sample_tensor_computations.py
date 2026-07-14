from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from . import TensorComputation
from .grammar import FlatDefinitionGrammar
from .nn import FlatDefinitionSeq2SeqTransformer
from .sampling import sample_tensor_computations
from .tokenizer import FlatDefinitionTokenizer


def _parse_int_csv(value: str) -> list[int]:
    try:
        items = [item.strip() for item in value.split(",")]
        if not items or any(item == "" for item in items):
            raise ValueError
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from exc


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


def _positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name: str, value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample TensorComputation candidates from a flat seq2seq checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--sample-batch-size", required=True, type=int)
    parser.add_argument("--source-len", required=True, type=int)
    parser.add_argument("--target-len", required=True, type=int)
    parser.add_argument("--max-range-id", required=True, type=int)
    parser.add_argument("--max-tensor-id", required=True, type=int)
    parser.add_argument("--max-index-id", required=True, type=int)
    parser.add_argument("--coeff-nums", required=True, type=_parse_int_csv)
    parser.add_argument("--coeff-dens", required=True, type=_parse_int_csv)
    parser.add_argument("--d-model", required=True, type=int)
    parser.add_argument("--num-layers", required=True, type=int)
    parser.add_argument("--num-heads", required=True, type=int)
    parser.add_argument("--mlp-hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("default", "xla", "cudnn"),
        default="default",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-outputs", type=_parse_int_csv, default=None)
    return parser


def _normalize_coeff_args(argv: list[str]) -> list[str]:
    normalized = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("--coeff-nums", "--coeff-dens") and index + 1 < len(argv):
            value = argv[index + 1]
            if not value.startswith("--"):
                normalized.append(f"{arg}={value}")
                index += 2
                continue
        normalized.append(arg)
        index += 1
    return normalized


def _validate_args(args: argparse.Namespace) -> None:
    _positive_int("samples", args.samples)
    _positive_int("sample_batch_size", args.sample_batch_size)
    _positive_int("source_len", args.source_len)
    _positive_int("target_len", args.target_len)
    _positive_int("d_model", args.d_model)
    _positive_int("num_layers", args.num_layers)
    _positive_int("num_heads", args.num_heads)
    if args.mlp_hidden_dim is not None:
        _positive_int("mlp_hidden_dim", args.mlp_hidden_dim)
    _positive_float("temperature", args.temperature)


def _build_tokenizer(args: argparse.Namespace) -> FlatDefinitionTokenizer:
    return FlatDefinitionTokenizer(
        max_range_id=args.max_range_id,
        max_tensor_id=args.max_tensor_id,
        max_index_id=args.max_index_id,
        coeff_nums=tuple(args.coeff_nums),
        coeff_dens=tuple(args.coeff_dens),
    )


def _build_model(
    args: argparse.Namespace,
    tokenizer: FlatDefinitionTokenizer,
) -> FlatDefinitionSeq2SeqTransformer:
    return FlatDefinitionSeq2SeqTransformer(
        source_len=args.source_len,
        target_len=args.target_len,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
        attention_implementation=_attention_from_name(args.attention_implementation),
        dtype=_dtype_from_name(args.dtype),
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(args.seed),
    )


def _restore_model_state(
    checkpoint_path: str | Path,
    model: nnx.Module,
) -> None:
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(0.0),
        wrt=nnx.Param,
    )
    target = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
    }
    try:
        restored = ocp.PyTreeCheckpointer().restore(
            Path(checkpoint_path).resolve(),
            item=target,
        )
    except Exception as exc:
        raise ValueError(
            "checkpoint restore failed; check CLI model flags and checkpoint path"
        ) from exc

    restored_model_state = restored["model"]
    _validate_state_shapes(nnx.state(model), restored_model_state)
    nnx.update(model, restored_model_state)


def _validate_state_shapes(expected: Any, restored: Any) -> None:
    expected_shapes = _state_shapes(expected)
    restored_shapes = _state_shapes(restored)
    if expected_shapes == restored_shapes:
        return

    expected_paths = set(expected_shapes)
    restored_paths = set(restored_shapes)
    missing = sorted(expected_paths - restored_paths)
    extra = sorted(restored_paths - expected_paths)
    mismatched = sorted(
        path
        for path in expected_paths & restored_paths
        if expected_shapes[path] != restored_shapes[path]
    )
    details = []
    if missing:
        details.append(f"missing:{missing[0]}")
    if extra:
        details.append(f"extra:{extra[0]}")
    if mismatched:
        path = mismatched[0]
        details.append(
            f"shape:{path} expected {expected_shapes[path]} restored {restored_shapes[path]}"
        )
    suffix = f" ({'; '.join(details)})" if details else ""
    raise ValueError(f"checkpoint model state does not match CLI model flags{suffix}")


def _state_shapes(state: Any) -> dict[str, tuple[int, ...] | None]:
    shapes = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        shapes[_tree_path_name(path)] = getattr(leaf, "shape", None)
    return shapes


def _tree_path_name(path: Sequence[object]) -> str:
    return "/".join(str(part) for part in path)


def _source_ids(
    input_computation: TensorComputation,
    tokenizer: FlatDefinitionTokenizer,
    *,
    source_len: int,
) -> list[int]:
    return tokenizer.encode_definitions_padded(
        input_computation.snapshot()["definitions"],
        length=source_len,
    )


def _empty_metrics() -> dict[str, int]:
    return {
        "total_samples": 0,
        "decode_failures": 0,
        "reconstruction_failures": 0,
        "verifier_failures": 0,
        "valid_samples": 0,
    }


def _add_metrics(total: dict[str, int], batch: dict[str, int]) -> None:
    for key in total:
        total[key] += int(batch[key])


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args = _parser().parse_args(_normalize_coeff_args(list(raw_argv)))
    _validate_args(args)

    input_computation = TensorComputation.load_json(args.input)
    tokenizer = _build_tokenizer(args)
    grammar = FlatDefinitionGrammar(tokenizer)
    model = _build_model(args, tokenizer)
    _restore_model_state(args.checkpoint, model)

    source_row = _source_ids(
        input_computation,
        tokenizer,
        source_len=args.source_len,
    )
    rng = jax.random.key(args.seed)
    metrics = _empty_metrics()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        samples_remaining = args.samples
        while samples_remaining > 0:
            batch_size = min(args.sample_batch_size, samples_remaining)
            samples_remaining -= batch_size
            rng, batch_rng = jax.random.split(rng)
            source_batch = jnp.asarray([source_row] * batch_size, dtype=jnp.int32)
            candidates, batch_metrics = sample_tensor_computations(
                model,
                batch_rng,
                input_computation,
                source_batch,
                tokenizer,
                grammar,
                target_len=args.target_len,
                outputs=args.verify_outputs,
                temperature=args.temperature,
            )
            _add_metrics(metrics, batch_metrics)
            for candidate in candidates:
                compact_json = json.dumps(json.loads(candidate.to_json_string()))
                output_file.write(f"{compact_json}\n")

    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
