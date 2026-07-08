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


def _make_eval_step(grammar: FlatDefinitionGrammar):
    def eval_step(model: nnx.Module, batch: dict[str, jax.Array]):
        return weighted_nll(model, batch, grammar, deterministic=True)

    return nnx.jit(eval_step)


def _evaluate_dataset(
    eval_step,
    model: nnx.Module,
    dataset: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, float | int]:
    weighted_nll_sum = 0.0
    weight_sum = 0.0
    num_batches = 0
    for batch in iter_supervised_batches(dataset, batch_size=batch_size):
        batch_nll, batch_weight = eval_step(model, _as_jax_batch(batch))
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the flat definition seq2seq model."
    )
    parser.add_argument("--train-arrays", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--valid-arrays", required=True)
    parser.add_argument("--valid-metadata", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
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
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.accumulate_steps <= 0:
        raise ValueError("accumulate_steps must be positive")
    if args.checkpoint_every_epochs < 0:
        raise ValueError("checkpoint_every_epochs must be non-negative")


def _train_epoch(
    trainer: SupervisedTrainer,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    dataset: dict[str, Any],
    *,
    batch_size: int,
    accumulate_steps: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    weighted_nll_sum = 0.0
    weight_sum = 0.0
    num_batches = 0
    num_updates = 0
    for batches in _iter_update_groups(
        dataset,
        batch_size=batch_size,
        accumulate_steps=accumulate_steps,
        rng=rng,
    ):
        metrics = trainer.update(model, optimizer, batches)
        weighted_nll_sum += float(metrics["weighted_nll_sum"])
        weight_sum += float(metrics["weight_sum"])
        num_batches += int(metrics["num_batches"])
        num_updates += 1
    mean_nll = weighted_nll_sum / weight_sum if weight_sum else 0.0
    return {
        "weighted_nll_sum": weighted_nll_sum,
        "weight_sum": weight_sum,
        "mean_nll": mean_nll,
        "num_batches": num_batches,
        "num_updates": num_updates,
    }


def _epoch_record(
    epoch: int,
    train_metrics: dict[str, float | int],
    valid_metrics: dict[str, float | int],
) -> dict[str, float | int]:
    record: dict[str, float | int] = {"epoch": epoch}
    for key, value in train_metrics.items():
        record[f"train_{key}"] = value
    for key, value in valid_metrics.items():
        record[f"valid_{key}"] = value
    return record


def _save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    name: str,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metadata: dict[str, Any],
) -> Path:
    checkpoint_path = Path(checkpoint_dir) / name
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        checkpoint_path,
        {
            "model": nnx.state(model),
            "optimizer": nnx.state(optimizer),
        },
        force=True,
    )
    if hasattr(checkpointer, "wait_until_finished"):
        checkpointer.wait_until_finished()
    (checkpoint_path / "metadata.json").write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n"
    )
    return checkpoint_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)

    train = load_preprocessed_supervised_dataset(
        args.train_arrays,
        args.train_metadata,
    )
    valid = load_preprocessed_supervised_dataset(
        args.valid_arrays,
        args.valid_metadata,
    )
    _validate_compatible_metadata(train["metadata"], valid["metadata"])

    tokenizer = _tokenizer_from_metadata(train["metadata"])
    grammar = FlatDefinitionGrammar(tokenizer)
    model = _build_model(args, train["metadata"], args.seed)
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
        wrt=nnx.Param,
    )
    trainer = SupervisedTrainer(grammar)
    eval_step = _make_eval_step(grammar)

    last_record: dict[str, float | int] | None = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(
            trainer,
            model,
            optimizer,
            train,
            batch_size=args.batch_size,
            accumulate_steps=args.accumulate_steps,
            rng=np.random.default_rng(args.seed + epoch),
        )
        valid_metrics = _evaluate_dataset(
            eval_step,
            model,
            valid,
            batch_size=args.batch_size,
        )
        last_record = _epoch_record(epoch, train_metrics, valid_metrics)
        print(json.dumps(last_record, sort_keys=True), flush=True)

        if (
            args.checkpoint_every_epochs
            and epoch % args.checkpoint_every_epochs == 0
        ):
            _save_checkpoint(
                args.checkpoint_dir,
                name=f"epoch-{epoch:04d}",
                model=model,
                optimizer=optimizer,
                metadata={
                    "epoch": epoch,
                    "kind": "periodic",
                    "metrics": last_record,
                },
            )

    _save_checkpoint(
        args.checkpoint_dir,
        name="final",
        model=model,
        optimizer=optimizer,
        metadata={
            "epoch": args.epochs,
            "kind": "final",
            "metrics": last_record,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
