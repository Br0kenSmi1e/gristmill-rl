from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flax import nnx

from .checkpoint import load_checkpoint
from .dataset import read_processed_jsonl
from .model import DirectOptimizerTransformer
from .trainer import DirectOptimizerTrainer, train_epochs


_STATIC_MODEL_FLAGS = (
    "source_len",
    "target_len",
    "scalar_value_min",
    "scalar_value_max",
    "d_model",
    "num_layers",
    "num_heads",
)
_OPTIONAL_MODEL_FLAGS = ("dropout", "init_scale")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    train_rows = read_processed_jsonl(args.train_dataset)
    valid_rows = (
        None if args.valid_dataset is None else read_processed_jsonl(args.valid_dataset)
    )
    test_rows = (
        None if args.test_dataset is None else read_processed_jsonl(args.test_dataset)
    )

    if args.checkpoint_in is None:
        model_kwargs = _fresh_model_kwargs(args, parser)
        model = DirectOptimizerTransformer(
            **model_kwargs,
            rngs=nnx.Rngs(args.seed),
        )
        trainer = DirectOptimizerTrainer(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        optimizer = trainer.init_optimizer(model)
        start_epoch = 0
        start_updates = 0
    else:
        checkpoint = load_checkpoint(
            args.checkpoint_in,
            expected_model_kwargs=_complete_static_model_kwargs(args),
        )
        model = checkpoint.model
        model_kwargs = dict(checkpoint.metadata["model_kwargs"])
        _validate_provided_model_kwargs(args, model_kwargs)
        trainer_kwargs = checkpoint.metadata.get("trainer_kwargs")
        if checkpoint.optimizer is not None and trainer_kwargs is not None:
            trainer = DirectOptimizerTrainer(**trainer_kwargs)
            optimizer = checkpoint.optimizer
        else:
            trainer = DirectOptimizerTrainer(
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            optimizer = trainer.init_optimizer(model)
        start_epoch = int(checkpoint.metadata["epoch"])
        start_updates = int(checkpoint.metadata["updates"])

    metrics = train_epochs(
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        model=model,
        trainer=trainer,
        optimizer=optimizer,
        epochs=args.epochs,
        source_len=int(model_kwargs["source_len"]),
        target_len=int(model_kwargs["target_len"]),
        scalar_value_min=int(model_kwargs["scalar_value_min"]),
        scalar_value_max=int(model_kwargs["scalar_value_max"]),
        seed=args.seed,
        checkpoint_out=args.checkpoint_out,
        start_epoch=start_epoch,
        start_updates=start_updates,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gristmill_symbolics.direct_optimizer.train",
    )
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--valid-dataset", type=Path)
    parser.add_argument("--test-dataset", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--epochs", type=_positive_int, default=1)
    parser.add_argument("--batch-size", required=True, type=_positive_int)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--source-len", type=_positive_int)
    parser.add_argument("--target-len", type=_positive_int)
    parser.add_argument("--scalar-value-min", type=int)
    parser.add_argument("--scalar-value-max", type=int)
    parser.add_argument("--d-model", type=_positive_int)
    parser.add_argument("--num-layers", type=_positive_int)
    parser.add_argument("--num-heads", type=_positive_int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--init-scale", type=float)
    return parser


def _fresh_model_kwargs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    missing = [
        f"--{name.replace('_', '-')}"
        for name in _STATIC_MODEL_FLAGS
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "fresh training requires static model flags: " + ", ".join(missing)
        )

    kwargs = {name: getattr(args, name) for name in _STATIC_MODEL_FLAGS}
    for name in _OPTIONAL_MODEL_FLAGS:
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    return kwargs


def _validate_provided_model_kwargs(
    args: argparse.Namespace,
    saved_kwargs: dict[str, Any],
) -> None:
    for name in (*_STATIC_MODEL_FLAGS, *_OPTIONAL_MODEL_FLAGS):
        value = getattr(args, name)
        if value is not None and saved_kwargs.get(name) != value:
            raise ValueError(f"mismatched model kwarg {name}")


def _complete_static_model_kwargs(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    kwargs = {name: getattr(args, name) for name in _STATIC_MODEL_FLAGS}
    if any(value is None for value in kwargs.values()):
        return None
    return kwargs


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
