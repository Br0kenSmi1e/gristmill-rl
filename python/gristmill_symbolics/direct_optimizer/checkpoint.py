from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import optax
import orbax.checkpoint as ocp
from flax import nnx

from .converter import CONVERTER_SCHEMA_VERSION
from .model import DirectOptimizerTransformer


CHECKPOINT_SCHEMA_VERSION = 1

_METADATA_FIELDS = {
    "schema_version",
    "converter_schema_version",
    "model_kwargs",
    "trainer_kwargs",
    "epoch",
    "updates",
    "last_train_loss",
    "last_valid_loss",
}
_STATIC_MODEL_KWARGS = (
    "source_len",
    "target_len",
    "scalar_value_min",
    "scalar_value_max",
    "d_model",
    "num_layers",
    "num_heads",
)
_OPTIONAL_MODEL_KWARGS = ("dropout", "init_scale")


@dataclass(frozen=True)
class DirectOptimizerCheckpoint:
    model: DirectOptimizerTransformer
    optimizer: nnx.Optimizer | None
    metadata: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    *,
    model: DirectOptimizerTransformer,
    optimizer: nnx.Optimizer,
    trainer: Any,
    epoch: int,
    updates: int,
    last_train_loss: float,
    last_valid_loss: float | None = None,
) -> None:
    _validate_save_metrics(
        epoch=epoch,
        updates=updates,
        last_train_loss=last_train_loss,
        last_valid_loss=last_valid_loss,
    )
    checkpoint_dir = Path(path)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = _make_sibling_temp_dir(checkpoint_dir, suffix=".tmp")
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "converter_schema_version": CONVERTER_SCHEMA_VERSION,
        "model_kwargs": model.model_kwargs(),
        "trainer_kwargs": trainer.constructor_kwargs(),
        "epoch": int(epoch),
        "updates": int(updates),
        "last_train_loss": float(last_train_loss),
        "last_valid_loss": (
            None if last_valid_loss is None else float(last_valid_loss)
        ),
    }

    try:
        _write_checkpoint_contents(
            staging_dir,
            metadata=metadata,
            model=model,
            optimizer=optimizer,
        )
        _publish_staged_checkpoint(staging_dir, checkpoint_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def load_checkpoint(
    path: str | Path,
    *,
    expected_model_kwargs: dict[str, Any] | None = None,
) -> DirectOptimizerCheckpoint:
    checkpoint_dir = Path(path)
    metadata = _load_metadata(checkpoint_dir)
    if expected_model_kwargs is not None:
        _validate_model_kwargs(
            metadata["model_kwargs"],
            expected_model_kwargs,
        )

    model = _restore_model(checkpoint_dir, metadata)
    optimizer = None
    trainer_kwargs = metadata.get("trainer_kwargs")
    if trainer_kwargs is not None:
        optimizer = _restore_optimizer(checkpoint_dir, model, trainer_kwargs)
    return DirectOptimizerCheckpoint(
        model=model,
        optimizer=optimizer,
        metadata=metadata,
    )


def load_model_for_inference(
    path: str | Path,
) -> tuple[DirectOptimizerTransformer, dict[str, Any]]:
    checkpoint_dir = Path(path)
    metadata = _load_metadata(checkpoint_dir)
    return _restore_model(checkpoint_dir, metadata), metadata


def _load_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata = json.loads(
        (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if set(metadata) != _METADATA_FIELDS:
        missing = sorted(_METADATA_FIELDS - set(metadata))
        extra = sorted(set(metadata) - _METADATA_FIELDS)
        raise ValueError(
            "invalid checkpoint metadata fields: "
            f"missing={missing}, extra={extra}"
        )
    if metadata["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version: {metadata['schema_version']}"
        )
    if metadata["converter_schema_version"] != CONVERTER_SCHEMA_VERSION:
        raise ValueError(
            "mismatched converter_schema_version: "
            f"{metadata['converter_schema_version']}"
        )
    if not isinstance(metadata["model_kwargs"], dict):
        raise ValueError("model_kwargs must be a mapping")
    if metadata["trainer_kwargs"] is not None and not isinstance(
        metadata["trainer_kwargs"],
        dict,
    ):
        raise ValueError("trainer_kwargs must be a mapping or null")
    _validate_save_metrics(
        epoch=metadata["epoch"],
        updates=metadata["updates"],
        last_train_loss=metadata["last_train_loss"],
        last_valid_loss=metadata["last_valid_loss"],
    )
    return metadata


def _restore_model(
    checkpoint_dir: Path,
    metadata: dict[str, Any],
) -> DirectOptimizerTransformer:
    model = DirectOptimizerTransformer(
        **metadata["model_kwargs"],
        rngs=nnx.Rngs(0),
    )
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(
        checkpoint_dir / "model_state",
        target=nnx.state(model),
    )
    nnx.update(model, restored_state)
    return model


def _restore_optimizer(
    checkpoint_dir: Path,
    model: DirectOptimizerTransformer,
    trainer_kwargs: dict[str, Any],
) -> nnx.Optimizer:
    tx = optax.adam(
        learning_rate=float(trainer_kwargs["learning_rate"]),
        b1=float(trainer_kwargs["b1"]),
        b2=float(trainer_kwargs["b2"]),
        eps=float(trainer_kwargs["eps"]),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(
        checkpoint_dir / "optimizer_state",
        target=nnx.state(optimizer),
    )
    nnx.update(optimizer, restored_state)
    return optimizer


def _write_checkpoint_contents(
    checkpoint_dir: Path,
    *,
    metadata: dict[str, Any],
    model: DirectOptimizerTransformer,
    optimizer: nnx.Optimizer,
) -> None:
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        checkpoint_dir / "model_state",
        nnx.state(model),
        force=True,
    )
    checkpointer.wait_until_finished()
    checkpointer.save(
        checkpoint_dir / "optimizer_state",
        nnx.state(optimizer),
        force=True,
    )
    checkpointer.wait_until_finished()


def _publish_staged_checkpoint(staging_dir: Path, checkpoint_dir: Path) -> None:
    backup_dir = None
    try:
        if checkpoint_dir.exists():
            backup_dir = _make_sibling_temp_dir(checkpoint_dir, suffix=".bak")
            shutil.rmtree(backup_dir)
            os.replace(checkpoint_dir, backup_dir)
        os.replace(staging_dir, checkpoint_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists():
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            os.replace(backup_dir, checkpoint_dir)
        raise
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)


def _make_sibling_temp_dir(path: Path, *, suffix: str) -> Path:
    return Path(
        tempfile.mkdtemp(
            prefix=f".{path.name}.",
            suffix=suffix,
            dir=path.parent,
        )
    )


def _validate_model_kwargs(
    saved_kwargs: dict[str, Any],
    expected_kwargs: dict[str, Any],
) -> None:
    for key in _STATIC_MODEL_KWARGS:
        if saved_kwargs.get(key) != expected_kwargs.get(key):
            raise ValueError(f"mismatched model kwarg {key}")
    for key in _OPTIONAL_MODEL_KWARGS:
        if key in expected_kwargs and saved_kwargs.get(key) != expected_kwargs[key]:
            raise ValueError(f"mismatched model kwarg {key}")


def _validate_save_metrics(
    *,
    epoch: int,
    updates: int,
    last_train_loss: float,
    last_valid_loss: float | None,
) -> None:
    _validate_nonnegative_int("epoch", epoch)
    _validate_nonnegative_int("updates", updates)
    _validate_finite_float("last_train_loss", last_train_loss)
    if last_valid_loss is not None:
        _validate_finite_float("last_valid_loss", last_valid_loss)


def _validate_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validate_finite_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
