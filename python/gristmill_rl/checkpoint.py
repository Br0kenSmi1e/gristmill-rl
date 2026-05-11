from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flax import nnx
import orbax.checkpoint as ocp

from gristmill_rl.features import FeatureConfig
from gristmill_rl.model import PolicyValueModel


SCHEMA_VERSION = 1
_MODEL_CLASS = "PolicyValueModel"


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    hidden_dim: int
    feature_config: FeatureConfig
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: PolicyValueModel
    feature_config: FeatureConfig
    metadata: dict[str, Any]


def _metadata_path(path: Path) -> Path:
    return path / "metadata.json"


def _state_path(path: Path) -> Path:
    return path / "state"


def _write_metadata(
    path: Path,
    *,
    feature_config: FeatureConfig,
    hidden_dim: int,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "class": _MODEL_CLASS,
            "hidden_dim": hidden_dim,
        },
        "features": asdict(feature_config),
        "metadata": metadata,
    }
    _metadata_path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def _read_metadata(path: Path) -> CheckpointMetadata:
    metadata_file = _metadata_path(path)
    if not metadata_file.exists():
        raise FileNotFoundError(f"checkpoint metadata not found: {metadata_file}")

    payload = json.loads(metadata_file.read_text())
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema_version {schema_version}")

    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("checkpoint metadata is missing model information")
    model_class = model.get("class")
    if model_class != _MODEL_CLASS:
        raise ValueError(f"Unsupported checkpoint model class {model_class!r}")

    hidden_dim = model.get("hidden_dim")
    if not isinstance(hidden_dim, int):
        raise ValueError("checkpoint metadata model.hidden_dim must be an integer")

    features = payload.get("features")
    if not isinstance(features, dict):
        raise ValueError("checkpoint metadata is missing feature configuration")

    user_metadata = payload.get("metadata", {})
    if not isinstance(user_metadata, dict):
        raise ValueError("checkpoint metadata.metadata must be an object")

    return CheckpointMetadata(
        schema_version=schema_version,
        hidden_dim=hidden_dim,
        feature_config=FeatureConfig(**features),
        metadata=user_metadata,
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: PolicyValueModel,
    feature_config: FeatureConfig,
    hidden_dim: int,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    if not isinstance(model, PolicyValueModel):
        raise TypeError("model must be a PolicyValueModel")

    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        if not overwrite:
            raise FileExistsError(f"checkpoint path already exists: {checkpoint_path}")
        if checkpoint_path.is_dir():
            shutil.rmtree(checkpoint_path)
        else:
            checkpoint_path.unlink()

    checkpoint_path.mkdir(parents=True)
    _, state = nnx.split(model.module)
    ocp.PyTreeCheckpointer().save(_state_path(checkpoint_path), state, force=True)
    _write_metadata(
        checkpoint_path,
        feature_config=feature_config,
        hidden_dim=hidden_dim,
        metadata=metadata or {},
    )


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    checkpoint_path = Path(path)
    checkpoint_metadata = _read_metadata(checkpoint_path)
    model = PolicyValueModel(hidden_dim=checkpoint_metadata.hidden_dim, rng_seed=0)
    _, abstract_state = nnx.split(model.module)
    restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_state)
    restored_state = ocp.PyTreeCheckpointer().restore(
        _state_path(checkpoint_path),
        item=abstract_state,
        restore_args=restore_args,
    )
    nnx.update(model.module, restored_state)
    return LoadedCheckpoint(
        model=model,
        feature_config=checkpoint_metadata.feature_config,
        metadata=checkpoint_metadata.metadata,
    )
