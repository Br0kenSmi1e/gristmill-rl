from __future__ import annotations

import json
import shutil
import uuid
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
    metadata: CheckpointMetadata


def _metadata_path(path: Path) -> Path:
    return path / "metadata.json"


def _state_path(path: Path) -> Path:
    return path / "state"


def _temporary_checkpoint_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")


def _backup_checkpoint_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.bak-{uuid.uuid4().hex}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _model_hidden_dim(model: PolicyValueModel) -> int:
    return int(model.module.state_1.out_features)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"checkpoint metadata.{field_name} must be a positive integer")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"checkpoint metadata.{field_name} must be a non-negative integer"
        )
    return value


def _validate_feature_config(feature_config: FeatureConfig) -> None:
    _positive_int(feature_config.max_candidates, "features.max_candidates")
    _non_negative_int(feature_config.max_left_terms, "features.max_left_terms")
    _non_negative_int(feature_config.max_right_terms, "features.max_right_terms")


def _validate_user_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata.metadata must be an object")
    return metadata


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

    try:
        payload = json.loads(metadata_file.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint metadata must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("checkpoint metadata must be an object")

    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema_version {schema_version}")

    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("checkpoint metadata.model must be an object")
    model_class = model.get("class")
    if model_class != _MODEL_CLASS:
        raise ValueError(f"Unsupported checkpoint model class {model_class!r}")

    hidden_dim = _positive_int(model.get("hidden_dim"), "model.hidden_dim")

    features = payload.get("features")
    if not isinstance(features, dict):
        raise ValueError("checkpoint metadata.features must be an object")
    feature_config = FeatureConfig(
        max_candidates=_positive_int(
            features.get("max_candidates"), "features.max_candidates"
        ),
        max_left_terms=_non_negative_int(
            features.get("max_left_terms"), "features.max_left_terms"
        ),
        max_right_terms=_non_negative_int(
            features.get("max_right_terms"), "features.max_right_terms"
        ),
    )

    user_metadata = payload.get("metadata", {})
    if not isinstance(user_metadata, dict):
        raise ValueError("checkpoint metadata.metadata must be an object")

    return CheckpointMetadata(
        schema_version=schema_version,
        hidden_dim=hidden_dim,
        feature_config=feature_config,
        metadata=user_metadata,
    )


def _publish_checkpoint(temp_path: Path, checkpoint_path: Path, *, overwrite: bool) -> None:
    backup_path: Path | None = None
    try:
        if checkpoint_path.exists():
            if not overwrite:
                raise FileExistsError(f"checkpoint path already exists: {checkpoint_path}")
            backup_path = _backup_checkpoint_path(checkpoint_path)
            checkpoint_path.rename(backup_path)

        temp_path.rename(checkpoint_path)
    except Exception:
        if backup_path is not None and backup_path.exists() and not checkpoint_path.exists():
            backup_path.rename(checkpoint_path)
        raise
    else:
        if backup_path is not None:
            _remove_path(backup_path)


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
    actual_hidden_dim = _model_hidden_dim(model)
    if hidden_dim != actual_hidden_dim:
        raise ValueError(
            f"hidden_dim {hidden_dim} does not match model hidden_dim {actual_hidden_dim}"
        )

    checkpoint_path = Path(path)
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(f"checkpoint path already exists: {checkpoint_path}")

    _validate_feature_config(feature_config)
    metadata_payload = _validate_user_metadata(metadata)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_checkpoint_path(checkpoint_path)
    try:
        temp_path.mkdir()
        _, state = nnx.split(model.module)
        ocp.PyTreeCheckpointer().save(_state_path(temp_path), state, force=True)
        _write_metadata(
            temp_path,
            feature_config=feature_config,
            hidden_dim=hidden_dim,
            metadata=metadata_payload,
        )
        _publish_checkpoint(temp_path, checkpoint_path, overwrite=overwrite)
    except Exception:
        _remove_path(temp_path)
        raise


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
        metadata=checkpoint_metadata,
    )
